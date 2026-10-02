"""
FAST-SDE (https://github.com/JiangWAV/FAST-SDE) のモデル本体を使い、
自前データ(距離ごとにフォルダ分け)で音源との距離推定モデルを学習するスクリプト。
finetune.py と同じ使い方のスタイルにしています。

事前準備:
    pip install torchlibrosa

想定するデータの並べ方 (フォルダ名がそのまま距離[cm]になる):

    DATA_DIR/
        26/
            xxx.wav
            yyy.wav
        30/
            zzz.wav
            ...
        100/
            ...

    フォルダ名は数値(距離をcm単位にしたもの)にしてください。小数点も使えます(例: "26.5")。

使い方:
    1. 下の「設定」セクションの DATA_DIR を自分のデータフォルダのパスに変更する。
    2. `python train_distance.py` を実行する。
    3. 学習済みモデルが CHECKPOINT_PATH に保存される。

【注意】
    このモデルは「距離推定」という回帰タスク用です。シャカパチかどうかを判定する
    finetune.py (分類タスク)とは別物で、両方を組み合わせて使う想定です
    (例: finetune.pyでシャカパチを検知 → train_distance.pyのモデルでその音がどれくらいの
    距離から鳴ったかを推定する、という2段構え)。
"""

import argparse
import glob
import os

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from fastsde_model.model import SeldNetSubbandFast

# ============================ 設定 (ここを変更) ============================
DATA_DIR = r"C:\Users\Owner\Downloads\sotsuken\jikken\Data\Distance_Data\5seconds"                 # ← 距離ごとにフォルダ分けされたデータのパス
CHECKPOINT_PATH =r"C:\Users\Owner\Downloads\sotsuken\jikken\code\Fast_SDE\results\pt\Distance_Data_epoch_50_add60,120,180_clip_5.pt"       # 学習後のモデルの保存先
EPOCHS = 50
BATCH_SIZE = 4
LR = 1e-3
VAL_RATIO = 0.2
CLIP_SECONDS = 5.0                         # 各音声をこの長さに揃える(FAST-SDEのデフォルトに合わせて3秒)
SAMPLE_RATE = 16000                        # FAST-SDEはデフォルトで16kHzを想定
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
HISTORY_PLOT_PATH = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\Fast_SDE\results\png\Distance_Data_training_curves_50epochs_add60,120,180_clip_5.png"  # 学習曲線の保存先
LOG_TXT_PATH = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\Fast_SDE\results\txt\Distance_Data_training_log_50epochs_add60,120,180_clip_5.txt"  # 学習ログの保存先
OVERFIT_PATIENCE = 10
EXCLUDE_DIR_NAMES = {"wavfiles"}   # パス中にこの名前のフォルダが含まれる場合、その配下は読み込み対象から除外する
# ===========================================================================


def pad_or_truncate(waveform, target_len):
    if len(waveform) >= target_len:
        return waveform[:target_len]
    return np.concatenate([waveform, np.zeros(target_len - len(waveform), dtype=np.float32)])
 
 
class DistanceFolderDataset(Dataset):
    """
    data_dir 以下のどこかの階層に「距離[cm]を表す数値名のフォルダ」がある、という
    並びのデータセット。マイク種別・シャッフル種別などのフォルダが距離フォルダの
    上や下にあっても構わない(例: data_dir/PCM-100/200/wavfiles/Deal/xxx.wav でもOK)。
    パス中に数値フォルダが複数ある場合は、data_dirに近い方(最初に見つかったもの)を
    距離ラベルとして採用する。
    """
 
    def __init__(self, data_dir, sample_rate, clip_seconds, extensions, exclude_dir_names=None):
        self.sample_rate = sample_rate
        self.clip_len = int(sample_rate * clip_seconds)
        exclude_dir_names = exclude_dir_names or set()
 
        self.samples = []
        self.distances_seen = []
        skipped_count = 0
        excluded_count = 0
 
        for root, dirs, files in os.walk(data_dir):
            # exclude_dir_names に含まれるフォルダ名は、配下ごと探索対象から除外する
            # (例: "wavfiles"を除外して、"wavfiles5seconds_cut"は別名なので除外されない)
            dirs[:] = [d for d in dirs if d not in exclude_dir_names]
 
            matched_files = [f for f in files if os.path.splitext(f)[1].lower() in extensions]
            if not matched_files:
                continue
 
            rel_path = os.path.relpath(root, data_dir)
            path_parts = [] if rel_path == "." else rel_path.split(os.sep)
 
            distance_cm = None
            for part in path_parts:
                try:
                    distance_cm = float(part)
                    break
                except ValueError:
                    continue
 
            if distance_cm is None:
                skipped_count += len(matched_files)
                continue
 
            for fname in matched_files:
                self.samples.append((os.path.join(root, fname), distance_cm))
            self.distances_seen.append(distance_cm)
 
        if skipped_count > 0:
            print(f"警告: 距離を表す数値フォルダが見つからなかったため、{skipped_count}件の音声をスキップしました。")
 
        if not self.samples:
            raise RuntimeError(f"'{data_dir}' 内に音声ファイルが見つかりません。")
 
        print(f"読み込んだデータ: {len(self.samples)} 件, "
              f"距離ラベル: {sorted(set(self.distances_seen))} cm")
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        path, distance_cm = self.samples[idx]
        waveform, _ = librosa.core.load(path, sr=self.sample_rate, mono=True)
        waveform = pad_or_truncate(waveform.astype(np.float32), self.clip_len)
        return torch.from_numpy(waveform), np.float32(distance_cm)
 
 
def build_model():
    # README記載の "UltraFast" 構成(軽量版)を使用
    return SeldNetSubbandFast(
        features_set="all",
        n_subbands=6,
        c_mid=16,
        n_blocks=2,
        fuse_c=48,
        use_gru=False,
        att_conf="Nothing",
    )
 
 
def run_epoch(model, loader, device, criterion, optimizer=None, epoch_label=""):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
 
    total_loss, total_abs_error, total = 0.0, 0.0, 0
    n_batches = len(loader)
    with torch.set_grad_enabled(is_train):
        for batch_idx, (waveform, distance) in enumerate(loader, start=1):
            waveform, distance = waveform.to(device), distance.to(device)
 
            pred = model(waveform).squeeze(-1)
            loss = criterion(pred, distance)
 
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
 
            total_loss += loss.item() * waveform.size(0)
            total_abs_error += torch.abs(pred - distance).sum().item()
            total += waveform.size(0)
 
            print(f"  [{epoch_label}] batch {batch_idx}/{n_batches} loss={loss.item():.2f}", flush=True)
 
    return total_loss / total, total_abs_error / total  # (MSE, MAE[cm])
 
 
def moving_average(values, window):
    """単純移動平均。先頭付近はその時点までの平均(窓が満たない分は短い窓で計算)で埋める。"""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i + 1]) / (i - start + 1))
    return result
 
 
def plot_training_curves(history, save_path, smoothing_window=5):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
 
    val_loss_smooth = moving_average(history["val_loss"], smoothing_window)
    val_mae_smooth = moving_average(history["val_mae"], smoothing_window)
 
    # 生の値は薄く・細く表示し、傾向を示す移動平均線を太く前面に出す
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", markersize=3,
                 linewidth=1, label="train")
    axes[0].plot(history["epoch"], history["val_loss"], marker="o", markersize=3,
                 linewidth=0.8, alpha=0.3, color="orange", label="val(生の値)")
    axes[0].plot(history["epoch"], val_loss_smooth, linewidth=2.2, color="darkorange",
                 label=f"val(移動平均, window={smoothing_window})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Loss")
    axes[0].legend(fontsize=8)
 
    axes[1].plot(history["epoch"], history["train_mae"], marker="o", markersize=3,
                 linewidth=1, label="train")
    axes[1].plot(history["epoch"], history["val_mae"], marker="o", markersize=3,
                 linewidth=0.8, alpha=0.3, color="orange", label="val(生の値)")
    axes[1].plot(history["epoch"], val_mae_smooth, linewidth=2.2, color="darkorange",
                 label=f"val(移動平均, window={smoothing_window})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE [cm]")
    axes[1].set_title("Mean Absolute Error (distance)")
    axes[1].legend(fontsize=8)
 
    fig.suptitle("Distance Estimation Training Curves (FAST-SDE)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
 
 
def check_overfitting(history, patience):
    val_losses = history["val_loss"]
    train_losses = history["train_loss"]
    if len(val_losses) <= patience:
        return
 
    best_val_loss = min(val_losses)
    best_epoch = val_losses.index(best_val_loss)
    epochs_since_best = len(val_losses) - 1 - best_epoch
    train_still_improving = train_losses[-1] < train_losses[best_epoch]
 
    if epochs_since_best >= patience and train_still_improving:
        print(f"\n⚠ 過学習の可能性があります: val_lossは{best_epoch + 1}エポック目("
              f"{best_val_loss:.4f})以降、{epochs_since_best}エポック改善していません。")
    else:
        print(f"\n過学習の兆候は今のところ見られません"
              f"(val_lossの最良値: {best_epoch + 1}エポック目の{best_val_loss:.4f})。")
 
 
def main(args):
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    print(f"使用デバイス: {device}")
 
    full_dataset = DistanceFolderDataset(args.data_dir, args.sample_rate, args.clip_seconds,
                                          args.audio_extensions, exclude_dir_names=EXCLUDE_DIR_NAMES)
 
    val_size = max(1, int(len(full_dataset) * args.val_ratio))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size],
                                       generator=torch.Generator().manual_seed(42))
 
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
 
    model = build_model().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
 
    best_val_mae = float("inf")
    history = {"epoch": [], "train_loss": [], "train_mae": [], "val_loss": [], "val_mae": []}
 
    log_dir = os.path.dirname(args.log_txt_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    log_file = open(args.log_txt_path, "w", encoding="utf-8")
 
    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae = run_epoch(model, train_loader, device, criterion, optimizer,
                                           epoch_label=f"Epoch {epoch}/{args.epochs} train")
        val_loss, val_mae = run_epoch(model, val_loader, device, criterion, optimizer=None,
                                       epoch_label=f"Epoch {epoch}/{args.epochs} val")
 
        log_line = (f"[Epoch {epoch}/{args.epochs}] "
                    f"train_loss={train_loss:.4f} train_mae={train_mae:.2f}cm | "
                    f"val_loss={val_loss:.4f} val_mae={val_mae:.2f}cm")
        print(log_line)
        log_file.write(log_line + "\n")
        log_file.flush()
 
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_mae"].append(train_mae)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
 
        if val_mae <= best_val_mae:
            best_val_mae = val_mae
            checkpoint_dir = os.path.dirname(args.checkpoint_path)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "sample_rate": args.sample_rate,
                "clip_seconds": args.clip_seconds,
            }, args.checkpoint_path)
            save_line = f"  -> 検証MAEが改善したので '{args.checkpoint_path}' に保存しました。"
            print(save_line)
            log_file.write(save_line + "\n")
            log_file.flush()
 
    log_file.close()
    print(f"\n学習完了。最良の検証MAE: {best_val_mae:.2f} cm")
    print(f"学習ログを '{args.log_txt_path}' に保存しました。")
 
    plot_training_curves(history, args.history_plot_path)
    print(f"学習曲線を '{args.history_plot_path}' に保存しました。")
    check_overfitting(history, args.overfit_patience)
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FAST-SDEモデルで距離推定をフォルダ分けデータから学習')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--checkpoint_path', type=str, default=CHECKPOINT_PATH)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--val_ratio', type=float, default=VAL_RATIO)
    parser.add_argument('--clip_seconds', type=float, default=CLIP_SECONDS)
    parser.add_argument('--sample_rate', type=int, default=SAMPLE_RATE)
    parser.add_argument('--audio_extensions', type=tuple, default=AUDIO_EXTENSIONS)
    parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available())
    parser.add_argument('--history_plot_path', type=str, default=HISTORY_PLOT_PATH)
    parser.add_argument('--log_txt_path', type=str, default=LOG_TXT_PATH)
    parser.add_argument('--overfit_patience', type=int, default=OVERFIT_PATIENCE)
 
    args = parser.parse_args()
    main(args)
 
