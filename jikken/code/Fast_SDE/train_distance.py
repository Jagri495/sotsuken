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
DATA_DIR = "distance_data"                 # ← 距離ごとにフォルダ分けされたデータのパス
CHECKPOINT_PATH = "distance_model.pt"      # 学習後のモデルの保存先
EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-3
VAL_RATIO = 0.2
CLIP_SECONDS = 3.0                         # 各音声をこの長さに揃える(FAST-SDEのデフォルトに合わせて3秒)
SAMPLE_RATE = 16000                        # FAST-SDEはデフォルトで16kHzを想定
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
HISTORY_PLOT_PATH = "distance_training_curves.png"
LOG_TXT_PATH = "distance_training_log.txt"
OVERFIT_PATIENCE = 10
# ===========================================================================


def pad_or_truncate(waveform, target_len):
    # ネットワークが固定長入力を前提とするため、長さを3秒相当に統一する。
    if len(waveform) >= target_len:
        return waveform[:target_len]
    return np.concatenate([waveform, np.zeros(target_len - len(waveform), dtype=np.float32)])


class DistanceFolderDataset(Dataset):
    """
    data_dir/距離[cm]/音声ファイル という並びのデータセット。
    フォルダ名をfloatにパースしてラベル(距離)として使う。
    """

    def __init__(self, data_dir, sample_rate, clip_seconds, extensions):
        self.sample_rate = sample_rate
        self.clip_len = int(sample_rate * clip_seconds)

        # 直下の数値名フォルダだけを距離ラベル候補として扱う。
        distance_folders = sorted([d for d in os.listdir(data_dir)
                                    if os.path.isdir(os.path.join(data_dir, d))])
        if not distance_folders:
            raise RuntimeError(f"'{data_dir}' 内に距離名のフォルダが見つかりません。")

        self.samples = []
        self.distances_seen = []
        for folder_name in distance_folders:
            # フォルダ名そのものを連続値の正解距離[cm]へ変換する。
            try:
                distance_cm = float(folder_name)
            except ValueError:
                print(f"警告: フォルダ名 '{folder_name}' を数値(距離)として解釈できないためスキップします。")
                continue

            folder_path = os.path.join(data_dir, folder_name)
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
            for f in files:
                self.samples.append((f, distance_cm))
            self.distances_seen.append(distance_cm)

        if not self.samples:
            raise RuntimeError(f"'{data_dir}' 内に音声ファイルが見つかりません。")

        print(f"読み込んだデータ: {len(self.samples)} 件, "
              f"距離ラベル: {sorted(set(self.distances_seen))} cm")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, distance_cm = self.samples[idx]
        # 読み込み時の再標本化・モノラル化で録音形式の差を入力前に吸収する。
        waveform, _ = librosa.core.load(path, sr=self.sample_rate, mono=True)
        waveform = pad_or_truncate(waveform.astype(np.float32), self.clip_len)
        return torch.from_numpy(waveform), np.float32(distance_cm)


def build_model():
    # README記載の "UltraFast" 構成(軽量版)を使用
    # 学習時と予測時で必ず同じ構成を生成する。重みだけがチェックポイントから復元される。
    return SeldNetSubbandFast(
        features_set="all",
        n_subbands=6,
        c_mid=16,
        n_blocks=2,
        fuse_c=48,
        use_gru=False,
        att_conf="Nothing",
    )


def run_epoch(model, loader, device, criterion, optimizer=None):
    # optimizer の有無で、勾配更新を伴う学習か評価専用かを切り替える。
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_abs_error, total = 0.0, 0.0, 0
    with torch.set_grad_enabled(is_train):
        for waveform, distance in loader:
            waveform, distance = waveform.to(device), distance.to(device)

            # モデル出力は [バッチ, 1] の距離予測なので、末尾の1次元だけ除く。
            pred = model(waveform).squeeze(-1)
            loss = criterion(pred, distance)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * waveform.size(0)
            # MSEとは別に、解釈しやすいセンチメートル単位の絶対誤差を集計する。
            total_abs_error += torch.abs(pred - distance).sum().item()
            total += waveform.size(0)

    return total_loss / total, total_abs_error / total  # (MSE, MAE[cm])


def plot_training_curves(history, save_path):
    # 損失とMAEを分けて描き、最適化の推移と実用上の距離誤差を併記する。
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="train")
    axes[0].plot(history["epoch"], history["val_loss"], marker="o", label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_mae"], marker="o", label="train")
    axes[1].plot(history["epoch"], history["val_mae"], marker="o", label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE [cm]")
    axes[1].set_title("Mean Absolute Error (distance)")
    axes[1].legend()

    fig.suptitle("Distance Estimation Training Curves (FAST-SDE)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def check_overfitting(history, patience):
    # 学習損失だけ下がり検証損失が改善しない状態を、警告対象として確認する。
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
                                          args.audio_extensions)

    val_size = max(1, int(len(full_dataset) * args.val_ratio))
    train_size = len(full_dataset) - val_size
    # 乱数シードを固定し、同じ入力集合なら同じ分割で再実験できるようにする。
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
        train_loss, train_mae = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_mae = run_epoch(model, val_loader, device, criterion, optimizer=None)

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

        # 最終エポックではなく、検証MAEが最小だった時点の重みを残す。
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
