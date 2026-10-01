"""
自前データ(クラスごとにフォルダ分け)でEfficientATモデルをファインチューニングするスクリプト。

想定するデータの並べ方 (1音声=1ラベルの単一クラス分類):

    DATA_DIR/
        dog/
            xxx.wav
            yyy.wav
        cat/
            zzz.wav
            ...
        ...

使い方:
    1. 下の「設定」セクションの DATA_DIR を自分のデータフォルダのパスに変更する。
    2. `python finetune.py` を実行する。
    3. 学習済みモデルが CHECKPOINT_PATH (デフォルト: finetuned_model.pt) に保存される。

コマンドライン引数でも上書きできる:
    python finetune.py --data_dir /path/to/data --epochs 30 --model_name mn10_as
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
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset, random_split

from helpers.utils import NAME_TO_WIDTH
from models.dymn.model import get_model as get_dymn
from models.mn.model import get_model as get_mobilenet
from models.preprocess import AugmentMelSTFT

# ============================ 設定 (ここを変更) ============================
DATA_DIR = r"C:\Users\Owner\Downloads\sotsuken\jikken\Data\study_mineandothers\wavfiles_5seconds_cut"       # ← クラスごとにフォルダ分けされたデータのパス
CHECKPOINT_PATH = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\EfficientAT-main\results\pt\Data_mineandothers_5seconds_finetuned_model_epoch50.pt"     # 学習後のモデルの保存先
EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-4
VAL_RATIO = 0.2                            # 検証用に取り分ける割合
CLIP_SECONDS = 5                           # 各音声をこの長さに揃える(切り詰め/ゼロ埋め)
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
HISTORY_PLOT_PATH =  r"C:\Users\Owner\Downloads\sotsuken\jikken\code\EfficientAT-main\results\png\Data_mineandothers_5seconds_training_curves_epoch50.png" # 学習曲線の保存先
LOG_TXT_PATH = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\EfficientAT-main\results\txt\Data_mineandothers_5seconds _training_log_epoch50.txt" # 学習ログの保存先
OVERFIT_PATIENCE = 5                       # 過学習判定: val_lossが改善しないエポックが何回続いたら警告するか
# =========================================================================


def pad_or_truncate(waveform, target_len):
    # 全サンプルを同じ長さにして、バッチ化できる入力形状へそろえる。
    if len(waveform) >= target_len:
        return waveform[:target_len]
    return np.concatenate([waveform, np.zeros(target_len - len(waveform), dtype=np.float32)])
 
 
class FolderAudioDataset(Dataset):
    """
    data_dir/クラス名/音声ファイル という並びのデータセットを読み込むクラス。
    """
 
    def __init__(self, data_dir, sample_rate, clip_seconds, class_to_idx=None):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.clip_len = int(sample_rate * clip_seconds)
 
        # フォルダ名をクラス名として読み、ソート順で数値ラベルを安定させる。
        class_names = sorted([d for d in os.listdir(data_dir)
                               if os.path.isdir(os.path.join(data_dir, d))])
        if not class_names:
            raise RuntimeError(f"'{data_dir}' 内にクラスフォルダが見つかりません。")
 
        # 評価時に既存の対応表を渡せば、学習時と同じクラス番号を維持できる。
        self.class_to_idx = class_to_idx or {c: i for i, c in enumerate(class_names)}
        self.classes = list(self.class_to_idx.keys())
 
        self.samples = []
        for class_name in class_names:
            class_dir = os.path.join(data_dir, class_name)
            files = []
            for ext in AUDIO_EXTENSIONS:
                files.extend(glob.glob(os.path.join(class_dir, f"*{ext}")))
            for f in files:
                self.samples.append((f, self.class_to_idx[class_name]))
 
        if not self.samples:
            raise RuntimeError(f"'{data_dir}' 内に音声ファイルが見つかりません。")
 
        print(f"読み込んだデータ: {len(self.samples)} 件, {len(self.classes)} クラス: {self.classes}")
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # ファイル形式・元の標本化周波数を統一して、生波形とクラス番号を返す。
        waveform, _ = librosa.core.load(path, sr=self.sample_rate, mono=True)
        waveform = pad_or_truncate(waveform.astype(np.float32), self.clip_len)
        return torch.from_numpy(waveform), label
 
 
class _SuppressPrint:
    """モデル読み込み時に出力される大量のアーキテクチャ表示を抑制する"""
    def __enter__(self):
        import sys
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        return self
 
    def __exit__(self, *args):
        import sys
        sys.stdout.close()
        sys.stdout = self._stdout
 
 
def build_model(args, num_classes, device):
    # AudioSet事前学習済みの特徴抽出器を読み、最後の分類ヘッドだけ対象クラス数に合わせる。
    with _SuppressPrint():
        if args.model_name.startswith("dymn"):
            model = get_dymn(width_mult=NAME_TO_WIDTH(args.model_name),
                              pretrained_name=args.model_name,
                              num_classes=num_classes)
        else:
            model = get_mobilenet(width_mult=NAME_TO_WIDTH(args.model_name),
                                   pretrained_name=args.model_name,
                                   head_type=args.head_type,
                                   num_classes=num_classes)
    model.to(device)
    return model
 
 
def run_epoch(model, mel, loader, device, criterion, optimizer=None, return_preds=False):
    # optimizer があれば学習、なければ検証。return_preds はF1計算用の予測収集を有効にする。
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    mel.train() if is_train else mel.eval()
 
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.set_grad_enabled(is_train):
        for waveform, label in loader:
            waveform, label = waveform.to(device), label.to(device)
 
            # mel(waveform) は (batch, n_mels, time) を返すので、モデル入力用に channel次元(=1)を追加する
            # 生波形をメルスペクトログラムへ変換してCNN入力の [B, C, Mel, Time] にする。
            spec = mel(waveform).unsqueeze(1)
            logits, _ = model(spec)
            loss = criterion(logits, label)
 
            # 検証時は勾配を更新せず、学習時だけ誤差逆伝播を行う。
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
 
            total_loss += loss.item() * waveform.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += waveform.size(0)
 
            if return_preds:
                all_preds.extend(preds.detach().cpu().numpy().tolist())
                all_labels.extend(label.detach().cpu().numpy().tolist())
 
    if return_preds:
        return total_loss / total, correct / total, all_preds, all_labels
    return total_loss / total, correct / total
 
 
def plot_training_curves(history, save_path):
    # Accuracyだけでなく、不均衡クラスの影響を受けにくいmacro F1も可視化する。
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.5))
 
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="train")
    axes[0].plot(history["epoch"], history["val_loss"], marker="o", label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
 
    axes[1].plot(history["epoch"], history["train_acc"], marker="o", label="train")
    axes[1].plot(history["epoch"], history["val_acc"], marker="o", label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Accuracy")
    axes[1].legend()
 
    axes[2].plot(history["epoch"], history["val_f1_macro"], marker="o", color="green", label="val (macro F1)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 (macro)")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("F1 Score")
    axes[2].legend()
 
    fig.suptitle("Training Curves")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
 
 
def check_overfitting(history, patience):
    """train_lossは下がり続けているのに、val_lossが一定期間改善していない場合に警告する簡易チェック。"""
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
              f"{best_val_loss:.4f})以降、{epochs_since_best}エポック改善していませんが、"
              f"train_lossは下がり続けています(現在{train_losses[-1]:.4f})。")
        print("  対策案: エポック数を減らす / 学習率を下げる / データ拡張を増やす / "
              "モデルをより単純にする、などを検討してください。")
    else:
        print(f"\n過学習の兆候は今のところ見られません"
              f"(val_lossの最良値: {best_epoch + 1}エポック目の{best_val_loss:.4f})。")
 
 
def main(args):
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    print(f"使用デバイス: {device}")
 
    full_dataset = FolderAudioDataset(args.data_dir, args.sample_rate, args.clip_seconds)
    num_classes = len(full_dataset.classes)
 
    val_size = max(1, int(len(full_dataset) * args.val_ratio))
    train_size = len(full_dataset) - val_size
    # 乱数シードを固定し、同じデータなら同じ学習・検証分割を再現できるようにする。
    train_set, val_set = random_split(full_dataset, [train_size, val_size],
                                       generator=torch.Generator().manual_seed(42))
 
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
 
    # 学習と検証で共有する前処理器。設定値は推論時にも一致させる必要がある。
    mel = AugmentMelSTFT(n_mels=args.n_mels, sr=args.sample_rate,
                          win_length=args.window_size, hopsize=args.hop_size)
    mel.to(device)
 
    model = build_model(args, num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
 
    best_val_acc = 0.0
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "val_f1_macro": []}
 
    log_dir = os.path.dirname(args.log_txt_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    log_file = open(args.log_txt_path, "w", encoding="utf-8")
 
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, mel, train_loader, device, criterion, optimizer)
        val_loss, val_acc, val_preds, val_labels = run_epoch(
            model, mel, val_loader, device, criterion, optimizer=None, return_preds=True)
        # 各クラスのF1を等しく平均し、クラス数の偏りを隠さない指標にする。
        val_f1_macro = f1_score(val_labels, val_preds, average="macro", zero_division=0)
 
        log_line = (f"[Epoch {epoch}/{args.epochs}] "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1_macro:.3f}")
        print(log_line)
        log_file.write(log_line + "\n")
        log_file.flush()  # 途中でクラッシュしても、ここまでのログが確実にファイルに残るように
 
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1_macro"].append(val_f1_macro)
 
        # 現行仕様ではmacro F1ではなく検証Accuracyが最高の重みを保存する。
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            checkpoint_dir = os.path.dirname(args.checkpoint_path)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": full_dataset.classes,
                "model_name": args.model_name,
                "head_type": args.head_type,
                "sample_rate": args.sample_rate,
            }, args.checkpoint_path)
            save_line = f"  -> 検証精度が改善したので '{args.checkpoint_path}' に保存しました。"
            print(save_line)
            log_file.write(save_line + "\n")
            log_file.flush()
 
    log_file.close()
    print(f"\n学習完了。最良の検証精度: {best_val_acc:.3f}")
    print(f"学習ログを '{args.log_txt_path}' に保存しました。")
 
    plot_training_curves(history, args.history_plot_path)
    print(f"学習曲線を '{args.history_plot_path}' に保存しました。")
    check_overfitting(history, args.overfit_patience)
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='フォルダ分けデータで単一クラス分類のファインチューニング')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--checkpoint_path', type=str, default=CHECKPOINT_PATH)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--val_ratio', type=float, default=VAL_RATIO)
    parser.add_argument('--clip_seconds', type=float, default=CLIP_SECONDS)
 
    parser.add_argument('--model_name', type=str, default='mn10_as')
    parser.add_argument('--head_type', type=str, default='mlp')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--history_plot_path', type=str, default=HISTORY_PLOT_PATH)
    parser.add_argument('--log_txt_path', type=str, default=LOG_TXT_PATH)
    parser.add_argument('--overfit_patience', type=int, default=OVERFIT_PATIENCE)
 
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_mels', type=int, default=128)
 
    args = parser.parse_args()
    main(args)
 
