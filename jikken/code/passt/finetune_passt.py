"""
PaSST (hear21passt) を使って、自前データ(クラスごとにフォルダ分け)で
ファインチューニングするスクリプト。EfficientAT版の finetune.py と同じ使い方。

事前準備:
    pip install hear21passt

想定するデータの並べ方 (1音声=1ラベルの単一クラス分類):

    DATA_DIR/
        dog/
            xxx.wav
            yyy.wav
        cat/
            zzz.wav
            ...

使い方:
    1. 下の「設定」セクションの DATA_DIR を自分のデータフォルダのパスに変更する。
    2. `python finetune_passt.py` を実行する。
    3. 学習済みモデルが CHECKPOINT_PATH に保存される。

PaSSTはAudioSetで事前学習済みのTransformerモデルです(論文: Koutini et al., 2022)。
デフォルトのarch "passt_s_swa_p16_128_ap476" は論文中でAudioSet mAP=0.476を達成した
公式の学習済み重みで、これをベースにファインチューニングするため、
論文と同等の強力な事前学習表現をそのまま活かせます。
"""

import argparse
import glob
import os
import warnings

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, random_split

from hear21passt.base import get_basic_model, get_model_passt

warnings.filterwarnings("ignore")

# ============================ 設定 (ここを変更) ============================
DATA_DIR = r"C:\Users\Owner\Downloads\sotsuken\jikken\Data\study_mineandothers\wavfiles_3seconds_cut"    # ← クラスごとにフォルダ分けされたデータのパス
CHECKPOINT_PATH =r"C:\Users\Owner\Downloads\sotsuken\jikken\code\passt\results\Data_mineandothers_3seconds_epochs50.pt"     # 学習後のモデルの保存先
EPOCHS = 50
BATCH_SIZE = 4                             # メモリが厳しい場合は 2 までさらに下げてもよい
LR = 1e-5                                  # Transformerのファインチューニングは小さめの学習率が安定
VAL_RATIO = 0.2
CLIP_SECONDS = 3                           # メモリが厳しい場合はさらに短く(3秒など)してもよい
ARCH = "passt_s_swa_p16_128_ap476"         # 論文でAudioSet mAP=0.476を達成したモデル
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
HISTORY_PLOT_PATH = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\passt\results\Data_mineandothers_3seconds_epochs50_training_curves_passt.png"
LOG_TXT_PATH = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\passt\results\Data_mineandothers_3seconds_epochs50_training_log_passt.txt"
OVERFIT_PATIENCE = 5
SAMPLE_RATE = 32000                        # PaSSTの事前学習時と同じ32kHz固定
# ===========================================================================


def pad_or_truncate(waveform, target_len):
    # PaSSTへ渡す全音声を同じ5秒長に統一する。
    if len(waveform) >= target_len:
        return waveform[:target_len]
    return np.concatenate([waveform, np.zeros(target_len - len(waveform), dtype=np.float32)])


class FolderAudioDataset(Dataset):
    def __init__(self, data_dir, sample_rate, clip_seconds, extensions, class_to_idx=None):
        self.sample_rate = sample_rate
        self.clip_len = int(sample_rate * clip_seconds)

        # フォルダ名を分類ラベルにし、ソート順でクラス番号を固定する。
        class_names = sorted([d for d in os.listdir(data_dir)
                               if os.path.isdir(os.path.join(data_dir, d))])
        if not class_names:
            raise RuntimeError(f"'{data_dir}' 内にクラスフォルダが見つかりません。")

        self.class_to_idx = class_to_idx or {c: i for i, c in enumerate(class_names)}
        self.classes = list(self.class_to_idx.keys())

        self.samples = []
        for class_name in class_names:
            class_dir = os.path.join(data_dir, class_name)
            files = []
            for ext in extensions:
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
        # 読み込み時に32 kHz・モノラルへ統一し、モデル前処理の前提を満たす。
        waveform, _ = librosa.core.load(path, sr=self.sample_rate, mono=True)
        waveform = pad_or_truncate(waveform.astype(np.float32), self.clip_len)
        return torch.from_numpy(waveform), label


class _SuppressPrint:
    """モデル読み込み時に出力される大量のログ・アーキテクチャ表示を抑制する"""
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
    # PaSSTの事前学習済み構成を作成し、出力クラス数だけ研究データに合わせる。
    with _SuppressPrint():
        model = get_basic_model(mode="logits")
        model.net = get_model_passt(arch=args.arch, n_classes=num_classes)
    model.to(device)
    return model


def run_epoch(model, loader, device, criterion, optimizer=None, epoch_label="", return_preds=False):
    # optimizer の有無で、重みを更新する学習と評価専用の検証を切り替える。
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    n_batches = len(loader)
    with torch.set_grad_enabled(is_train):
        for batch_idx, (waveform, label) in enumerate(loader, start=1):
            waveform, label = waveform.to(device), label.to(device)

            # PaSSTは生波形を受け取り、内部でスペクトログラム化してクラススコアを返す。
            with _SuppressPrint():
                logits = model(waveform)
            loss = criterion(logits, label)

            # 検証では勾配計算・パラメータ更新を行わない。
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

            print(f"  [{epoch_label}] batch {batch_idx}/{n_batches} "
                  f"loss={loss.item():.4f}", flush=True)

    if return_preds:
        return total_loss / total, correct / total, all_preds, all_labels
    return total_loss / total, correct / total


def plot_training_curves(history, save_path):
    # 学習と検証の損失・Accuracy・macro F1を並べ、過学習を確認しやすくする。
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

    fig.suptitle("Training Curves (PaSST)")
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

    full_dataset = FolderAudioDataset(args.data_dir, args.sample_rate, args.clip_seconds,
                                       args.audio_extensions)
    num_classes = len(full_dataset.classes)

    val_size = max(1, int(len(full_dataset) * args.val_ratio))
    train_size = len(full_dataset) - val_size
    # 固定シードで分割し、学習結果の再現性を確保する。
    train_set, val_set = random_split(full_dataset, [train_size, val_size],
                                       generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

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
        train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer,
                                           epoch_label=f"Epoch {epoch}/{args.epochs} train")
        val_loss, val_acc, val_preds, val_labels = run_epoch(
            model, val_loader, device, criterion, optimizer=None,
            epoch_label=f"Epoch {epoch}/{args.epochs} val", return_preds=True)
        # クラスごとに同じ重みを置くmacro F1を、Accuracyと併せて記録する。
        val_f1_macro = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        log_line = (f"[Epoch {epoch}/{args.epochs}] "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1_macro:.3f}")
        print(log_line)
        log_file.write(log_line + "\n")
        log_file.flush()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1_macro"].append(val_f1_macro)

        # 現行の保存条件は検証Accuracyの最高値。最終エポックの重みとは限らない。
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            checkpoint_dir = os.path.dirname(args.checkpoint_path)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "net_state_dict": model.net.state_dict(),
                "classes": full_dataset.classes,
                "arch": args.arch,
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
    parser = argparse.ArgumentParser(description='PaSSTで自前データをファインチューニング')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--checkpoint_path', type=str, default=CHECKPOINT_PATH)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--val_ratio', type=float, default=VAL_RATIO)
    parser.add_argument('--clip_seconds', type=float, default=CLIP_SECONDS)
    parser.add_argument('--arch', type=str, default=ARCH)
    parser.add_argument('--sample_rate', type=int, default=SAMPLE_RATE)
    parser.add_argument('--audio_extensions', type=tuple, default=AUDIO_EXTENSIONS)
    parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available())
    parser.add_argument('--history_plot_path', type=str, default=HISTORY_PLOT_PATH)
    parser.add_argument('--log_txt_path', type=str, default=LOG_TXT_PATH)
    parser.add_argument('--overfit_patience', type=int, default=OVERFIT_PATIENCE)

    args = parser.parse_args()
    main(args)
