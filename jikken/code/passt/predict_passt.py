"""
finetune_passt.py で作成したPaSSTのファインチューニング済みモデルを使って、
フォルダ内の音声ファイルを一括で予測するスクリプト。EfficientAT版の
predict_finetuned.py と同じ使い方・同じ混同行列機能を持つ。

使い方:
    1. 下の「設定」セクションを変更する。
    2. `python predict_passt.py` を実行する。

【混同行列について】
    AUDIO_FOLDER が学習時と同じ「クラス名ごとのフォルダ分け」になっている場合、
    フォルダ名を正解ラベルとみなして精度を自動計算し、
    混同行列の画像 (CONFUSION_MATRIX_PATH) を保存します。
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
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

from hear21passt.base import get_basic_model, get_model_passt

warnings.filterwarnings("ignore")

# ============================ 設定 (ここを変更) ============================
CHECKPOINT_PATH = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\passt\results\wavfiles_5seconds_Zoom_test_cut_epochs30.pt"
AUDIO_FOLDER = r"C:\Users\Owner\Downloads\sotsuken\jikken\Data\Data_other\Data_D\wavfile_5seconds_cut"
RESULT_CSV = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\passt\results\Zoom_test_cut_DataD_predictions_passt.csv"
CONFUSION_MATRIX_PATH = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\passt\results\Zoom_test_cut_DataD_confusion_matrix_passt.png"
METRICS_CSV_PATH = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\passt\results\Zoom_test_cut_DataD_metrics_passt.csv"
CLIP_SECONDS = 10
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
# ===========================================================================


def pad_or_truncate(waveform, target_len):
    # 学習時と同じ固定長にそろえ、入力サイズの差をなくす。
    if len(waveform) >= target_len:
        return waveform[:target_len]
    return np.concatenate([waveform, np.zeros(target_len - len(waveform), dtype=np.float32)])


def find_audio_files(folder, extensions):
    # 評価対象のフォルダ以下から、指定拡張子の音声だけを再帰的に取得する。
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True))
    return sorted(files)


class _SuppressPrint:
    def __enter__(self):
        import sys
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        return self

    def __exit__(self, *args):
        import sys
        sys.stdout.close()
        sys.stdout = self._stdout


def guess_true_label(path, classes):
    # 親フォルダ名が学習済みクラスに一致する場合のみ、評価用の正解ラベルとして採用する。
    parent_folder = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if parent_folder in classes:
        return parent_folder
    return None


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    # 行が正解、列が予測。対角線上の数値が正解数を表す。
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    idx_of = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        cm[idx_of[t], idx_of[p]] += 1

    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.1 * n + 1.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    acc = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0
    ax.set_title(f"Confusion Matrix (accuracy={acc:.3f}, n={cm.sum()})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return cm, acc


def compute_and_print_metrics(y_true, y_pred, classes, save_csv_path):
    # クラス別指標とmacro/micro平均をCSVにも残し、後から比較できるようにする。
    """クラスごとのPrecision/Recall/F1と、マクロ/マイクロ平均を計算・表示・保存する。"""
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )

    rows = []
    print("\n--- クラスごとの指標 ---")
    print(f"{'class':30s} {'precision':>10s} {'recall':>10s} {'f1':>10s} {'support':>8s}")
    for c, p, r, f1, sup in zip(classes, precisions, recalls, f1s, supports):
        print(f"{c:30s} {p:10.3f} {r:10.3f} {f1:10.3f} {sup:8d}")
        rows.append({"class": c, "precision": p, "recall": r, "f1": f1, "support": int(sup)})

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average="macro", zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average="micro", zero_division=0
    )
    print(f"{'macro avg':30s} {macro_p:10.3f} {macro_r:10.3f} {macro_f1:10.3f} {sum(supports):8d}")
    print(f"{'micro avg (=accuracy)':30s} {micro_p:10.3f} {micro_r:10.3f} {micro_f1:10.3f} {sum(supports):8d}")

    rows.append({"class": "macro avg", "precision": macro_p, "recall": macro_r,
                 "f1": macro_f1, "support": int(sum(supports))})
    rows.append({"class": "micro avg", "precision": micro_p, "recall": micro_r,
                 "f1": micro_f1, "support": int(sum(supports))})

    pd.DataFrame(rows).to_csv(save_csv_path, index=False)
    print(f"\n指標を '{save_csv_path}' に保存しました。")

    return macro_f1


def main(args):
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    # 保存済みの重み・クラス順・アーキテクチャ名を読み出す。
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    classes = ckpt["classes"]
    arch = ckpt["arch"]
    sample_rate = ckpt.get("sample_rate", 32000)

    # 保存時と同じPaSST構成を先に生成してから、学習済み重みを復元する。
    with _SuppressPrint():
        model = get_basic_model(mode="logits")
        model.net = get_model_passt(arch=arch, n_classes=len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    audio_files = find_audio_files(args.audio_folder, args.audio_extensions)
    if not audio_files:
        print(f"'{args.audio_folder}' 内に音声ファイルが見つかりませんでした。")
        return

    clip_len = int(sample_rate * args.clip_seconds)
    rows = []
    for path in audio_files:
        try:
            waveform, _ = librosa.core.load(path, sr=sample_rate, mono=True)
        except Exception as e:
            print(f"[スキップ] {path}: 読み込み失敗 ({e})")
            continue

        waveform = pad_or_truncate(waveform.astype(np.float32), clip_len)
        waveform_t = torch.from_numpy(waveform[None, :]).to(device)

        # 評価時は勾配不要なので、推論専用モードで実行する。
        with torch.no_grad(), _SuppressPrint():
            logits = model(waveform_t)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        true_label = guess_true_label(path, classes)

        status = "" if true_label is None else (" [OK]" if true_label == classes[pred_idx] else " [NG]")
        print(f"{path}: predicted={classes[pred_idx]} ({probs[pred_idx]:.3f})"
              + (f", true={true_label}" if true_label else "") + status)

        row = {
            "file": path,
            "true_label": true_label,
            "predicted": classes[pred_idx],
            "confidence": float(probs[pred_idx]),
        }
        for c, p in zip(classes, probs):
            row[f"prob_{c}"] = float(p)
        rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows)
    result_dir = os.path.dirname(args.result_csv)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    df.to_csv(args.result_csv, index=False)
    print(f"\n全結果を '{args.result_csv}' に保存しました。")

    # フォルダ名から正解が判定できる音声だけを評価に含める。
    labeled = df[df["true_label"].notna()]
    if len(labeled) > 0:
        cm_dir = os.path.dirname(args.confusion_matrix_path)
        if cm_dir:
            os.makedirs(cm_dir, exist_ok=True)
        y_true = labeled["true_label"].tolist()
        y_pred = labeled["predicted"].tolist()

        cm, acc = plot_confusion_matrix(y_true, y_pred, classes, args.confusion_matrix_path)
        print(f"正解ラベル付きファイル {len(labeled)} 件で精度を計算しました: accuracy={acc:.3f}")
        print(f"混同行列を '{args.confusion_matrix_path}' に保存しました。")

        compute_and_print_metrics(y_true, y_pred, classes, args.metrics_csv_path)
    else:
        print("正解ラベル(クラス名フォルダ)が見つからなかったため、混同行列・指標は作成しませんでした。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ファインチューニング済みPaSSTでフォルダ内音声を予測')
    parser.add_argument('--checkpoint_path', type=str, default=CHECKPOINT_PATH)
    parser.add_argument('--audio_folder', type=str, default=AUDIO_FOLDER)
    parser.add_argument('--result_csv', type=str, default=RESULT_CSV)
    parser.add_argument('--confusion_matrix_path', type=str, default=CONFUSION_MATRIX_PATH)
    parser.add_argument('--metrics_csv_path', type=str, default=METRICS_CSV_PATH)
    parser.add_argument('--clip_seconds', type=float, default=CLIP_SECONDS)
    parser.add_argument('--audio_extensions', type=tuple, default=AUDIO_EXTENSIONS)
    parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available())

    args = parser.parse_args()
    main(args)
