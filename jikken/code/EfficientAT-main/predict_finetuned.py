"""
finetune.py で作成した自前モデル(チェックポイント)を使って、
フォルダ内の音声ファイルを一括で予測するスクリプト。

使い方:
    1. 下の「設定」セクションを変更する。
       - CHECKPOINT_PATH: finetune.py が保存したモデルファイル(例: finetuned_model.pt)
       - AUDIO_FOLDER: 予測したい音声が入っているフォルダ
    2. `python predict_finetuned.py` を実行する。
    3. 各ファイルの予測クラスと確信度がコンソールに表示され、
       全結果が RESULT_CSV にまとめて保存される。

【混同行列(confusion matrix)について】
    AUDIO_FOLDER が学習時と同じように「クラス名ごとのフォルダ分け」になっている場合
    (例: AUDIO_FOLDER/dog/*.wav, AUDIO_FOLDER/cat/*.wav)、
    フォルダ名を正解ラベルとみなして自動的に精度を計算し、
    混同行列の画像 (CONFUSION_MATRIX_PATH) を保存します。
    正解ラベルが分からない(フォルダ分けされていない)音声だけの場合は、
    予測結果の保存のみ行われます(混同行列は作成されません)。
"""

import argparse
import glob
import os

import librosa
import matplotlib
matplotlib.use("Agg")  # 画面がない環境でも画像保存できるように
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

from helpers.utils import NAME_TO_WIDTH
from models.dymn.model import get_model as get_dymn
from models.mn.model import get_model as get_mobilenet
from models.preprocess import AugmentMelSTFT

# ============================ 設定 (ここを変更) ============================
CHECKPOINT_PATH = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\EfficientAT-main\results\pt\Zoom_test_wavfiles_5seconds_cut_finetuned_model_epoch30.pt"
AUDIO_FOLDER = r"C:\Users\Owner\Downloads\sotsuken\jikken\Data\Data_other\Data_D\wavfile_5seconds_cut"
RESULT_CSV = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\EfficientAT-main\results\Zoom_test_wavfiles_5seconds_cut_prediction_results_epoch30_predict_Data_D.csv"
CONFUSION_MATRIX_PATH = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\EfficientAT-main\results\Zoom_test_wavfiles_5seconds_cut_confusion_matrix_epoch30_predict_Data_D.png"   # 正解ラベルが分かる場合のみ作成される
METRICS_CSV_PATH = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\EfficientAT-main\results\Zoom_test_wavfiles_5seconds_cut_metrics_epoch30_predict_Data_D.csv"                 # クラスごとのPrecision/Recall/F1
CLIP_SECONDS = 5
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
# ===========================================================================


def pad_or_truncate(waveform, target_len):
    # 学習時と同じクリップ長へそろえる。長い音声は先頭を使い、短い音声は無音で補う。
    if len(waveform) >= target_len:
        return waveform[:target_len]
    return np.concatenate([waveform, np.zeros(target_len - len(waveform), dtype=np.float32)])


def find_audio_files(folder):
    # クラス別サブフォルダを含め、対応形式の音声をすべて集める。
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True))
    return sorted(files)


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


def guess_true_label(path, audio_folder, classes):
    """
    音声ファイルの直上フォルダ名が既知のクラス名と一致する場合、それを正解ラベルとみなす。
    一致しない場合(=正解ラベルが不明)は None を返す。
    """
    # 評価用にクラス名フォルダで整理された場合のみ、親フォルダを正解として使う。
    parent_folder = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if parent_folder in classes:
        return parent_folder
    return None


def compute_and_print_metrics(y_true, y_pred, classes, save_csv_path):
    """クラスごとのPrecision/Recall/F1と、マクロ/マイクロ平均を計算・表示・保存する。"""
    # labels を明示し、予測が1件もないクラスも表から消えないようにする。
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )

    rows = []
    print("\n--- クラスごとの指標 ---")
    print(f"{'class':30s} {'precision':>10s} {'recall':>10s} {'f1':>10s} {'support':>8s}")
    for c, p, r, f1, sup in zip(classes, precisions, recalls, f1s, supports):
        print(f"{c:30s} {p:10.3f} {r:10.3f} {f1:10.3f} {sup:8d}")
        rows.append({"class": c, "precision": p, "recall": r, "f1": f1, "support": int(sup)})

    # macro平均は各クラスを同じ重みで、micro平均は全サンプルをまとめて評価する。
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


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    # 行=正解クラス、列=予測クラス。対角成分が正しく分類できた件数を表す。
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    idx_of = {c: i for i, c in enumerate(classes)}
    # 文字列ラベルを行列の添字へ変換して件数を加算する。
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


def main(args):
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    # チェックポイントからクラス順・モデル構成・標本化周波数も一緒に復元する。
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    classes = ckpt["classes"]
    model_name = ckpt["model_name"]
    head_type = ckpt.get("head_type", "mlp")
    sample_rate = ckpt.get("sample_rate", 32000)

    # 学習済み重みを読み込む前に、保存時と同じネットワーク構成を生成する。
    with _SuppressPrint():
        if model_name.startswith("dymn"):
            model = get_dymn(width_mult=NAME_TO_WIDTH(model_name), num_classes=len(classes))
        else:
            model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), head_type=head_type,
                                   num_classes=len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    mel = AugmentMelSTFT(n_mels=args.n_mels, sr=sample_rate,
                          win_length=args.window_size, hopsize=args.hop_size)
    mel.to(device)
    mel.eval()

    audio_files = find_audio_files(args.audio_folder)
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

        # 推論だけなので勾配を作らず、計算量とGPU/CPUメモリを抑える。
        with torch.no_grad():
            spec = mel(waveform_t).unsqueeze(1)
            logits, _ = model(spec)
        # 出力スコアを確率に変換し、最大値のクラスをファイルの予測結果とする。
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        true_label = guess_true_label(path, args.audio_folder, classes)

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
    df.to_csv(args.result_csv, index=False)
    print(f"\n全結果を '{args.result_csv}' に保存しました。")

    # 正解ラベルが判明するファイルだけを、混同行列と評価指標の対象にする。
    labeled = df[df["true_label"].notna()]
    if len(labeled) > 0:
        y_true = labeled["true_label"].tolist()
        y_pred = labeled["predicted"].tolist()

        cm, acc = plot_confusion_matrix(y_true, y_pred, classes, args.confusion_matrix_path)
        print(f"正解ラベル付きファイル {len(labeled)} 件で精度を計算しました: accuracy={acc:.3f}")
        print(f"混同行列を '{args.confusion_matrix_path}' に保存しました。")

        compute_and_print_metrics(y_true, y_pred, classes, args.metrics_csv_path)
    else:
        print("正解ラベル(クラス名フォルダ)が見つからなかったため、混同行列・指標は作成しませんでした。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ファインチューニング済みモデルでフォルダ内音声を予測')
    parser.add_argument('--checkpoint_path', type=str, default=CHECKPOINT_PATH)
    parser.add_argument('--audio_folder', type=str, default=AUDIO_FOLDER)
    parser.add_argument('--result_csv', type=str, default=RESULT_CSV)
    parser.add_argument('--confusion_matrix_path', type=str, default=CONFUSION_MATRIX_PATH)
    parser.add_argument('--metrics_csv_path', type=str, default=METRICS_CSV_PATH)
    parser.add_argument('--clip_seconds', type=float, default=CLIP_SECONDS)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_mels', type=int, default=128)

    args = parser.parse_args()
    main(args)
