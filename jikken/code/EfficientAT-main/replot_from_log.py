"""
finetune.py / finetune_passt.py の学習ログ(コンソール出力をコピーしたテキスト)から
学習曲線のグラフを再生成するスクリプト。再学習は不要です。

使い方:
    1. 学習実行時のログ全体(またはエポック行を含む部分)を LOG_FILE のテキストファイルに保存する
       (メモ帳に貼り付けて保存するだけでOK。余計な行が混ざっていても無視されます)
    2. python replot_from_log.py を実行する
"""

import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================ 設定 (ここを変更) ============================
LOG_FILE = "traainin_log.txt"          # ログを保存したテキストファイル
OUTPUT_PLOT_PATH = "traaininng_curves_replotted.png"
# ===========================================================================

# 例: "[Epoch 4/30] train_loss=0.5728 train_acc=0.857 | val_loss=0.5810 val_acc=0.837 val_f1_macro=0.816"
LINE_PATTERN = re.compile(
    r"\[Epoch (\d+)/\d+\]\s*"
    r"train_loss=([\d.]+)\s*train_acc=([\d.]+)\s*\|\s*"
    r"val_loss=([\d.]+)\s*val_acc=([\d.]+)\s*val_f1(?:_macro)?=([\d.]+)"
)


def parse_log(log_text):
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "val_f1_macro": []}
    for match in LINE_PATTERN.finditer(log_text):
        epoch, train_loss, train_acc, val_loss, val_acc, val_f1 = match.groups()
        history["epoch"].append(int(epoch))
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["val_f1_macro"].append(float(val_f1))
    return history


def plot_training_curves(history, save_path):
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

    fig.suptitle("Training Curves (re-plotted from log)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="学習ログから学習曲線グラフを再生成する")
    parser.add_argument("--log_file", type=str, default=LOG_FILE)
    parser.add_argument("--output_plot_path", type=str, default=OUTPUT_PLOT_PATH)
    args = parser.parse_args()

    with open(args.log_file, "r", encoding="utf-8", errors="ignore") as f:
        log_text = f.read()

    history = parse_log(log_text)
    if not history["epoch"]:
        print("エラー: ログの中から 'Epoch ... train_loss=... val_f1...' 形式の行が見つかりませんでした。")
        print("finetune.py / finetune_passt.py 実行時のコンソール出力をそのまま保存したファイルか確認してください。")
        return

    print(f"{len(history['epoch'])} エポック分のログを読み込みました。")
    plot_training_curves(history, args.output_plot_path)
    print(f"グラフを '{args.output_plot_path}' に保存しました。")


if __name__ == "__main__":
    main()
