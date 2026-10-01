import os

import matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from sklearn.metrics import confusion_matrix, classification_report

# ============================ 設定 ============================
# 推論スクリプトが出力したCSVファイルのパス
CSV_PATH = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\EfficientAT-main\results\predictions_EfficientAT_0818_syakapatifrontandback_DataC_epock12.csv"
CLIP_SECONDS = 5
# 混同行列の画像保存先
SAVE_IMAGE_PATH = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\EfficientAT-main\results\confusion_matrix_EfficientAT_syakapatifrontandback_DataC_epock12.png"

# 日本語フォントの候補(上から順に、環境にインストールされている最初のものを使う)
JP_FONT_CANDIDATES = [
    "Yu Gothic", "Meiryo", "MS Gothic",          # Windows標準
    "Hiragino Sans", "Hiragino Kaku Gothic Pro",  # macOS標準
    "Noto Sans CJK JP", "IPAexGothic", "TakaoGothic",  # Linux向け
]
# ==============================================================


def setup_japanese_font():
    """環境にインストールされている日本語フォントを自動検出して設定する。
    見つからない場合は警告を出し、デフォルトフォントのまま続行する(文字化けする可能性あり)。"""
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in JP_FONT_CANDIDATES:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            print(f"日本語フォント '{font_name}' を使用します。")
            break
    else:
        print("警告: 日本語フォントが見つかりませんでした。日本語ラベルが文字化けする可能性があります。")
        print(f"       (候補: {', '.join(JP_FONT_CANDIDATES)})")

    plt.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け対策


def get_true_label(file_path):
    """
    ファイルパスから正解クラス(親フォルダ名)を取得する関数
    例: ".../Data_D/wavfile_5seconds_cut/dog/sample01.wav" -> "dog"
    """
    return os.path.basename(os.path.dirname(file_path))


def main():
    setup_japanese_font()

    if not os.path.exists(CSV_PATH):
        print(f"エラー: 指定されたCSVファイルが見つかりません ({CSV_PATH})")
        return

    df = pd.read_csv(CSV_PATH)

    # 1. ファイルパスから正解ラベル(True Label)を抽出
    df['true_label'] = df['file'].apply(get_true_label)

    # 2. クラス一覧の取得(正解ラベルと予測ラベルの重複のない和集合)
    labels = sorted(list(set(df['true_label'].unique()) | set(df['predicted'].unique())))

    # 3. 混同行列の計算
    cm = confusion_matrix(df['true_label'], df['predicted'], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # コンソールにテキスト形式で混同行列を表示
    print("=== 混同行列 (Confusion Matrix) ===")
    print(cm_df)
    print("\n" + "="*40 + "\n")

    # 分類精度(Accuracy / Precision / Recall / F1-score)を表示
    print("=== 分類レポート ===")
    print(classification_report(df['true_label'], df['predicted'], target_names=labels, zero_division=0))

    # 4. ヒートマップの描画と保存
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Predicted Label (予測クラス)', fontsize=12)
    plt.ylabel('True Label (正解クラス)', fontsize=12)

    # タイトルはpngファイル名(拡張子なし)と同じにする
    title = os.path.splitext(os.path.basename(SAVE_IMAGE_PATH))[0]
    plt.title(title, fontsize=14)
    plt.tight_layout()

    os.makedirs(os.path.dirname(SAVE_IMAGE_PATH), exist_ok=True)
    plt.savefig(SAVE_IMAGE_PATH, dpi=300)
    print(f"混同行列の画像を保存しました: {SAVE_IMAGE_PATH}")
    # plt.show()  # テスト実行のためコメントアウト


if __name__ == '__main__':
    main()
