from transformers import pipeline
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'MS Gothic'
import os

# 1. 音声分類AIモデルを読み込む
print("AIモデルを読み込み中...")
classifier = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")

# 2. wavファイルのパスを指定
audio_file_path = r"C:\Users\Owner\Downloads\sotsuken\jikken\syakapati_me\260531_005.WAV" # ★実際のファイル名に変えてください"

# 3. soundfileを使って安全に音声を読み込む
print("音声を読み込み中...")
audio_data, sampling_rate = sf.read(audio_file_path)

# 4. 音声を解析
print("音声を解析中...")
# soundfileはデフォルトで2次元配列（ステレオ）になることがあるため、左チャンネル[0]または1次元化して渡します
if len(audio_data.shape) > 1:
    audio_data = audio_data[:, 0]

predictions = classifier({"raw": audio_data, "sampling_rate": sampling_rate})

# 5. 結果を表示（確率が高い上位5つ）
print("\n【解析結果】この音である可能性が高いもの：")
labels = []
scores = []
for pred in predictions:
    percent = pred['score'] * 100
    labels.append(pred['label'])
    scores.append(percent)
    print(f"🎵 {pred['label']}: {pred['score'] * 100:.2f}%")
# 6. 正しい手順で上位5つをグラフにする

# ①【まず切り出す】最初に並んでいる「本物のトップ5」を先頭から5つだけ取得する
top_5_labels = labels[:5]
top_5_scores = scores[:5]

# ②【グラフ用に逆順にする】
# barhは下から上に向かって描画されるため、1位を一番上に持ってくるためにここで逆順にします
top_5_labels.reverse()
top_5_scores.reverse()

# 横棒グラフを描画
plt.figure(figsize=(10, 6))
bars = plt.barh(top_5_labels, top_5_scores, color='skyblue')
plt.xlabel('確率 (%)')

file_name = os.path.basename(audio_file_path)
plt.title(f"Audio Classification Results\n({file_name})", fontsize=14, fontweight='bold', pad=15)

# ※ plt.gca().invert_yaxis() は不要になった（混乱の元になる）ので削除しました。

# 棒の横に数値を表示
for bar, score in zip(bars, top_5_scores):
    plt.text(score + 0.2, bar.get_y() + bar.get_height()/2, f'{score:.2f}%', va='center', ha='left')

plt.xlim(0, max(top_5_scores) * 1.15) # 文字が見切れないように横軸に少し余白を作る
plt.tight_layout()
plt.show()