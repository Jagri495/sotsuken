import os
from pydub import AudioSegment

# 設定
input_dir = r"C:\Users\Owner\Downloads\sotsuken\jikken\syakapati_other\wav"   # 元のWAVファイルが入っているフォルダ
output_dir = r"C:\Users\Owner\Downloads\sotsuken\jikken\syakapati_other\png" # 変換後の保存先フォルダ
target_length_ms = 5000      # 揃えたい時間（ミリ秒単位。5秒 = 5000ms）

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    if filename.endswith(".WAV") or filename.endswith(".wav"):
        file_path = os.path.join(input_dir, filename)
        
        # 音声ファイルの読み込み
        audio = AudioSegment.from_wav(file_path)
        current_length = len(audio)
        
        if current_length > target_length_ms:
            # ① 指定より長い場合はトリミング（後ろをカット）
            trimmed_audio = audio[:target_length_ms]
        else:
            # ② 指定より短い場合は無音（サイレンス）を追加
            silence_gap = target_length_ms - current_length
            silence = AudioSegment.silent(duration=silence_gap)
            trimmed_audio = audio + silence
            
        # 保存
        output_path = os.path.join(output_dir, filename+'_start_5seconds.wav')
        trimmed_audio.export(output_path, format="wav")
        print(f"変換完了: {filename}")

print("すべての処理が終了しました！")