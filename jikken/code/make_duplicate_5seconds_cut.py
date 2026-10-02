import os
from pydub import AudioSegment

# 設定
input_root = r"C:\Users\Owner\Downloads\sotsuken\jikken\Data\Distance_Data\PCM-100\180\wavfiles"  # 複数のサブフォルダが入っている親フォルダ
output_root = r"C:\Users\Owner\Downloads\sotsuken\jikken\Data\Distance_Data\PCM-100\180\wavfiles_1seconds_cut"  # 変換後の保存先の親フォルダ
target_length_ms = 1000      # 5秒

if not os.path.exists(output_root):
    os.makedirs(output_root)

# input_root直下の各サブフォルダを順番に処理
for folder_name in os.listdir(input_root):
    input_dir = os.path.join(input_root, folder_name)

    # フォルダでなければスキップ
    if not os.path.isdir(input_dir):
        continue

    # 出力先にも同じ名前のフォルダを作成
    output_dir = os.path.join(output_root, folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"--- フォルダ処理中: {folder_name} ---")

    for filename in os.listdir(input_dir):
        if filename.endswith(".WAV") or filename.endswith(".wav"):
            file_path = os.path.join(input_dir, filename)

            # 音声ファイルの読み込み
            audio = AudioSegment.from_wav(file_path)
            current_length = len(audio)

            # 切り上げ除算で、余りの分も1チャンクとして数える
            total_chunks = -(-current_length // target_length_ms)

            base_name = os.path.splitext(filename)[0]  # 拡張子を除いたファイル名

            for count_cutting in range(total_chunks):
                start = count_cutting * target_length_ms
                end = (count_cutting + 1) * target_length_ms
                cut_5seconds = audio[start:end]

                if end > current_length:
                    # 最後のチャンクが5秒未満なら無音を追加して埋める
                    silence_gap = end - current_length
                    silence = AudioSegment.silent(duration=silence_gap)
                    cut_5seconds = cut_5seconds + silence

                # チャンクごとに別ファイルとして保存
                output_path = os.path.join(
                    output_dir, f"{base_name}_{count_cutting}.wav"
                )
                cut_5seconds.export(output_path, format="wav")

            print(f"変換完了: {folder_name}/{filename}（{total_chunks}個に分割）")

print("すべての処理が終了しました！")
