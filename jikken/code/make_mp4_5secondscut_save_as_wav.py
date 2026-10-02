import os
import static_ffmpeg
from pydub import AudioSegment

# ffmpeg と ffprobe のパスを自動セットアップ
static_ffmpeg.add_paths()

# 設定
input_root = r"C:\Users\Owner\Downloads\sotsuken\jikken\Data\Distance_Data\Zoom\180\wavfiles"  # 複数のサブフォルダ（Hindu_shuffleなど）が入っている親フォルダ
output_root = r"C:\Users\Owner\Downloads\sotsuken\jikken\Data\Distance_Data\Zoom\180\wavfiles_1seconds_cut"  # 変換後の保存先の親フォルダ
target_length_ms = 1000  # 5秒 (ミリ秒)

if not os.path.exists(input_root):
    print(
        f"【エラー】入力フォルダが存在しません: {input_root}\nパスを確認してください。"
    )
else:
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    total_processed_count = 0

    # input_root直下の各サブフォルダ（Hindu_shuffleなど）を順番に処理
    for folder_name in os.listdir(input_root):
        input_dir = os.path.join(input_root, folder_name)

        # フォルダでなければスキップ
        if not os.path.isdir(input_dir):
            continue

        # 出力先にも同じ名前のフォルダを作成
        output_folder_root = os.path.join(output_root, folder_name)
        os.makedirs(output_folder_root, exist_ok=True)

        print(f"=== フォルダ処理中: {folder_name} ===")

        processed_count = 0

        # サブフォルダ内および直下のMP4/M4Aを再帰的に探索
        for root, dirs, files in os.walk(input_dir):
            mp4_files = [
                f for f in files
                if f.lower().endswith(".m4a") or f.lower().endswith(".mp4")
            ]
            if not mp4_files:
                continue

            # 保存先フォルダ構造を再現
            rel_path = os.path.relpath(root, input_dir)
            output_dir = (
                output_folder_root
                if rel_path == "."
                else os.path.join(output_folder_root, rel_path)
            )
            os.makedirs(output_dir, exist_ok=True)

            print(
                f"--- 処理中サブフォルダ: {rel_path if rel_path != '.' else 'root'} ---"
            )

            for filename in mp4_files:
                file_path = os.path.join(root, filename)

                # MP4/M4Aから音声を読み込み
                ext = os.path.splitext(filename)[1].lower()
                audio_format = "m4a" if ext == ".m4a" else "mp4"
                audio = AudioSegment.from_file(file_path, format=audio_format)
                current_length = len(audio)

                total_chunks = -(-current_length // target_length_ms)
                base_name = os.path.splitext(filename)[0]

                for count_cutting in range(total_chunks):
                    start = count_cutting * target_length_ms
                    end = (count_cutting + 1) * target_length_ms
                    cut_5seconds = audio[start:end]

                    if end > current_length:
                        silence_gap = end - current_length
                        silence = AudioSegment.silent(duration=silence_gap)
                        cut_5seconds = cut_5seconds + silence

                    output_path = os.path.join(
                        output_dir, f"{base_name}_{count_cutting}.wav"
                    )
                    cut_5seconds.export(output_path, format="wav")

                print(
                    f"変換完了: {folder_name}/{filename}（{total_chunks}個のWAVに分割）"
                )
                processed_count += 1

        if processed_count == 0:
            print(
                f"【注意】 {input_dir} 内に .mp4 または .m4a ファイルが見つかりませんでした。"
            )
        else:
            print(f"--- {folder_name} 完了（{processed_count}ファイル処理） ---\n")

        total_processed_count += processed_count

    if total_processed_count == 0:
        print(
            f"【注意】 {input_root} 内のどのサブフォルダにも .mp4 または .m4a ファイルが見つかりませんでした。"
        )

    print("すべての処理が終了しました！")
