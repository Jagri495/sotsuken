import os
from moviepy import VideoFileClip

def batch_trim_videos():
    # 1. スクリプトが実行されている現在のフォルダ（ディレクトリ）のパスを取得
    current_dir = r"C:\Users\Owner\Downloads\sotsuken\jikken\Data_10min\mp4files"
    
    # 2. 保存用の出力フォルダを作成（存在しない場合のみ新規作成）
    output_dir = r"C:\Users\Owner\Downloads\sotsuken\jikken\Data_10min\mp4files_cut_start10seconds"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"出力フォルダを作成しました: {output_dir}")

    # 3. 現在のフォルダ内にあるファイルを1つずつチェック
    for filename in os.listdir(current_dir):
        # 拡張子が「.mp4」のファイルのみを対象にする（大文字・小文字を区別しない）
        if filename.lower().endswith(".mp4"):
            
            # 入力動画のフルパスを構築
            input_path = os.path.join(current_dir, filename)
            # 出力動画のフルパスを構築（出力フォルダ内へ保存）
            output_path = os.path.join(output_dir, f"Start10seconds_{filename}")
            
            print(f"--- 処理開始: {filename} ---")
            
            try:
                # 動画ファイルを読み込み
                video = VideoFileClip(input_path)
                
                # 動画の長さを取得（秒単位）
                duration = video.duration
                
                # 10秒に満たない短い動画の場合は、動画の終わりまでを切り出す
                end_time = min(10, duration)
                
                # 最初（0秒）から指定した時間（end_time）までを切り出す
                # ※.subclip(開始秒, 終了秒)
                trimmed_video = video.subclipped(0, end_time)
                
                # 切り出した動画をファイルとして書き出し
                # codec="libx264": 一般的なH.264形式
                # audio_codec="aac": 音声の標準形式
                trimmed_video.write_videofile(
                    output_path, 
                    codec="libx264", 
                    audio_codec="aac"
                )
                
                # メモリ解放のため、動画ファイルをクローズする
                trimmed_video.close()
                video.close()
                
                print(f"成功: {filename} の最初の {end_time} 秒を保存しました。")
                
            except Exception as e:
                # エラーが発生した場合はスキップして次の動画へ
                print(f"エラー（{filename} の処理中に失敗）: {e}")

    print("\n--- すべての処理が完了しました！ ---")

# スクリプトが直接実行された場合のみ、上記の関数を動かす
if __name__ == "__main__":
    batch_trim_videos()