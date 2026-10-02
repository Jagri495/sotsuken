import os
# MoviePy v2.x 以降のインポート方法
from moviepy import VideoFileClip
# リサイズ処理を安全に行うためのエフェクト関数をインポート
from moviepy.video.fx import Resize

def batch_convert_mp4_to_gif():
    # ----------------------------------------------------
    # 【設定】ここで切り出したい「秒数」を設定できます！
    # ----------------------------------------------------
    START_SEC = 20   # 切り出し開始の秒数 (例: 2)
    END_SEC = 30     # 切り出し終了の秒数 (例: 5)
    
    # GIFの画質・ファイルサイズ調整用
    GIF_FPS = 10     # 1秒あたりのフレーム数
    GIF_WIDTH = 480  # 横幅のピクセル数

    # 1. スクリプトが実行されている現在のフォルダを取得
    current_dir = r"C:\Users\Owner\Downloads\sotsuken\jikken\Data_10min\mp4files"
    
    # 2. 保存用の「gif_output」フォルダを作成（存在しない場合のみ）
    output_dir = r"C:\Users\Owner\Downloads\sotsuken\jikken\Data_10min\GIFfiles\cut_start20to30seconds"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"GIF出力フォルダを作成しました: {output_dir}")

    # 3. 現在のフォルダ内にあるファイルを走査
    for filename in os.listdir(current_dir):
        # MP4ファイルのみを対象にする
        if filename.lower().endswith(".mp4"):
            
            input_path = os.path.join(current_dir, filename)
            
            # 出力ファイル名を決定 (例: Deal_shuffle_processed.gif)
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_processed.gif"
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"\n--- GIF変換開始: {filename} ---")
            
            try:
                # 動画ファイルを読み込み
                video = VideoFileClip(input_path)
                
                # 動画の総長さを取得
                duration = video.duration
                
                # 指定された終了秒が動画の長さを超えている場合は、動画の末尾に合わせる
                actual_end = min(END_SEC, duration)
                
                # 【修正点】バージョンに依存しない標準のスライス構文 [開始:終了] を使用
                # これによりメソッド名のエラー（AttributeError）を完全に回避します
                clipped_video = video[START_SEC:actual_end]
                
                # 【修正点】最新版MoviePyで推奨されている安全なリサイズ方法に変更
                # 横幅(width)を指定し、縦横比を維持したまま縮小します
                resized_video = clipped_video.with_effects([Resize(width=GIF_WIDTH)])
                
                print(f"カット範囲: {START_SEC}秒 ～ {actual_end}秒 (総長: {duration:.1f}秒)")
                print(f"解像度横幅: {GIF_WIDTH}px / フレームレート: {GIF_FPS}fps で書き出し中...")
                
                # GIFファイルとして書き出し
                resized_video.write_gif(
                    output_path, 
                    fps=GIF_FPS, 
                )
                
                # メモリ解放のためクローズ
                resized_video.close()
                clipped_video.close()
                video.close()
                
                print(f"成功: {output_filename} を保存しました。")
                
            except Exception as e:
                print(f"エラー（{filename} の変換中に失敗）: {e}")

    print("\n--- すべてのGIF変換処理が完了しました！ ---")

if __name__ == "__main__":
    batch_convert_mp4_to_gif()