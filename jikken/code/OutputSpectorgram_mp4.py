import os
import glob
import subprocess
import sys

def find_ffmpeg():
    """PC内からffmpeg.exeの標準的なパスを自動で探す関数"""
    # 1. 通常の環境変数から探す
    if subprocess.run(["where", "ffmpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        return "ffmpeg"
        
    # 2. Pythonのパッケージ（ffmpeg_downloader）の場所を推測して探す
    executable_dir = os.path.dirname(sys.executable)
    possible_paths = [
        os.path.join(executable_dir, "Lib", "site-packages", "ffmpeg_downloader", "bin", "ffmpeg.exe"),
        os.path.join(executable_dir, "Scripts", "ffmpeg.exe"),
    ]
    
    # 3. ユーザーフォルダ以下の可能性のある場所を一通り探す
    user_profile = os.environ.get("USERPROFILE", "C:\\Users\\Owner")
    # AppData内の各Pythonバージョンを自動スキャン
    local_programs = os.path.join(user_profile, "AppData", "Local", "Programs", "Python")
    if os.path.isdir(local_programs):
        for py_dir in os.listdir(local_programs):
            path = os.path.join(local_programs, py_dir, "Lib", "site-packages", "ffmpeg_downloader", "bin", "ffmpeg.exe")
            possible_paths.append(path)

    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    return None

def convert_mp4_to_wav(directory_path):
    if not os.path.isdir(directory_path):
        print(f"エラー: 指定されたディレクトリが存在しません -> {directory_path}")
        return

    search_pattern = os.path.join(directory_path, '*.mp4')
    mp4_files = glob.glob(search_pattern)

    if not mp4_files:
        print(f"指定されたディレクトリに .mp4 ファイルが見つかりません: {directory_path}")
        return

    # FFmpegの場所を自動探索
    ffmpeg_cmd = find_ffmpeg()
    if not ffmpeg_cmd:
        print("\n【致命的なエラー】: FFmpeg（本体）がPC内に見つかりません。")
        print("PowerShellで一度以下のコマンドを実行してください：")
        print("pip install ffmpeg-downloader")
        print("ffdl install")
        return

    print(f"FFmpegを使用します: {ffmpeg_cmd}")
    print(f"合計 {len(mp4_files)} 件の mp4 ファイルを wav に変換します...\n")

    for mp4_file in mp4_files:
        try:
            print(f"変換中: {os.path.basename(mp4_file)}")
            
            base_name = os.path.splitext(mp4_file)[0]
            output_path = f"{base_name}.wav"
            
            # 外部プロセスとして FFmpeg を実行
            cmd = [ffmpeg_cmd, '-y', '-i', mp4_file, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', output_path]
            
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            print(f" -> 変換成功: {os.path.basename(output_path)}")
            
        except subprocess.CalledProcessError as e:
            print(f"FFmpegエラー発生 ({os.path.basename(mp4_file)}): {e.stderr.decode('utf-8', errors='ignore')}")
        except Exception as e:
            print(f"エラー発生 ({os.path.basename(mp4_file)}): {e}")

if __name__ == "__main__":
    target_directory = r"C:\Users\Owner\Downloads\sotsuken\jikken\Data_10min\shakapati_front"
    convert_mp4_to_wav(target_directory)
    print("\nすべての変換処理が完了しました。")