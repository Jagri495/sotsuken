import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def generate_spectrograms(directory_path,output_dir):
    """
    指定したディレクトリ内の全wavファイルからスペクトログラム画像を生成し、
    同じディレクトリに保存する関数。
    """
    # ディレクトリの存在確認
    if not os.path.isdir(directory_path):
        print(f"エラー: 指定されたディレクトリが存在しません -> {directory_path}")
        return

    # 検索パターンの作成 (ディレクトリ内の.wavファイルをすべて取得)
    search_pattern = os.path.join(directory_path, '*.wav')
    wav_files = glob.glob(search_pattern)

    if not wav_files:
        print(f"指定されたディレクトリに .wav ファイルが見つかりません: {directory_path}")
        return

    print(f"合計 {len(wav_files)} 件の wav ファイルを処理します...\n")

    for wav_file in wav_files:
        try:
            print(f"処理中: {os.path.basename(wav_file)}")
            
            # 音声ファイルを読み込む (sr=None で元のサンプリングレートを維持)
            y, sr = librosa.load(wav_file, sr=None)
            
            # メルスペクトログラムの計算
            # 音声信号を周波数成分に分解し、人間の聴覚特性に合わせたスケール(メル尺度)に変換
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            
            # 振幅をデシベル(dB)単位に変換（視覚的に分かりやすくするため）
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            # プロットの作成
            plt.rcParams['font.family'] = 'MS Gothic'  # 日本語フォントの設定（必要に応じて変更）
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Spectrogram: {os.path.basename(wav_file)}')
            plt.tight_layout()
            
            # 保存先のパスを作成 (元のファイル名の拡張子を .png に変更)
            file_name = os.path.basename(wav_file)
            base_name = os.path.splitext(file_name)[0]
            output_path = os.path.join(output_dir, f"{base_name}.png")#保存先に保存
            
            # 画像として保存
            plt.savefig(output_path)
            
            # メモリ解放（ループ処理でメモリリークを防ぐために重要）
            plt.close()
            
            print(f" -> 保存完了: {os.path.basename(output_path)}")
            
        except Exception as e:
            print(f"エラー発生 ({os.path.basename(wav_file)}): {e}")

# ==========================================
# 実行部分
# ==========================================
if __name__ == "__main__":
    # ここにwavファイルが保存されているディレクトリのパスを指定してください
    # 例: Windowsの場合は r"C:\Users\Name\Music\WavFiles" のように r をつけるとパス指定が簡単です
    target_directory = r"C:\Users\Owner\Downloads\sotsuken\jikken\Test\wavfiles\shakapati_back"
    output_directory = r"C:\Users\Owner\Downloads\sotsuken\jikken\Test\wavfiles_5seconds_cut\shakapati_back"
    generate_spectrograms(target_directory, output_directory)
    print("\nすべての処理が完了しました。")