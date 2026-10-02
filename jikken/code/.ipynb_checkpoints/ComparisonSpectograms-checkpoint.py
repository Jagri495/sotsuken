import os
import glob
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# 1. スペクトログラムPNGがまとまっているフォルダのパスを指定
folder_path = r"C:\Users\Owner\Downloads\wavfiles" # ★実際のフォルダ名に変えてください

# 2. フォルダ内のPNGファイル一覧を取得
png_files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
file_names = [os.path.basename(f) for f in png_files]

num_files = len(png_files)
if num_files < 2:
    print("比較するにはフォルダ内に2枚以上のPNGファイルが必要です。")
    exit()

print(f"計 {num_files} 枚の画像を総当たりで比較します...")

# 3. 類似度を格納するための空のマトリクス（表）を作成
similarity_matrix = pd.DataFrame(index=file_names, columns=file_names, dtype=float)

# 4. 総当たりで計算（2重ループ）
for i in range(num_files):
    # 画像1を読み込んでグレースケール化＆リサイズ
    img1 = cv2.imread(png_files[i], cv2.IMREAD_GRAYSCALE)
    img1_resized = cv2.resize(img1, (256, 256))
    
    for j in range(i, num_files):
        if i == j:
            # 同じ画像同士の比較は当然 100% (1.0)
            similarity_matrix.iloc[i, j] = 1.0
            continue
            
        # 画像2を読み込んでグレースケール化＆リサイズ
        img2 = cv2.imread(png_files[j], cv2.IMREAD_GRAYSCALE)
        img2_resized = cv2.resize(img2, (256, 256))
        
        # 類似度（SSIM）を計算
        score, _ = ssim(img1_resized, img2_resized, full=True)
        
        # 表の対角（A vs B と B vs A は同じ）に結果を代入
        similarity_matrix.iloc[i, j] = score
        similarity_matrix.iloc[j, i] = score

# ％表記（0.85 → 85.0）に見やすく変換
similarity_matrix_percent = similarity_matrix * 100

# 5. 結果をExcelで開けるCSVファイルとして保存
output_csv = "spectrogram_similarity_results.csv"
similarity_matrix_percent.to_csv(output_csv)
print(f"\n🎉 計算完了！結果をCSVファイルに保存しました: {output_csv}")

# 6. 【視覚化】ヒートマップ（色付きの表）として画面に表示
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix_percent, annot=True, fmt=".1f", cmap="YlGnBu", vmin=0, vmax=100)
plt.title("Spectrogram Similarity Matrix (%)", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()