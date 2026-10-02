import os
import glob
import matplotlib.pyplot as plt
from PIL import Image

# 1. あなたのスペクトログラムPNGがまとまっているフォルダのパスを指定
folder_path = r"C:\Users\Owner\Downloads\wavfiles" # 実際のフォルダ名に変えてください

# 2. フォルダ内のPNGファイルの一覧を取得
png_files = glob.glob(os.path.join(folder_path, "*.png"))

if not png_files:
    print("PNGファイルが見つかりませんでした。パスを確認してください。")
else:
    # 3. 画像を並べて表示する設定（例：横に4個ずつ並べる）
    num_files = min(len(png_files), 12)  # 最大12枚まで表示
    cols = 4
    rows = (num_files + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()
    
    for i, file_path in enumerate(png_files[:num_files]):
        img = Image.open(file_path)
        axes[i].imshow(img)
        axes[i].set_title(os.path.basename(file_path), fontsize=8)
        axes[i].axis('off') # 枠線を消す
        
    # 余ったプロット枠を消す
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()