# 必要なライブラリのインストール
import cv2
from skimage.metrics import structural_similarity as ssim

# 1. 比較したい2つのスペクトログラム画像のパスを指定
img_path1 = r"C:\Users\Owner\Downloads\wavfiles\260531_002.png"
img_path2 = r"C:\Users\Owner\Downloads\sotsuken\jikken\syakapati_other\260610_006.png"

# 2. 最初からグレースケール（白黒）として安全に読み込む
img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

# ※万が一ファイルが見つからない場合のチェック
if img1 is None or img2 is None:
    print("❌ エラー: 画像ファイルが見つからないか、読み込めません。パスを確認してください。")
    exit()

# 3. サイズが違うとエラーになるため、同じ大きさに合わせる（256x256）
img1_resized = cv2.resize(img1, (256, 256))
img2_resized = cv2.resize(img2, (256, 256))

# 4. 類似度（SSIM）を計算
similarity, diff = ssim(img1_resized, img2_resized, full=True)

print(f"📊 2つのスペクトログラムの類似度: {similarity * 100:.2f}%")