"""
train_distance.py で作成した距離推定モデルを使って、フォルダ内の音声の距離を予測するスクリプト。

使い方:
    python predict_distance.py

【正解ラベルとの比較について】
    AUDIO_FOLDER が学習時と同じ「距離[cm]ごとのフォルダ分け」になっている場合、
    フォルダ名を正解距離とみなして誤差(MAE/RMSE)を自動計算し、
    「予測 vs 正解」の散布図 (SCATTER_PLOT_PATH) を保存します。
    フォルダ分けされていない(距離不明の)音声の場合は、予測結果の保存のみ行われます。
"""

import argparse
import glob
import os

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from fastsde_model.model import SeldNetSubbandFast

# ============================ 設定 (ここを変更) ============================
CHECKPOINT_PATH = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\Fast_SDE\results\pt\Distance_Data_epoch_50_add60,120,180_clip_5.pt"
AUDIO_FOLDER = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\Fast_SDE\results\distance_data_clip_5"
RESULT_CSV = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\Fast_SDE\results\distance_data_clip_5\csv\distance_predictions_clip_5.csv"
SCATTER_PLOT_PATH = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\Fast_SDE\results\distance_data_clip_5\png\distance_scatter_clip_5.png"
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
# ===========================================================================


def pad_or_truncate(waveform, target_len):
    # 学習時と同じ固定長へ揃え、入力形状の不一致を防ぐ。
    if len(waveform) >= target_len:
        return waveform[:target_len]
    return np.concatenate([waveform, np.zeros(target_len - len(waveform), dtype=np.float32)])


def find_audio_files(folder, extensions):
    # 距離フォルダを含む配下全体から、対応する音声形式を再帰的に集める。
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True))
    return sorted(files)


def guess_true_distance(path):
    """親フォルダ名が数値であれば、それを正解距離とみなす。"""
    # 評価用データでは、親フォルダ名を正解距離として読み取る。
    parent_folder = os.path.basename(os.path.dirname(os.path.abspath(path)))
    try:
        return float(parent_folder)
    except ValueError:
        return None


def build_model():
    # train_distance.py と同一のネットワーク構成。ここが違うと重みを読めない。
    return SeldNetSubbandFast(
        features_set="all",
        n_subbands=6,
        c_mid=16,
        n_blocks=2,
        fuse_c=48,
        use_gru=False,
        att_conf="Nothing",
    )


def plot_scatter(y_true, y_pred, save_path):
    # 対角線は「予測値=正解値」を表し、点のずれが距離推定の誤差になる。
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6)

    lims = [min(min(y_true), min(y_pred)) - 5, max(max(y_true), max(y_pred)) + 5]
    ax.plot(lims, lims, 'r--', label='ideal (pred = true)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("True distance [cm]")
    ax.set_ylabel("Predicted distance [cm]")

    mae = float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))
    rmse = float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))
    ax.set_title(f"Distance Prediction (MAE={mae:.2f}cm, RMSE={rmse:.2f}cm)")
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return mae, rmse


def main(args):
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    # 保存済み重みと、学習時の標本化周波数・クリップ長を復元する。
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    sample_rate = ckpt.get("sample_rate", 16000)
    clip_seconds = ckpt.get("clip_seconds", 3.0)

    model = build_model()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    audio_files = find_audio_files(args.audio_folder, args.audio_extensions)
    if not audio_files:
        print(f"'{args.audio_folder}' 内に音声ファイルが見つかりませんでした。")
        return

    clip_len = int(sample_rate * clip_seconds)
    rows = []
    for path in audio_files:
        try:
            waveform, _ = librosa.core.load(path, sr=sample_rate, mono=True)
        except Exception as e:
            print(f"[スキップ] {path}: 読み込み失敗 ({e})")
            continue

        waveform = pad_or_truncate(waveform.astype(np.float32), clip_len)
        waveform_t = torch.from_numpy(waveform[None, :]).to(device)

        # 推論では勾配を計算しないため、メモリ消費と計算を抑える。
        with torch.no_grad():
            pred_distance = model(waveform_t).item()

        true_distance = guess_true_distance(path)
        status = ""
        if true_distance is not None:
            status = f", true={true_distance:.1f}cm, error={abs(pred_distance - true_distance):.1f}cm"
        print(f"{path}: predicted={pred_distance:.1f}cm{status}")

        rows.append({"file": path, "true_distance_cm": true_distance,
                     "predicted_distance_cm": pred_distance})

    if not rows:
        return

    df = pd.DataFrame(rows)
    result_dir = os.path.dirname(args.result_csv)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    df.to_csv(args.result_csv, index=False)
    print(f"\n全結果を '{args.result_csv}' に保存しました。")

    # 距離名フォルダに入った評価音声だけ、MAE/RMSEと散布図の対象にする。
    labeled = df[df["true_distance_cm"].notna()]
    if len(labeled) > 0:
        scatter_dir = os.path.dirname(args.scatter_plot_path)
        if scatter_dir:
            os.makedirs(scatter_dir, exist_ok=True)
        mae, rmse = plot_scatter(labeled["true_distance_cm"].tolist(),
                                  labeled["predicted_distance_cm"].tolist(),
                                  args.scatter_plot_path)
        print(f"正解距離付きファイル {len(labeled)} 件: MAE={mae:.2f}cm, RMSE={rmse:.2f}cm")
        print(f"散布図を '{args.scatter_plot_path}' に保存しました。")
    else:
        print("正解距離(フォルダ名が数値)が見つからなかったため、散布図は作成しませんでした。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FAST-SDEモデルでフォルダ内音声の距離を予測')
    parser.add_argument('--checkpoint_path', type=str, default=CHECKPOINT_PATH)
    parser.add_argument('--audio_folder', type=str, default=AUDIO_FOLDER)
    parser.add_argument('--result_csv', type=str, default=RESULT_CSV)
    parser.add_argument('--scatter_plot_path', type=str, default=SCATTER_PLOT_PATH)
    parser.add_argument('--audio_extensions', type=tuple, default=AUDIO_EXTENSIONS)
    parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available())

    args = parser.parse_args()
    main(args)
