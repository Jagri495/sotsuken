"""
マイク機種(例: Zoom H6 vs Sony PCM-D100)が変わっても、距離推定モデルの精度が
保たれるかを評価するスクリプト。

train_distance.py / predict_distance.py と同じフォルダに置いて実行してください
(このスクリプトはその2つのファイルから関数を再利用します)。

【評価の考え方】
    マイクA(例: Zoom)のデータで学習したモデルを、
      1. マイクA自身の検証データ(同マイク)
      2. マイクB(例: PCM-D100)の全データ(異なるマイク)
    の両方で評価し、精度差を見ます。マイクBを学習に使った場合も同様に行い、
    計4パターン(同マイク2件+異なるマイク2件)の結果を一覧にします。
    「同マイク」と「異なるマイク」の精度差が小さいほど、機種への汎化性能が高いことになります。

    あわせて、2マイク間の音量比(ILD相当の指標)も参考情報として出力します。
    録音開始タイミングが同期していない今回の収録方法では、方向推定にそのまま使うことは
    できませんが、「2マイクの音量差にどの程度の傾向があるか」を把握する材料にはなります。

使い方:
    1. 下の設定セクションで、各マイクのデータフォルダのパスを指定する
       (finetune.pyのデータと同じく、DATA_DIR/距離[cm]/... という並び。
        フォルダ構成が多少ネストしていても train_distance.py 側のロジックで対応済み)
    2. python evaluate_mic_generalization.py を実行する
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset

from train_distance import (
    DistanceFolderDataset, build_model, run_epoch, EXCLUDE_DIR_NAMES,
)
from predict_distance import plot_scatter


class _SuppressPrint:
    """run_epoch()が出すバッチごとの逐次ログ(このラッパーでは冗長)を抑制する。"""
    def __enter__(self):
        import sys
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        return self

    def __exit__(self, *args):
        import sys
        sys.stdout.close()
        sys.stdout = self._stdout

# ============================ 設定 (ここを変更) ============================
MIC_A_NAME = "Zoom"
MIC_A_DATA_DIR = "distance_data_zoom"          # ← Zoomのデータフォルダのパス

MIC_B_NAME = "PCM-D100"
MIC_B_DATA_DIR = "distance_data_pcmd100"       # ← PCM-D100のデータフォルダのパス

SAMPLE_RATE = 16000
CLIP_SECONDS = 3.0
VAL_RATIO = 0.2       # 各マイク内での学習/検証分割の割合(「同マイク」評価用)
EPOCHS = 3
BATCH_SIZE = 16
LR = 1e-3
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

OUTPUT_DIR = "mic_generalization_results"
# ===========================================================================


def train_on_mic(data_dir, mic_name, device):
    """1つのマイクのデータで学習し、そのマイク自身の検証データでの結果も返す。"""
    print(f"\n{'='*60}\n[{mic_name}] 学習データ読み込み: {data_dir}\n{'='*60}")
    full_dataset = DistanceFolderDataset(data_dir, SAMPLE_RATE, CLIP_SECONDS,
                                          AUDIO_EXTENSIONS, exclude_dir_names=EXCLUDE_DIR_NAMES)

    val_size = max(1, int(len(full_dataset) * VAL_RATIO))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size],
                                       generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_mae = float("inf")
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        with _SuppressPrint():
            train_loss, train_mae = run_epoch(model, train_loader, device, criterion, optimizer,
                                               epoch_label=f"{mic_name} Epoch {epoch}/{EPOCHS} train")
            val_loss, val_mae = run_epoch(model, val_loader, device, criterion, optimizer=None,
                                           epoch_label=f"{mic_name} Epoch {epoch}/{EPOCHS} val")
        if epoch % 10 == 0 or epoch == EPOCHS:
            print(f"  [{mic_name}] Epoch {epoch}/{EPOCHS} train_mae={train_mae:.2f}cm val_mae={val_mae:.2f}cm")
        if val_mae <= best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_val_mae


def evaluate_on_full_dataset(model, data_dir, device):
    """指定フォルダの全データに対してMAE/RMSEを計算する(既に学習で見たデータかどうかは問わない)。"""
    dataset = DistanceFolderDataset(data_dir, SAMPLE_RATE, CLIP_SECONDS,
                                     AUDIO_EXTENSIONS, exclude_dir_names=EXCLUDE_DIR_NAMES)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for waveform, distance in loader:
            waveform = waveform.to(device)
            pred = model(waveform).squeeze(-1).cpu().numpy()
            preds.extend(pred.tolist())
            trues.extend(distance.numpy().tolist())

    preds, trues = np.array(preds), np.array(trues)
    mae = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    return mae, rmse, trues, preds


def train_on_mixed(data_dir_a, data_dir_b, device):
    """2マイク分のデータを混ぜて学習する(パターンC)。両マイクを区別せず1つのデータセットとして扱う。"""
    print(f"\n{'='*60}\n[混合学習] {MIC_A_NAME} + {MIC_B_NAME}\n{'='*60}")
    dataset_a = DistanceFolderDataset(data_dir_a, SAMPLE_RATE, CLIP_SECONDS,
                                       AUDIO_EXTENSIONS, exclude_dir_names=EXCLUDE_DIR_NAMES)
    dataset_b = DistanceFolderDataset(data_dir_b, SAMPLE_RATE, CLIP_SECONDS,
                                       AUDIO_EXTENSIONS, exclude_dir_names=EXCLUDE_DIR_NAMES)
    mixed_dataset = ConcatDataset([dataset_a, dataset_b])
    print(f"混合データ件数: {len(mixed_dataset)} 件 ({MIC_A_NAME}: {len(dataset_a)}件, {MIC_B_NAME}: {len(dataset_b)}件)")

    val_size = max(1, int(len(mixed_dataset) * VAL_RATIO))
    train_size = len(mixed_dataset) - val_size
    train_set, val_set = random_split(mixed_dataset, [train_size, val_size],
                                       generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_mae = float("inf")
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        with _SuppressPrint():
            run_epoch(model, train_loader, device, criterion, optimizer,
                      epoch_label=f"Mixed Epoch {epoch}/{EPOCHS} train")
            _, val_mae = run_epoch(model, val_loader, device, criterion, optimizer=None,
                                    epoch_label=f"Mixed Epoch {epoch}/{EPOCHS} val")
        if epoch % 10 == 0 or epoch == EPOCHS:
            print(f"  [混合] Epoch {epoch}/{EPOCHS} val_mae={val_mae:.2f}cm")
        if val_mae <= best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_val_mae


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"使用デバイス: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- マイクAで学習 ---
    model_a, val_mae_a = train_on_mic(MIC_A_DATA_DIR, MIC_A_NAME, device)
    torch.save({"model_state_dict": model_a.state_dict(), "sample_rate": SAMPLE_RATE,
                "clip_seconds": CLIP_SECONDS}, os.path.join(OUTPUT_DIR, f"model_trained_on_{MIC_A_NAME}.pt"))

    # --- マイクBで学習 ---
    model_b, val_mae_b = train_on_mic(MIC_B_DATA_DIR, MIC_B_NAME, device)
    torch.save({"model_state_dict": model_b.state_dict(), "sample_rate": SAMPLE_RATE,
                "clip_seconds": CLIP_SECONDS}, os.path.join(OUTPUT_DIR, f"model_trained_on_{MIC_B_NAME}.pt"))

    # --- クロス評価: Aで学習→Bで評価、Bで学習→Aで評価 ---
    print(f"\n{'='*60}\nクロス評価\n{'='*60}")
    mae_a_on_b, rmse_a_on_b, trues_ab, preds_ab = evaluate_on_full_dataset(model_a, MIC_B_DATA_DIR, device)
    mae_b_on_a, rmse_b_on_a, trues_ba, preds_ba = evaluate_on_full_dataset(model_b, MIC_A_DATA_DIR, device)

    plot_scatter(trues_ab.tolist(), preds_ab.tolist(),
                 os.path.join(OUTPUT_DIR, f"scatter_train_{MIC_A_NAME}_test_{MIC_B_NAME}.png"))
    plot_scatter(trues_ba.tolist(), preds_ba.tolist(),
                 os.path.join(OUTPUT_DIR, f"scatter_train_{MIC_B_NAME}_test_{MIC_A_NAME}.png"))

    # --- パターンC: 混合学習 → 各マイク単体でテスト ---
    model_mixed, val_mae_mixed = train_on_mixed(MIC_A_DATA_DIR, MIC_B_DATA_DIR, device)
    torch.save({"model_state_dict": model_mixed.state_dict(), "sample_rate": SAMPLE_RATE,
                "clip_seconds": CLIP_SECONDS}, os.path.join(OUTPUT_DIR, "model_trained_on_mixed.pt"))

    mae_mixed_on_a, rmse_mixed_on_a, trues_ma, preds_ma = evaluate_on_full_dataset(model_mixed, MIC_A_DATA_DIR, device)
    mae_mixed_on_b, rmse_mixed_on_b, trues_mb, preds_mb = evaluate_on_full_dataset(model_mixed, MIC_B_DATA_DIR, device)
    plot_scatter(trues_ma.tolist(), preds_ma.tolist(),
                 os.path.join(OUTPUT_DIR, f"scatter_train_Mixed_test_{MIC_A_NAME}.png"))
    plot_scatter(trues_mb.tolist(), preds_mb.tolist(),
                 os.path.join(OUTPUT_DIR, f"scatter_train_Mixed_test_{MIC_B_NAME}.png"))

    # --- 結果まとめ ---
    summary = pd.DataFrame([
        {"学習マイク": MIC_A_NAME, "評価データ": f"{MIC_A_NAME}(同マイク・検証データ)", "MAE_cm": round(val_mae_a, 2), "種別": "同マイク"},
        {"学習マイク": MIC_A_NAME, "評価データ": f"{MIC_B_NAME}(異なるマイク・全データ)", "MAE_cm": round(mae_a_on_b, 2), "RMSE_cm": round(rmse_a_on_b, 2), "種別": "異なるマイク"},
        {"学習マイク": MIC_B_NAME, "評価データ": f"{MIC_B_NAME}(同マイク・検証データ)", "MAE_cm": round(val_mae_b, 2), "種別": "同マイク"},
        {"学習マイク": MIC_B_NAME, "評価データ": f"{MIC_A_NAME}(異なるマイク・全データ)", "MAE_cm": round(mae_b_on_a, 2), "RMSE_cm": round(rmse_b_on_a, 2), "種別": "異なるマイク"},
        {"学習マイク": "混合(A+B)", "評価データ": f"{MIC_A_NAME}(全データ)", "MAE_cm": round(mae_mixed_on_a, 2), "RMSE_cm": round(rmse_mixed_on_a, 2), "種別": "混合学習"},
        {"学習マイク": "混合(A+B)", "評価データ": f"{MIC_B_NAME}(全データ)", "MAE_cm": round(mae_mixed_on_b, 2), "RMSE_cm": round(rmse_mixed_on_b, 2), "種別": "混合学習"},
        {"学習マイク": "混合(A+B)", "評価データ": "混合(検証データ)", "MAE_cm": round(val_mae_mixed, 2), "種別": "混合学習(内部検証)"},
    ])
    summary_path = os.path.join(OUTPUT_DIR, "mic_generalization_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}\n結果まとめ\n{'='*60}")
    print(summary.to_string(index=False))
    print(f"\n'{summary_path}' に保存しました。")

    gap_a = mae_a_on_b - val_mae_a
    gap_b = mae_b_on_a - val_mae_b
    print(f"\n[{MIC_A_NAME}学習→{MIC_B_NAME}評価] 同マイクとの差: {gap_a:+.2f}cm")
    print(f"[{MIC_B_NAME}学習→{MIC_A_NAME}評価] 同マイクとの差: {gap_b:+.2f}cm")
    print("(差が小さいほど、マイク機種への汎化性能が高いことを意味します)")


if __name__ == '__main__':
    main()
