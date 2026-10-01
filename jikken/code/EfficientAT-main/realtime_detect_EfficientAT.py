"""
マイクからのリアルタイム音声を監視し、指定したクラス(例: シャカパチ)を検知したら
コールバック(デフォルトはログ記録+コンソール表示)を実行するプロトタイプ。

finetune.py (EfficientAT) で作成したチェックポイントを使います。

事前準備:
    pip install sounddevice soundfile librosa numpy pandas

【使い方1】実際のマイクでリアルタイム検知:
    python realtime_detect.py

【使い方2】マイクを使わず、既存の音声/動画ファイルで動作をシミュレート(ハードウェア不要のテスト用):
    python realtime_detect.py --simulate_file path/to/long_audio.wav

【設計方針】
    検知ロジックと通知ロジックを分離しています。検知したら on_detect() コールバックが
    呼ばれるだけで、その中身(ログ記録/音を鳴らす/LEDを光らせる等)は自由に差し替え可能です。
    デフォルトはログ記録(CSV)+該当区間の音声クリップ保存(あとで人手チェックするため)です。

【検知のロジック】
    1. 直近 WINDOW_SECONDS 秒の音声をバッファに保持
    2. HOP_SECONDS ごとに、バッファ全体をモデルに入力して分類
    3. 予測クラスが TARGET_CLASSES に含まれ、確信度が CONFIDENCE_THRESHOLD 以上の状態が
       MIN_CONSECUTIVE 回連続したら「検知」とみなす(誤検知・チャタリング防止のデバウンス)
"""

import argparse
import csv
import os
import queue
import threading
import time
import warnings
from collections import deque
from datetime import datetime

import librosa
import numpy as np
import soundfile as sf
import torch

from helpers.utils import NAME_TO_WIDTH
from models.dymn.model import get_model as get_dymn
from models.mn.model import get_model as get_mobilenet
from models.preprocess import AugmentMelSTFT

warnings.filterwarnings("ignore")

# ============================ 設定 (ここを変更) ============================
CHECKPOINT_PATH = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\EfficientAT-main\results\pt\Zoom_test_wavfiles_5seconds_cut_finetuned_model_epoch30.pt"   # finetune.py が保存したチェックポイント
TARGET_CLASSES = ["shakapati_front", "shakapati_back"]   # 検知対象とみなすクラス名(実際のクラス名に合わせて変更)

WINDOW_SECONDS = 5              # 学習時のCLIP_SECONDSと合わせる
HOP_SECONDS = 0.5               # 推論の間隔(短いほど反応が早いが計算負荷が上がる)
CONFIDENCE_THRESHOLD = 0.6      # この確信度以上を「検知候補」とする
MIN_CONSECUTIVE = 2             # 何回連続で検知候補になったら本検知とするか(デバウンス)
COOLDOWN_SECONDS = 3.0          # 一度検知した後、再度検知扱いするまでの最短間隔(連続アラート防止)

MIC_DEVICE = None               # マイクのデバイス番号。None なら既定のマイクを使用
BLOCK_SECONDS = 0.1             # マイクからの読み取り単位(小さいほど反応が早いがCPU負荷増)

LOG_CSV = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\EfficientAT-main\results\Zoomtestandorigin_wavfiles_5seconds_cut_epoch30_realtime_detections.csv"
SAVE_CLIP_ON_DETECT = True      # 検知時に前後の音声を保存するか(あとで人手チェックするため)
CLIP_DIR = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\EfficientAT-main\results\Zoomtestandorigin_wavfiles_5seconds_cut_epoch30_detected_clips"
CLIP_PADDING_SECONDS = 1.0      # 検知区間の前後にどれだけ余分に保存するか
# ===========================================================================


class _SuppressPrint:
    def __enter__(self):
        import sys
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        return self

    def __exit__(self, *args):
        import sys
        sys.stdout.close()
        sys.stdout = self._stdout


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    classes = ckpt["classes"]
    model_name = ckpt["model_name"]
    head_type = ckpt.get("head_type", "mlp")
    sample_rate = ckpt.get("sample_rate", 32000)

    with _SuppressPrint():
        if model_name.startswith("dymn"):
            model = get_dymn(width_mult=NAME_TO_WIDTH(model_name), num_classes=len(classes))
        else:
            model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), head_type=head_type,
                                   num_classes=len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    mel = AugmentMelSTFT(n_mels=128, sr=sample_rate, win_length=800, hopsize=320)
    mel.to(device)
    mel.eval()

    return model, mel, classes, sample_rate


def predict(model, mel, waveform, device):
    """waveform: 1次元numpy配列(1ウィンドウ分)。予測クラスIndexと確信度を返す。"""
    waveform_t = torch.from_numpy(waveform[None, :].astype(np.float32)).to(device)
    with torch.no_grad():
        spec = mel(waveform_t).unsqueeze(1)
        logits, _ = model(spec)
    import torch.nn.functional as F
    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    idx = int(np.argmax(probs))
    return idx, float(probs[idx]), probs


class DetectionLogger:
    """検知ロジックと通知ロジックを分離するためのコールバック集約クラス。"""

    def __init__(self, log_csv, save_clip, clip_dir, sample_rate):
        self.log_csv = log_csv
        self.save_clip = save_clip
        self.clip_dir = clip_dir
        self.sample_rate = sample_rate
        if save_clip:
            os.makedirs(clip_dir, exist_ok=True)

        if not os.path.exists(log_csv):
            with open(log_csv, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(["timestamp", "predicted_class", "confidence", "clip_file"])

    def on_detect(self, predicted_class, confidence, audio_clip=None):
        """検知したときに呼ばれる。ここを差し替えれば音・LEDなど別の通知方式にできる。"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        clip_filename = ""

        if self.save_clip and audio_clip is not None:
            safe_ts = now.replace(":", "-").replace(" ", "_").replace(".", "-")
            clip_filename = os.path.join(self.clip_dir, f"{safe_ts}_{predicted_class}.wav")
            sf.write(clip_filename, audio_clip, self.sample_rate)

        with open(self.log_csv, "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow([now, predicted_class, f"{confidence:.3f}", clip_filename])

        # ==== 今はコンソール表示のみ。ここを音・LED等の実出力に差し替え可能 ====
        print(f"\n🔔 検知: {predicted_class} (確信度={confidence:.3f}) at {now}")
        if clip_filename:
            print(f"   音声クリップ保存: {clip_filename}")


class RealtimeDetector:
    def __init__(self, model, mel, classes, sample_rate, device, notifier: DetectionLogger):
        self.model = model
        self.mel = mel
        self.classes = classes
        self.sample_rate = sample_rate
        self.device = device
        self.notifier = notifier

        self.window_len = int(WINDOW_SECONDS * sample_rate)
        self.clip_pad_len = int(CLIP_PADDING_SECONDS * sample_rate)

        # 十分な長さ(窓+前後クリップ用の余白)を持つバッファ
        buffer_len = self.window_len + 2 * self.clip_pad_len
        self.buffer = np.zeros(buffer_len, dtype=np.float32)

        self.consecutive_count = 0
        self.last_candidate_class = None
        self.last_detect_time = 0.0

    def push_audio(self, chunk):
        """新しい音声チャンクをリングバッファに追記する。"""
        n = len(chunk)
        self.buffer = np.roll(self.buffer, -n)
        self.buffer[-n:] = chunk

    def step(self):
        """バッファの直近WINDOW_SECONDS秒に対して1回推論し、検知判定を行う。"""
        window = self.buffer[-self.window_len:]
        idx, conf, probs = predict(self.model, self.mel, window, self.device)
        predicted_class = self.classes[idx]

        is_candidate = predicted_class in TARGET_CLASSES and conf >= CONFIDENCE_THRESHOLD

        if is_candidate and predicted_class == self.last_candidate_class:
            self.consecutive_count += 1
        elif is_candidate:
            self.last_candidate_class = predicted_class
            self.consecutive_count = 1
        else:
            self.last_candidate_class = None
            self.consecutive_count = 0

        now = time.time()
        fired = False
        if (is_candidate and self.consecutive_count >= MIN_CONSECUTIVE
                and (now - self.last_detect_time) >= COOLDOWN_SECONDS):
            fired = True
            self.last_detect_time = now
            clip = self.buffer.copy() if SAVE_CLIP_ON_DETECT else None
            self.notifier.on_detect(predicted_class, conf, audio_clip=clip)

        return predicted_class, conf, fired


def run_from_microphone(detector, sample_rate):
    import sounddevice as sd

    block_size = int(BLOCK_SECONDS * sample_rate)
    audio_q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, flush=True)
        audio_q.put(indata[:, 0].copy())

    print(f"マイクを開始します(デバイス: {MIC_DEVICE or '既定'})... Ctrl+Cで終了")
    with sd.InputStream(samplerate=sample_rate, channels=1, blocksize=block_size,
                         device=MIC_DEVICE, callback=callback):
        try:
            while True:
                chunk = audio_q.get()
                detector.push_audio(chunk)
                detector.step()
        except KeyboardInterrupt:
            print("\n終了します。")


def run_from_file(detector, sample_rate, file_path, realtime_pace=True):
    """マイクの代わりにファイルを読み込み、ブロックごとに擬似リアルタイム再生してテストする。"""
    y, sr = librosa.core.load(file_path, sr=sample_rate, mono=True)
    y = y.astype(np.float32)

    block_size = int(BLOCK_SECONDS * sample_rate)
    n_blocks = len(y) // block_size
    print(f"シミュレーション開始: {file_path} ({len(y)/sample_rate:.1f}秒, {n_blocks}ブロック)")

    hop_blocks = max(1, int(HOP_SECONDS / BLOCK_SECONDS))
    for i in range(n_blocks):
        chunk = y[i * block_size:(i + 1) * block_size]
        detector.push_audio(chunk)

        if i % hop_blocks == 0:
            pred_class, conf, fired = detector.step()
            elapsed = i * BLOCK_SECONDS
            mark = " <== FIRED" if fired else ""
            print(f"  t={elapsed:6.1f}s  pred={pred_class:20s} conf={conf:.3f}{mark}")

        if realtime_pace:
            time.sleep(BLOCK_SECONDS)

    print("\nシミュレーション終了。")


def main():
    parser = argparse.ArgumentParser(description="マイク or ファイルでリアルタイム風シャカパチ検知")
    parser.add_argument("--checkpoint_path", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--simulate_file", type=str, default=None,
                         help="マイクの代わりにこのファイルで動作をシミュレートする")
    parser.add_argument("--no_realtime_pace", action="store_true",
                         help="シミュレーション時、実時間を待たず最速で処理する(高速デバッグ用)")
    parser.add_argument("--list_devices", action="store_true", help="マイクデバイス一覧を表示して終了")
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"使用デバイス: {device}")

    model, mel, classes, sample_rate = load_model(args.checkpoint_path, device)
    print(f"クラス: {classes}")
    print(f"検知対象クラス: {TARGET_CLASSES}")

    notifier = DetectionLogger(LOG_CSV, SAVE_CLIP_ON_DETECT, CLIP_DIR, sample_rate)
    detector = RealtimeDetector(model, mel, classes, sample_rate, device, notifier)

    if args.simulate_file:
        run_from_file(detector, sample_rate, args.simulate_file,
                       realtime_pace=not args.no_realtime_pace)
    else:
        run_from_microphone(detector, sample_rate)


if __name__ == '__main__':
    main()
