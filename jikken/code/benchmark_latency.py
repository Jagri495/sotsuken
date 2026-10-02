import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

from helpers.utils import NAME_TO_WIDTH
from models.mn.model import get_model as get_mobilenet
from models.preprocess import AugmentMelSTFT

device = torch.device("cpu")
WINDOW_SECONDS = 5
SAMPLE_RATE = 32000
N_RUNS = 15
N_WARMUP = 3

def benchmark(name, forward_fn):
    # ウォームアップ(初回はキャッシュ等の影響で遅いため除外)
    for _ in range(N_WARMUP):
        forward_fn()
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        forward_fn()
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    print(f"\n=== {name} ===")
    print(f"平均: {times.mean()*1000:.1f}ms  標準偏差: {times.std()*1000:.1f}ms  "
          f"最大: {times.max()*1000:.1f}ms  最小: {times.min()*1000:.1f}ms")
    return times

# ---------- EfficientAT (mn10_as, 5クラス相当) ----------
import sys
class _Suppress:
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open("/dev/null", "w")
    def __exit__(self, *a):
        sys.stdout.close()
        sys.stdout = self._stdout

with _Suppress():
    eat_model = get_mobilenet(width_mult=NAME_TO_WIDTH("mn10_as"), head_type="mlp", num_classes=5)
eat_model.to(device).eval()
mel = AugmentMelSTFT(n_mels=128, sr=SAMPLE_RATE, win_length=800, hopsize=320)
mel.to(device).eval()

waveform = torch.from_numpy(np.random.randn(1, WINDOW_SECONDS * SAMPLE_RATE).astype(np.float32))

def eat_forward():
    with torch.no_grad():
        spec = mel(waveform).unsqueeze(1)
        eat_model(spec)

eat_times = benchmark("EfficientAT (mn10_as)", eat_forward)

# ---------- PaSST ----------
try:
    from hear21passt.base import get_basic_model, get_model_passt
    with _Suppress():
        passt_model = get_basic_model(mode="logits")
        passt_model.net = get_model_passt(arch="passt_s_swa_p16_128_ap476", n_classes=5)
    passt_model.to(device).eval()

    waveform_passt = torch.from_numpy(np.random.randn(1, WINDOW_SECONDS * SAMPLE_RATE).astype(np.float32))

    def passt_forward():
        with torch.no_grad(), _Suppress():
            passt_model(waveform_passt)

    passt_times = benchmark("PaSST (passt_s_swa_p16_128_ap476)", passt_forward)
except Exception as e:
    print(f"PaSSTのベンチマークでエラー: {e}")
    passt_times = None

# ---------- 判定 ----------
HOP_SECONDS = 0.5
print(f"\n=== HOP_SECONDS={HOP_SECONDS}秒(次の判定までの間隔)との比較 ===")
print(f"EfficientAT: 平均{eat_times.mean()*1000:.1f}ms -> "
      f"{'OK(間に合う)' if eat_times.mean() < HOP_SECONDS else 'NG(間に合わない)'}")
if passt_times is not None:
    print(f"PaSST: 平均{passt_times.mean()*1000:.1f}ms -> "
          f"{'OK(間に合う)' if passt_times.mean() < HOP_SECONDS else 'NG(間に合わない)'}")
