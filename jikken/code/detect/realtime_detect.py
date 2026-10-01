# ==============================================================================
# モジュール全体の概要説明（ドキュメント文字列）
# ==============================================================================
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
    デフォルトはログ記録(DB detections.db + CSV)+該当区間の音声クリップ保存(あとで人手チェックするため)です。

【フロントエンド(frontend_app.py)との連携】
    frontend_app.py が実際に参照するのは db.py 経由の detections.db のみです。
    live_status.json / realtime_detections.csv (LOG_CSV) は同じ内容のバックアップ出力で、
    frontend_app.py からは読まれません(人手での確認・障害時の控え用途)。

【検知のロジック(v2: エッジトリガー方式)】
    1. 直近 WINDOW_SECONDS 秒の音声をバッファに保持
    2. HOP_SECONDS ごとに、バッファ全体をモデルに入力して分類
    3. 予測クラスが TARGET_CLASSES に含まれ、確信度が CONFIDENCE_THRESHOLD 以上、かつ
       1位/2位クラスの確率差(マージン)が MARGIN_THRESHOLD 以上の状態が MIN_CONSECUTIVE 回
       連続したら「検知候補が継続している」とみなす(マージン条件は、未知/曖昧な音でも
       1位クラスに高い確信度が出てしまう問題への対症療法として追加したもの。根本解決ではない)
    4. 検知候補の状態に "入った瞬間" (that までは検知候補ではなかった → 検知候補になった)
       だけを1回の検知として扱う。同じ音が窓に映り込み続けている間は再発火しない。
       候補で無くなった(=一旦静かになった)ことを検知して初めて、次の検知が可能になる。
       (v1では時間ベースのクールダウンだけに頼っていたため、1回の音に対して
       複数回発火するバグがあった。v2ではこれを状態(in_event)で管理して防ぐ)

【デバッグ用の可視化(realtime_detect.py のコンソール側で確認できるもの)】
    - 約2秒に1回、rms・予測クラス・確信度に加えて、推論時間・step()全体の処理時間・DB書き込み時間を表示する。
      フロントエンドへの反映が遅いと感じた場合、まずここで「realtime_detect.py側で処理が遅れていないか」を
      切り分けられる(フロントエンド側はPOLL_INTERVAL_MSごとのポーリングなので、そちらは別途フロントエンドの
      設定値を確認すること)。
    - マイク起動後、最初に音声データを受信した時点で1回だけ「🎤 最初の音声データを受信」を表示する。
      また、RMSがほぼ0の状態が続くと警告を表示する。「そもそも音を取得できているか」の切り分け用。
    - 1回のstep()がHOP_SECONDSを超えて時間がかかった場合、遅延の兆候として警告を表示する。
    - 検知が発火した瞬間、ビープ音(クラスごとに音の高さが違う)+区切り線付きのコンソール表示で
      その場で分かるようにしている(NOTIFY_SOUND_ENABLEDでON/OFF可能)。あくまで開発・デバッグ用の
      即時フィードバックであり、README 3章の「通知方式の最終形」とは別物(そちらは未確定のまま)。
"""

# ==============================================================================
# 必要なライブラリのインポート
# ==============================================================================
import argparse  # コマンドライン引数を処理するライブラリ
import csv  # CSVファイルを読み書きするライブラリ
import json  # JSONデータを扱うライブラリ
import os  # ファイルパスやOS機能にアクセスするライブラリ
import queue  # スレッド間でデータを安全にやり取りするキュー
import re  # クラス名の正規化(学習データ由来のsuffix除去)に使う正規表現ライブラリ
import time  # 時間計測やスリープを行うライブラリ
import warnings  # 警告メッセージを制御するライブラリ
from collections import deque  # 高速な両端キュー（履歴の保持に使用）
from datetime import datetime  # 日付と時刻を取得するライブラリ
from typing import Callable  # 型ヒント用(clock引数の型を明示するために使用)

import librosa  # 音声解析・処理ライブラリ
import numpy as np  # 数値計算ライブラリ
import soundfile as sf  # 音声ファイルの読み書きライブラリ
import torch  # ディープラーニングフレームワーク (PyTorch)

# プロジェクト内の別モジュールからのインポート
from helpers.utils import NAME_TO_WIDTH  # モデル名からネットワーク幅を取得する補助関数
from models.dymn.model import get_model as get_dymn  # DYMNモデルを取得する関数
from models.mn.model import get_model as get_mobilenet  # MobileNetモデルを取得する関数
from models.preprocess import AugmentMelSTFT  # 音声波形をメルスペクトログラムに変換する前処理クラス
from fastsde_model.model import SeldNetSubbandFast  # 距離推定用のFAST-SDEモデル
from db import init_db, insert_detection, save_live_status, get_live_status, get_recent_detections  # DB操作用関数
init_db()  # データベースが無ければ作成する初期化処理

warnings.filterwarnings("ignore")  # 実行時の不要な警告を非表示にする

# ============================ 設定 (ここを変更) ============================
SCRIPT_VERSION = "2026-09-28"  # このファイルを最後に編集した日付。「古いコードを実行しているかも」を
                                # 簡単に確認できるよう表示する(importされただけでも表示されるよう、
                                # モジュール読み込み時点で即表示する。main()の中だけだと、
                                # test_realtime_detect.py等からimportして使うケースで表示されず、
                                # 「どのファイルが実際に読まれているか分からない」事故につながっていた)。
print(f"[realtime_detect.py] version: {SCRIPT_VERSION} "
      f"(importされた時点で必ず表示されます。この日付が古ければ、detect/フォルダ内のこのファイル自体が"
      f"更新されていません)", flush=True)
CHECKPOINT_PATH = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\EfficientAT-main\results\pt\Data_mineandothers_finetuned_model_epoch100.pt"      # 分類モデルの学習済み重みファイルのパス
TARGET_CLASSES = ["shakapati"]   # 検知対象とするクラス名のリスト(統合後の名前。下のCLASS_ALIASES参照)

# 【9/25追加】front/backを区別せず「shakapati」として一括検知するための統合設定。
# 学習自体はfront/back別クラスのまま(再学習はしない)。統合はここ(推論後の確率の合算)だけで行う。
# 空の辞書 {} にすれば、統合前の(front/backを区別する)従来の挙動に戻せる。
CLASS_ALIASES = {
    "shakapati_front": "shakapati",
    "shakapati_back": "shakapati",
}

# --- 距離推定(FAST-SDE)を組み合わせる場合の設定 ---
DISTANCE_CHECKPOINT_PATH = r"C:\Users\Owner\Downloads\sotsuken\jikken\code\Fast_SDE\results\pt\Distance_Data_epoch_100.pt"   # 距離推定モデルの重みファイルパス（無効化する場合はNone）

WINDOW_SECONDS = 5              # 推論時に入力とする音声の長さ（秒）。学習時と合わせる
HOP_SECONDS = 0.5               # 推論を実行する間隔（秒）。短いほど高頻度に判定する
CONFIDENCE_THRESHOLD = 0.6      # AIの予測確信度がこの値以上なら「検知候補」とする
MARGIN_THRESHOLD = 0.2          # 1位クラスと2位クラスの確率差。これ未満は「僅差で選ばれただけ」とみなし候補から除外する。
                                 # 未知/曖昧な音でも1位クラスに高い確信度が出てしまう問題への対症療法(仮の緩和策)。
                                 # 根本的な未知音検出(OOD対策)ではない点に注意。0にすると従来通り無効化できる。
MIN_CONSECUTIVE = 2             # 何回連続で検知候補になったら「本検知」とするか(誤検知防止)
COOLDOWN_SECONDS = 1.0          # 次の新しいイベントとして検知するまでの最短間隔（秒）

MIC_DEVICE = None               # 使用するマイクのデバイス番号。None ならOSの既定マイクを使用
BLOCK_SECONDS = 0.1             # マイクから1回に読み取る音声の長さ（秒）

# --- 無音ゲーティング(重要: 「無音でも検知してしまう」問題への対策) ---
SILENCE_GATING_ENABLED = True    # 音量が小さい時に判定をスキップする機能のON/OFF
# 【9/26変更】無音環境でも検知してしまう問題への対策として有効化。
# 実測のRMSログ(live_status.json)では静かな部屋でRMS=0.001〜0.003程度だったため、
# 下のSILENCE_RMS_THRESHOLD=0.01は妥当な初期値と考えられる。ただし部屋・マイクの機種・
# ゲイン設定によって適正値は変わるので、起動時コンソールの[状態] rms=... ログを見て、
# 「本当に無音のときのrms」より少し上の値になるよう、必要なら調整すること。
SILENCE_RMS_THRESHOLD = 0.01    # この値未満の音量(RMS)は「無音」とみなし推論をスキップする閾値

# --- マイクから音声信号そのものが届いているかを確認するための簡易ウォッチドッグ ---
# 「音を取得できているか怪しい」というデバッグ用途。周囲が静かなだけならRMSはこれより十分大きい値になる。
NO_SIGNAL_RMS_THRESHOLD = 1e-6  # これより小さいRMSが続く場合、マイクの信号自体が来ていない可能性が高い
NO_SIGNAL_WARN_STREAK = 20      # 何ステップ連続でNO_SIGNAL_RMS_THRESHOLD未満だったら警告を出すか

# --- フロントエンド連携用のライブ状態出力 ---
# 実際にフロントエンド(frontend_app.py)が読むのは detections.db (db.DB_PATH)。
# LIVE_STATUS_PATH / LOG_CSV は同じ内容のバックアップ出力で、frontend_app.py は読まない。
LIVE_STATUS_PATH = "live_status.json"   # 現在の状態を書き出すJSONファイルパス(バックアップ用)
HISTORY_LENGTH = 150                    # フロントエンドに波形表示として渡す、過去の音量データの保持件数

LOG_CSV = "realtime_detections.csv"     # 検知履歴を記録するCSVファイルのパス(バックアップ用)
SAVE_CLIP_ON_DETECT = True      # 検知した瞬間の音声をWAVファイルとして保存するかどうか
CLIP_DIR = "detected_clips"     # 保存するWAVファイルの出力先フォルダ
CLIP_PADDING_SECONDS = 1.0      # 検知音声の前後何秒を余分にWAVファイルに含めるか

# --- 検知時の即時フィードバック(開発・デバッグ用) ---
# 【重要】これはあくまで「テスト中に検知した瞬間をその場で分かるようにする」ための開発者向けの
# 仕掛けであり、README 3章「通知方式の最終形」(音・LED・審判経由など、卒論として未確定の研究課題)
# とは別物。個人特定不可の制約下でどう実運用の通知を設計するかという論点はまだ何も決めていない。
NOTIFY_SOUND_ENABLED = True     # 検知した瞬間にビープ音を鳴らすか(Windows専用。他OSでは自動的に無音)
NOTIFY_SOUND_BY_CLASS = {       # 【9/25更新】front/backの統合(CLASS_ALIASES)により、区別する意味が
                                # 無くなったので"shakapati"1つにまとめた。CLASS_ALIASESを{}にして
                                # front/back区別に戻す場合は、ここも元のように分けること。
    "shakapati": (1000, 150),  # (周波数Hz, 長さms)
}
NOTIFY_SOUND_DEFAULT = (900, 150)  # 上記に無いクラスが検知された場合のフォールバック音
# ===========================================================================


# ------------------------------------------------------------------------------
# コンソール出力を一時的に抑制するコンテキストマネージャー（モデルロード時のログ消し用）
# ------------------------------------------------------------------------------
class _SuppressPrint:
    def __enter__(self):
        import sys
        self._stdout = sys.stdout  # 元の標準出力を退避
        sys.stdout = open(os.devnull, "w")  # 標準出力を /dev/null (破棄) に向ける
        return self

    def __exit__(self, *args):
        import sys
        sys.stdout.close()  # ダミーの出力を閉じる
        sys.stdout = self._stdout  # 元の標準出力を復元する


# ------------------------------------------------------------------------------
# クラス名の正規化(学習データ由来のsuffix除去)
# ------------------------------------------------------------------------------
# 学習データが "shakapati_front_5seconds_cut.wav" のように切り出し秒数付きのファイル名で
# 管理されていたため、チェックポイントに保存されているクラス名にもこのsuffixがそのまま
# 残っている場合がある(1秒/3秒/5秒窓で学習したモデルそれぞれで異なるsuffixになり得る)。
# TARGET_CLASSESやground_truth.csvはサフィックス無しの名前で統一したいので、
# モデル読み込み時にここで正規化する。
_CLIP_SUFFIX_PATTERN = re.compile(r"_\d+seconds?_cut$", re.IGNORECASE)


def strip_clip_suffix(name):
    """"shakapati_front_5seconds_cut" のようなクラス名から末尾のsuffixを除去して
    "shakapati_front" にする。該当するsuffixが無ければ、そのまま返す。"""
    return _CLIP_SUFFIX_PATTERN.sub("", name)


# ------------------------------------------------------------------------------
# 音声分類モデルを読み込む関数
# ------------------------------------------------------------------------------
def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)  # 指定デバイス(CPU/GPU)に重みをロード
    classes_raw = ckpt["classes"]  # モデルが学習したクラスのリストを取得(チェックポイントそのままの名前)
    classes = [CLASS_ALIASES.get(strip_clip_suffix(c), strip_clip_suffix(c)) for c in classes_raw]
    # 学習データ由来のsuffixを除去し(strip_clip_suffix)、さらにfront/back等の統合(CLASS_ALIASES)を適用。
    # 結果としてclassesの中に同じ名前("shakapati"等)が複数回出てくることがあるが、
    # これは意図した挙動(predict()側で確率を合算するために使う)。
    if classes != classes_raw:
        # 【9/25発見の重大バグの修正箇所】チェックポイントのクラス名に"_5seconds_cut"等のsuffixが
        # 付いたまま(例: "shakapati_front_5seconds_cut")使うと、TARGET_CLASSES(サフィックス無し)と
        # 一生マッチせず、is_candidateが常にFalseになる=検知が絶対に発火しない、という重大な不具合が
        # 実際に発生していた(9/25、本番運用で一度も検知が記録されていないことから発覚)。
        print(f"  ℹ️ チェックポイントのクラス名を正規化しました: {classes_raw} → {classes}", flush=True)
    model_name = ckpt["model_name"]  # 使用されているモデルのアーキテクチャ名を取得
    head_type = ckpt.get("head_type", "mlp")  # 分類ヘッドの種類を取得（デフォルトはmlp）
    sample_rate = ckpt.get("sample_rate", 32000)  # 学習時のサンプリングレートを取得（デフォルト32kHz）

    with _SuppressPrint():  # モデル構築時の不要なprintを隠す
        if model_name.startswith("dymn"):
            # DYMNモデルの場合のインスタンス化
            model = get_dymn(width_mult=NAME_TO_WIDTH(model_name), num_classes=len(classes))
        else:
            # MobileNet/EfficientATモデルの場合のインスタンス化
            model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), head_type=head_type,
                                   num_classes=len(classes))
    model.load_state_dict(ckpt["model_state_dict"])  # モデルに重みを適用
    model.to(device)  # モデルを指定デバイスに転送
    model.eval()  # モデルを推論モード（評価モード）に設定

    # 音声波形をメルスペクトログラムに変換する層の設定
    mel = AugmentMelSTFT(n_mels=128, sr=sample_rate, win_length=800, hopsize=320)
    mel.to(device)
    mel.eval()

    return model, mel, classes, sample_rate  # ロードしたモデル一式を返す


# ------------------------------------------------------------------------------
# 距離推定モデルを読み込む関数
# ------------------------------------------------------------------------------
def load_distance_model(checkpoint_path, device):
    """train_distance.py (FAST-SDE) が保存したチェックポイントを読み込む。
    ファイルが無い場合は None を返し、距離推定機能を無効化する。"""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None, None, None  # ファイルがない場合は無効化

    ckpt = torch.load(checkpoint_path, map_location=device)  # 重みをロード
    sample_rate = ckpt.get("sample_rate", 16000)  # 距離モデル用のサンプリングレート（デフォルト16kHz）
    clip_seconds = ckpt.get("clip_seconds", 3.0)  # 入力音声の長さ（デフォルト3.0秒）

    # 距離推定用モデル SeldNetSubbandFast のインスタンス化
    model = SeldNetSubbandFast(
        features_set="all", n_subbands=6, c_mid=16, n_blocks=2,
        fuse_c=48, use_gru=False, att_conf="Nothing",
    )
    model.load_state_dict(ckpt["model_state_dict"])  # 重みを適用
    model.to(device)
    model.eval()

    return model, sample_rate, clip_seconds  # モデルと設定を返す


# ------------------------------------------------------------------------------
# 音声から距離を推定する関数
# ------------------------------------------------------------------------------
def estimate_distance(distance_model, waveform, orig_sr, target_sr, clip_seconds, device):
    """検知クリップ(orig_sr, 任意長)を距離推定モデルの入力形式(target_sr, clip_seconds)に変換して推論する。"""
    if orig_sr != target_sr:
        # 入力音声のサンプリングレートがモデルの要求と異なる場合はリサンプリング
        waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)

    target_len = int(target_sr * clip_seconds)  # モデルが要求するデータ長(サンプル数)を計算
    if len(waveform) >= target_len:
        waveform = waveform[:target_len]  # 長すぎる場合は切り詰める
    else:
        # 短すぎる場合はゼロパディング(無音追加)で長さを合わせる
        waveform = np.concatenate([waveform, np.zeros(target_len - len(waveform), dtype=np.float32)])

    # numpy配列をPyTorchテンソルに変換しデバイスへ転送
    waveform_t = torch.from_numpy(waveform[None, :].astype(np.float32)).to(device)
    with torch.no_grad():  # 勾配計算を無効化（メモリ節約と高速化）
        pred_cm = distance_model(waveform_t).item()  # 推論を実行し、Pythonの数値(cm)として取得

    # 【9/28追加】距離は物理的に負の値になり得ない。学習データの収録範囲(30〜200cm程度)を
    # 大きく外れる値が出た場合も含め、モデルの出力が信頼できない可能性を明示的に警告する
    # (値を勝手に補正・クリップはしない。異常値そのものが「学習データの範囲外の入力」等の
    # 診断情報になるため、生の値のまま返す)。
    if pred_cm < 0 or pred_cm > 300:
        print(f"  ⚠️ 距離推定の出力が学習データの想定範囲(30〜200cm程度)を外れています: "
              f"{pred_cm:.1f}cm。モデルの精度検証(predict_distance.pyでのMAE/RMSE算出)が"
              f"まだ済んでいないため、この値は参考程度に留めてください。", flush=True)
    return pred_cm


# ------------------------------------------------------------------------------
# 音声がどのクラスに属するか予測する関数
# ------------------------------------------------------------------------------
def predict(model, mel, waveform, device, classes):
    """waveform: 1次元numpy配列(1ウィンドウ分)。統合後のクラス名・確信度・
    統合後の確率配列(降順)を返す。

    【9/25設計変更】以前は生の5クラス分類でargmaxを取ってから、呼び出し側で
    self.classes[idx]をCLASS_ALIASESで文字列変換するだけだった。しかしそれだと、
    例えば shakapati_front=0.35, shakapati_back=0.35, Deal_shuffle=0.40 のようなケースで、
    本来なら「shakapati」の方が合計確信度が高い(0.35+0.35=0.70 > 0.40)のに、
    生のargmaxはDeal_shuffleを選んでしまう。これを避けるため、ここでクラス名ごとに
    確率を合算してから argmax を取るように変更した。"""
    waveform_t = torch.from_numpy(waveform[None, :].astype(np.float32)).to(device)
    with torch.no_grad():
        spec = mel(waveform_t).unsqueeze(1)  # 音声波形をメルスペクトログラムに変換
        logits, _ = model(spec)  # 分類モデルにスペクトログラムを入力し、生スコア(ロジット)を得る
    import torch.nn.functional as F
    raw_probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()  # 生の(統合前)確率配列

    # クラス名ごとに確率を合算する(classesに同じ名前が複数あれば、ここで合算される)
    merged = {}
    for i, class_name in enumerate(classes):
        merged[class_name] = merged.get(class_name, 0.0) + float(raw_probs[i])

    predicted_class = max(merged, key=merged.get)  # 合算後の確率が最も高いクラス名
    conf = merged[predicted_class]
    merged_probs = np.array(sorted(merged.values(), reverse=True))  # margin計算用に降順で並べる

    return predicted_class, float(conf), merged_probs  # 統合後クラス名、確信度、確率配列(降順)を返す


# ------------------------------------------------------------------------------
# 検知した瞬間にその場で分かるようにするための即時フィードバック(開発・デバッグ用)
# ------------------------------------------------------------------------------
_sound_warning_shown = False  # winsoundが無い環境で警告を1回だけ出すためのフラグ


def play_detection_beep(predicted_class):
    """検知した瞬間にビープ音を鳴らす。Windows専用(winsoundは標準ライブラリなので追加インストール不要)。
    Windows以外の環境や、音が鳴らせない環境(サーバー等)では、エラーにせず黙ってスキップする。
    クラスごとに音の高さを変えているので、画面を見ていなくても「front/backどちらが鳴ったか」を
    ある程度聞き分けられる。"""
    global _sound_warning_shown
    if not NOTIFY_SOUND_ENABLED:
        return
    freq, duration_ms = NOTIFY_SOUND_BY_CLASS.get(predicted_class, NOTIFY_SOUND_DEFAULT)
    try:
        import winsound  # Windows専用の標準ライブラリ。importをここに置き、他OSでの起動失敗を避ける
        winsound.Beep(freq, duration_ms)
    except (ImportError, RuntimeError, OSError):
        # winsoundが無い(Windows以外)、またはサウンドデバイスが無い環境。検知処理自体は止めない
        if not _sound_warning_shown:
            print("  ℹ️ ビープ音が鳴らせない環境のようです(winsound非対応、またはサウンドデバイス無し)。"
                  "NOTIFY_SOUND_ENABLED = False にすれば、この確認は不要になります。", flush=True)
            _sound_warning_shown = True


# ------------------------------------------------------------------------------
# 検知時の記録・通知処理をまとめたクラス
# ------------------------------------------------------------------------------
class DetectionLogger:
    """検知ロジックと通知ロジックを分離するためのコールバック集約クラス。"""

    def __init__(self, log_csv, save_clip, clip_dir, sample_rate,
                 distance_model=None, distance_sample_rate=None, distance_clip_seconds=None,
                 device="cpu"):
        # 初期化時に各種設定とモデルを保持
        self.log_csv = log_csv
        self.save_clip = save_clip
        self.clip_dir = clip_dir
        self.sample_rate = sample_rate
        self.distance_model = distance_model
        self.distance_sample_rate = distance_sample_rate
        self.distance_clip_seconds = distance_clip_seconds
        self.device = device
        
        if save_clip:
            os.makedirs(clip_dir, exist_ok=True)  # 音声保存用のフォルダがなければ作成する

        if not os.path.exists(log_csv):
            # ログ用CSVが存在しなければ、ヘッダー行を書き込んで新規作成
            with open(log_csv, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(
                    ["timestamp", "predicted_class", "confidence", "estimated_distance_cm", "clip_file"])

    def on_detect(self, predicted_class, confidence, audio_clip=None):
        """検知したときに呼ばれる。ここを差し替えれば音・LEDなど別の通知方式にできる。"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 現在時刻をミリ秒まで取得
        clip_filename = ""

        # 「その場で分かる」ための即時フィードバック(開発・デバッグ用)を最優先で鳴らす。
        # ファイル書き込み等より前に呼ぶことで、体感の遅延をできるだけ小さくする。
        play_detection_beep(predicted_class)

        if self.save_clip and audio_clip is not None:
            # ファイル名に使えない文字を置換し、WAVファイルとして保存
            safe_ts = now.replace(":", "-").replace(" ", "_").replace(".", "-")
            clip_filename = os.path.join(self.clip_dir, f"{safe_ts}_{predicted_class}.wav")
            sf.write(clip_filename, audio_clip, self.sample_rate)

        # 距離推定(FAST-SDE)。DISTANCE_CHECKPOINT_PATHが未設定/読み込み失敗なら None のまま
        distance_cm = None
        if self.distance_model is not None and audio_clip is not None:
            distance_cm = estimate_distance(  # 距離推定関数を呼び出し
                self.distance_model, audio_clip, self.sample_rate,
                self.distance_sample_rate, self.distance_clip_seconds, self.device)

        distance_str = f"{distance_cm:.1f}" if distance_cm is not None else ""  # CSV書き込み用の文字列表現

        # CSVに今回の検知データを追記
        with open(self.log_csv, "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow([now, predicted_class, f"{confidence:.3f}", distance_str, clip_filename])

        # ==== 画面表示でも一目で分かるよう、区切り線で目立たせる。実出力(LED等)はまだ未実装 ====
        distance_msg = f", 推定距離={distance_cm:.1f}cm" if distance_cm is not None else ""
        print(f"\n{'=' * 50}")
        print(f"🔔 検知: {predicted_class} (確信度={confidence:.3f}{distance_msg}) at {now}")
        print(f"{'=' * 50}")
        if clip_filename:
            print(f"   音声クリップ保存: {clip_filename}")
            
        # 検知履歴をDB (detections.db) にもインサート。CSV (LOG_CSV) は
        # バックアップ用途のみで、frontend_app.py はこのCSVを読まずDBだけを参照する。
        insert_detection(
            timestamp=now,
            predicted_class=predicted_class,
            confidence=confidence,
            estimated_distance_cm=distance_cm,
            rms=float(np.sqrt(np.mean(audio_clip.astype(np.float64) ** 2))) if audio_clip is not None else None
        )


# ------------------------------------------------------------------------------
# リアルタイムの音声検知を管理するメインロジッククラス
# ------------------------------------------------------------------------------
class RealtimeDetector:
    def __init__(self, model, mel, classes, sample_rate, device, notifier: DetectionLogger,
                 clock: Callable[[], float] = time.time,
                 enable_live_status: bool = True):
        """clock / enable_live_status はどちらもテスト用デモ(test_realtime_detect.py)から
        差し替えるためのフック。デフォルト値は今までの本番動作(壁時計・DB書き込み有効)と
        完全に同じなので、これらを指定しなければ挙動は一切変わらない。
        - clock: 「現在時刻」の代わりに使う関数。既定は time.time (壁時計)。
          評価用デモでは「音声ファイル上の経過秒数」を返す関数に差し替え、
          --no_realtime_pace的な高速シミュレーションでもCOOLDOWN_SECONDSの
          判定が実時間経過とズレない(=誤って検知を取りこぼさない)ようにする。
        - enable_live_status: Falseにすると _write_live_status() が何もしない
          (detections.db / live_status.json を一切書き換えない)。評価用デモの実行で
          本番の状態を汚さないようにするためのフラグ。
        """
        self.model = model
        self.mel = mel
        self.classes = classes
        self.sample_rate = sample_rate
        self.device = device
        self.notifier = notifier  # 記録・通知用のロガーを保持
        self._clock = clock                        #
        self.enable_live_status = enable_live_status  #

        # 推論用ウィンドウの長さ(サンプル数)と、保存用余白の長さを計算
        self.window_len = int(WINDOW_SECONDS * sample_rate)
        self.clip_pad_len = int(CLIP_PADDING_SECONDS * sample_rate)

        # 直近の音声を保持するためのリングバッファをゼロで初期化
        buffer_len = self.window_len + 2 * self.clip_pad_len
        self.buffer = np.zeros(buffer_len, dtype=np.float32)

        self.consecutive_count = 0  # 検知候補が連続した回数
        self.last_candidate_class = None  # 直前の推論で候補となったクラス
        self.in_event = False       # 現在、検知イベントの"最中"かどうか(v2の追加分)
        self.last_detect_time = 0.0  # 最後に発火した時刻

        # フロントエンドの波形風表示用: 直近HISTORY_LENGTH件の(時刻, RMS, 検知中かどうか)を保持
        self.rms_history = deque(maxlen=HISTORY_LENGTH)

        # --- デバッグ用の計測値(realtime_detect.py側のコンソールで確認するためのもの) ---
        self.last_infer_ms = 0.0     # 直近の推論(predict呼び出し)1回にかかった時間
        self.last_step_ms = 0.0      # 直近のstep()呼び出し全体(推論+記録)にかかった時間
        self.last_db_write_ms = 0.0  # 直近のライブ状態DB書き込み(save_live_status)にかかった時間
        self._no_signal_streak = 0   # RMSがNO_SIGNAL_RMS_THRESHOLD未満で連続した回数(信号未検出の検出用)

    def push_audio(self, chunk):
        """新しい音声チャンクをリングバッファに追記する。"""
        n = len(chunk)
        self.buffer = np.roll(self.buffer, -n)  # 配列を左にシフト（古いデータを押し出す）
        self.buffer[-n:] = chunk  # 末尾に新しい音声データを入れる

    def step(self):
        """バッファの直近WINDOW_SECONDS秒に対して1回推論し、検知判定を行う。"""
        step_start = time.perf_counter()  # この1回のstep()全体の処理時間を計測するための開始時刻

        window = self.buffer[-self.window_len:]  # バッファから推論に必要な長さだけ切り出す

        # 切り出した音声のRMS（二乗平均平方根）を計算して音量を求める
        rms = float(np.sqrt(np.mean(window.astype(np.float64) ** 2)))
        # 無音ゲーティングがONで、かつ音量が閾値未満なら無音と判定
        is_silence = SILENCE_GATING_ENABLED and rms < SILENCE_RMS_THRESHOLD

        # --- 「音を取得できているか怪しい」を切り分けるためのウォッチドッグ ---
        # 無音ゲーティングの設定に関わらず、RMSがほぼ完全に0(=信号が来ていない)状態が
        # 続いていないかを常にチェックする(単に部屋が静かなだけならこの閾値は下回らない)。
        if rms < NO_SIGNAL_RMS_THRESHOLD:
            self._no_signal_streak += 1
        else:
            self._no_signal_streak = 0
        if self._no_signal_streak == NO_SIGNAL_WARN_STREAK:
            print(f"  ⚠️ 直近{NO_SIGNAL_WARN_STREAK}ステップ、RMSがほぼ0です(信号なし)。"
                  f"マイクが正しく接続・選択されているか確認してください"
                  f"(--list_devices でデバイス一覧を確認できます)。", flush=True)

        infer_ms = 0.0
        margin = None  # 1位クラスと2位クラスの確率差(未知音の棄却に使う)

        if is_silence:
            # 無音とみなし、モデル判定自体を行わない(誤検知の元を断つ + 計算コストも節約できる)
            predicted_class, conf = None, 0.0
            self.last_candidate_class = None
            self.consecutive_count = 0
            self.in_event = False  # 無音になればイベントは終了
        else:
            # 音声があるので推論を実行する(推論だけにかかった時間を分けて計測)
            infer_start = time.perf_counter()
            predicted_class, conf, probs = predict(self.model, self.mel, window, self.device, self.classes)
            infer_ms = (time.perf_counter() - infer_start) * 1000

            # probsは predict() が既に降順(確率が高い順)で返してくる(統合後の確率配列)。
            # マージンが小さい = モデルが僅差で1位を選んでいるだけ = 未知/曖昧な音である可能性が高い。
            margin = float(probs[0] - probs[1]) if len(probs) >= 2 else float(probs[0])

            # 予測クラスがターゲットに含まれ、確信度が閾値以上、かつマージンも十分ある場合のみ「候補」
            is_candidate = (predicted_class in TARGET_CLASSES
                             and conf >= CONFIDENCE_THRESHOLD
                             and margin >= MARGIN_THRESHOLD)

            if is_candidate and predicted_class == self.last_candidate_class:
                self.consecutive_count += 1  # 同じクラスの候補が連続していればカウントアップ
            elif is_candidate:
                self.last_candidate_class = predicted_class
                self.consecutive_count = 1   # 新しいクラスの候補ならカウントを1にリセット
            else:
                # 候補で無くなった = 今のイベントは終わった。次の検知を許可する状態に戻す。
                self.last_candidate_class = None
                self.consecutive_count = 0
                self.in_event = False

        now = self._clock()  # 壁時計(既定)、または評価用デモが差し替えた「音声上の経過秒数」
        fired = False  # 今回新しく発火（検知判定）したかどうかのフラグ

        # 「候補が続いている」かつ「まだこのイベントで発火していない」かつ「クールダウン経過」の
        # 3条件がそろった"最初の瞬間"だけ発火する(=エッジトリガー)。無音時は is_candidate 自体が
        # 成立しないケース(predicted_class=None)なので、自動的にここはスキップされる。
        # is_candidate の判定条件(TARGET_CLASSES / CONFIDENCE_THRESHOLD / MARGIN_THRESHOLD)は
        # 上の候補判定と完全に同じ条件をここでも使う(条件がズレると数え間違いの原因になるため)。
        if (not is_silence and predicted_class in TARGET_CLASSES and conf >= CONFIDENCE_THRESHOLD
                and margin is not None and margin >= MARGIN_THRESHOLD
                and self.consecutive_count >= MIN_CONSECUTIVE
                and not self.in_event
                and (now - self.last_detect_time) >= COOLDOWN_SECONDS):
            fired = True
            self.in_event = True  # イベント中状態に移行
            self.last_detect_time = now
            clip = self.buffer.copy() if SAVE_CLIP_ON_DETECT else None  # 保存用ならバッファをコピー
            notify_start = time.perf_counter()
            self.notifier.on_detect(predicted_class, conf, audio_clip=clip)  # ロガーに通知
            notify_ms = (time.perf_counter() - notify_start) * 1000
            print(f"   ⏱ 記録処理(CSV/DB/クリップ保存/距離推定)にかかった時間: {notify_ms:.1f}ms", flush=True)

        now_str = datetime.now().strftime("%H:%M:%S")
        # 波形表示用のRMS履歴をキューに追加
        self.rms_history.append({"t": now_str, "rms": round(rms, 5), "in_event": self.in_event})

        # フロントエンド向けに現在の情報をJSONとして書き出す
        self._write_live_status(rms, is_silence, predicted_class, conf, fired)

        # このstep()全体(推論+DB/ファイル書き込み)にかかった時間を記録。
        # HOP_SECONDSより明らかに長い場合は、リアルタイム処理に追いつけていない(遅延が蓄積する)兆候。
        self.last_infer_ms = infer_ms
        self.last_step_ms = (time.perf_counter() - step_start) * 1000
        if self.last_step_ms > HOP_SECONDS * 1000:
            print(f"  ⚠️ 1回の処理に{self.last_step_ms:.0f}ms かかりました"
                  f"(目標のHOP_SECONDS={HOP_SECONDS * 1000:.0f}ms を超過)。"
                  f"処理が音声入力に追いつけていない可能性があります。", flush=True)

        return predicted_class, conf, fired

    def _write_live_status(self, rms, is_silence, predicted_class, conf, fired):
        """フロントエンド(frontend_app.py)が読み取るための現在状態をJSONに書き出す。
        書き込み失敗(権限等)があっても検知処理自体は止めない。"""
        if not self.enable_live_status:
            return  # 評価用デモ実行時など、DB/JSONを一切汚したくない場合はここで即終了
        if not LIVE_STATUS_PATH:
            return  # パスが未設定ならスキップ
        status = {  # 出力する状態データの辞書を作成
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "rms": round(rms, 5),
            "is_silence": is_silence,
            "predicted_class": predicted_class,
            "confidence": round(conf, 3) if predicted_class else None,
            "in_event": self.in_event,
            "fired": fired,
            "target_classes": TARGET_CLASSES,
            "silence_threshold": SILENCE_RMS_THRESHOLD,
            "rms_history": list(self.rms_history),   # 波形風表示用の直近履歴(古い→新しいの順)
        }
        # frontend_app.py が実際に参照するのはこちら(detections.db)。書き込み時間を計測しておく。
        db_start = time.perf_counter()
        save_live_status(status)  # DB (detections.db) に保存
        self.last_db_write_ms = (time.perf_counter() - db_start) * 1000
        # live_status.json への書き込みはバックアップ用途のみ。frontend_app.py はこのファイルを読まない。
        try:
            tmp_path = LIVE_STATUS_PATH + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(status, f, ensure_ascii=False)  # JSONとして一時ファイルに保存
            os.replace(tmp_path, LIVE_STATUS_PATH)  # 読み取り中の中途半端なファイルを避けるため一時ファイル経由
        except OSError:
            pass  # ファイルの書き込みエラーは無視して継続


# ------------------------------------------------------------------------------
# マイクからの入力で動作させる関数
# ------------------------------------------------------------------------------
def run_from_microphone(detector, sample_rate):
    import sounddevice as sd  # マイク制御ライブラリ

    block_size = int(BLOCK_SECONDS * sample_rate)  # 1ブロックあたりのサンプル数
    audio_q = queue.Queue()  # 録音スレッドとメインスレッド間でデータを受け渡すキュー

    def callback(indata, frames, time_info, status):
        # マイクからデータが来たときに呼ばれるコールバック
        if status:
            print(status, flush=True)  # エラーがあれば表示
        audio_q.put(indata[:, 0].copy())  # 左チャンネルのデータをキューに入れる

    print(f"マイクを開始します(デバイス: {MIC_DEVICE or '既定'})... Ctrl+Cで終了")
    print(f"無音ゲーティング: {'有効' if SILENCE_GATING_ENABLED else '無効'} "
          f"(閾値 RMS<{SILENCE_RMS_THRESHOLD})")
    print(f"推論間隔(HOP_SECONDS): {HOP_SECONDS}秒ごと / マイク読み取り単位(BLOCK_SECONDS): {BLOCK_SECONDS}秒")

    # run_from_file() と同じ考え方で、BLOCK_SECONDS単位のブロックを何個受け取ったら
    # 1回推論するかを計算する(=HOP_SECONDSごとに間引く)。
    # 【修正】以前はここでの間引きが無く、毎ブロック(BLOCK_SECONDS=0.1秒)ごとに推論とDB/JSON書き込みが
    # 走っていた(意図したHOP_SECONDS=0.5秒の5倍の頻度)。無駄な推論負荷・I/O負荷の原因になっていたため、
    # run_from_file() と同じ間引き方式に統一した。
    hop_blocks = max(1, int(HOP_SECONDS / BLOCK_SECONDS))
    status_every_blocks = max(1, int(2.0 / BLOCK_SECONDS))  # 約2秒に1回、状態を表示する間隔(ブロック数換算)

    block_count = 0
    step_count = 0
    first_chunk_received = False
    last_pred_class, last_conf = None, 0.0

    # オーディオストリーム（録音）の開始
    with sd.InputStream(samplerate=sample_rate, channels=1, blocksize=block_size,
                         device=MIC_DEVICE, callback=callback):
        try:
            while True:  # ずっとループして処理
                chunk = audio_q.get()  # キューから音声チャンクを取り出す（データが来るまで待機）

                if not first_chunk_received:
                    # 「音を取得できているか怪しい」を早期に切り分けるため、最初の1回だけ明示的に表示する。
                    first_chunk_received = True
                    first_rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
                    print(f"  🎤 マイクから最初の音声データを受信しました(RMS={first_rms:.4f})。"
                          f"この値が常に0に近い場合はマイク未接続/ミュート/デバイス選択違いの可能性があります。",
                          flush=True)

                detector.push_audio(chunk)  # 音声をバッファに追加
                block_count += 1

                if block_count % hop_blocks == 0:  # HOP_SECONDSごとにだけ推論・記録を行う
                    pred_class, conf, fired = detector.step()  # 推論処理を1回進める
                    step_count += 1
                    last_pred_class, last_conf = pred_class, conf

                # 約2秒に1回、現在の状態を1行で表示(検知イベント以外は基本ここだけが目に見える情報になる)
                # 処理時間(推論・DB書き込み)もあわせて表示し、フロントエンドへの反映が遅い場合に
                # 原因がここ(realtime_detect.py側)にあるかどうかをこの場で切り分けられるようにする。
                if block_count % status_every_blocks == 0:
                    window = detector.buffer[-detector.window_len:]
                    rms_val = float(np.sqrt(np.mean(window.astype(np.float64) ** 2)))
                    display_class = last_pred_class if last_pred_class is not None else "(silence)"
                    print(f"  [状態] rms={rms_val:.4f}  pred={display_class}  conf={last_conf:.3f}  "
                          f"推論={detector.last_infer_ms:.1f}ms  step合計={detector.last_step_ms:.1f}ms  "
                          f"DB書込={detector.last_db_write_ms:.1f}ms", flush=True)
        except KeyboardInterrupt:  # Ctrl+C が押された時
            print("\n終了します。")


# ------------------------------------------------------------------------------
# 既存の音声ファイルを使ってシミュレーション動作させる関数
# ------------------------------------------------------------------------------
def run_from_file(detector, sample_rate, file_path, realtime_pace=True):
    """マイクの代わりにファイルを読み込み、ブロックごとに擬似リアルタイム再生してテストする。"""
    y, sr = librosa.core.load(file_path, sr=sample_rate, mono=True)  # ファイル全体を読み込む
    y = y.astype(np.float32)

    block_size = int(BLOCK_SECONDS * sample_rate)
    n_blocks = len(y) // block_size  # 全体をブロックサイズで分割
    print(f"シミュレーション開始: {file_path} ({len(y)/sample_rate:.1f}秒, {n_blocks}ブロック)")

    hop_blocks = max(1, int(HOP_SECONDS / BLOCK_SECONDS))  # 何ブロックごとに推論を走らせるかの計算
    for i in range(n_blocks):
        chunk = y[i * block_size:(i + 1) * block_size]  # ブロックごとに切り出し
        detector.push_audio(chunk)

        if i % hop_blocks == 0:  # 指定のホップ間隔に達したら推論
            pred_class, conf, fired = detector.step()
            elapsed = i * BLOCK_SECONDS  # 経過時間（秒）
            mark = " <== FIRED" if fired else ""  # 発火した時の目印
            display_class = pred_class if pred_class is not None else "(silence)"
            rms = detector.buffer[-detector.window_len:]
            rms_val = float(np.sqrt(np.mean(rms.astype(np.float64) ** 2)))
            print(f"  t={elapsed:6.1f}s  rms={rms_val:.4f}  pred={display_class:20s} conf={conf:.3f} "
                  f"推論={detector.last_infer_ms:.1f}ms{mark}")

        if realtime_pace:  # 実時間シミュレートが有効な場合
            time.sleep(BLOCK_SECONDS)  # ブロックの長さ分だけわざと待機する

    print("\nシミュレーション終了。")


# ------------------------------------------------------------------------------
# スクリプト実行時のエントリーポイント
# ------------------------------------------------------------------------------
def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="マイク or ファイルでリアルタイム風シャカパチ検知")
    parser.add_argument("--checkpoint_path", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--simulate_file", type=str, default=None,
                         help="マイクの代わりにこのファイルで動作をシミュレートする")
    parser.add_argument("--no_realtime_pace", action="store_true",
                         help="シミュレーション時、実時間を待たず最速で処理する(高速デバッグ用)")
    parser.add_argument("--list_devices", action="store_true", help="マイクデバイス一覧を表示して終了")
    parser.add_argument("--distance_checkpoint_path", type=str, default=DISTANCE_CHECKPOINT_PATH,
                         help="距離推定(FAST-SDE)のチェックポイント。指定しない/見つからない場合は距離推定を行わない")
    args = parser.parse_args()

    if args.list_devices:
        # --list_devices が指定されたらマイク一覧を表示して終了する
        import sounddevice as sd
        print(sd.query_devices())
        return

    # CUDA(GPU)が使えるか判定してデバイスを決定
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"使用デバイス: {device}")

    # 分類用モデルのロード
    model, mel, classes, sample_rate = load_model(args.checkpoint_path, device)
    print(f"クラス: {classes}")
    print(f"検知対象クラス: {TARGET_CLASSES}")

    # 距離推定用モデルのロード
    distance_model, distance_sr, distance_clip_sec = load_distance_model(
        args.distance_checkpoint_path, device)
    if distance_model is not None:
        print(f"距離推定モデルを読み込みました: {args.distance_checkpoint_path} "
              f"(sr={distance_sr}, clip_seconds={distance_clip_sec})")
    else:
        print("距離推定モデルは使用しません(未指定、またはファイルが見つかりません)")

    # 通知・記録用のロガーオブジェクトを作成
    notifier = DetectionLogger(LOG_CSV, SAVE_CLIP_ON_DETECT, CLIP_DIR, sample_rate,
                                distance_model=distance_model, distance_sample_rate=distance_sr,
                                distance_clip_seconds=distance_clip_sec, device=device)
    # 検知ロジックの管理オブジェクトを作成
    detector = RealtimeDetector(model, mel, classes, sample_rate, device, notifier)

    # モード分岐（ファイルシミュレーションかマイク実行か）
    if args.simulate_file:
        run_from_file(detector, sample_rate, args.simulate_file,
                       realtime_pace=not args.no_realtime_pace)
    else:
        run_from_microphone(detector, sample_rate)


if __name__ == '__main__':
    main()  # モジュールとしてではなく直接実行された場合に main() を呼び出す