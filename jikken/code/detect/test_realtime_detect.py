# ==============================================================================
# モジュール全体の概要説明[cite: 3]
# ==============================================================================
"""
realtime_detect.py の検知ロジック(RealtimeDetector)を、音声ファイル+正解ラベル(ground_truth.csv)
に対してオフラインで評価するためのテスト用デモスクリプト。

【本番コードとの関係(重要)】
    本番の realtime_detect.py 本体・main()・run_from_file()/run_from_microphone() には
    一切手を加えていない(RealtimeDetectorへのclock/enable_live_status引数追加のみ、
    どちらもデフォルト値により既存動作は不変)。このスクリプトは realtime_detect.py を
    "モジュールとしてimportして使う側" であり、本番の detections.db / live_status.json /
    realtime_detections.csv / detected_clips/ には一切書き込まない(EvalDetectionLoggerは
    DetectionLoggerを継承しない独立クラスで、そもそもそれらのファイルへの参照を持たない)。

【時間表記についての注意(重要)】
    このスクリプトが扱う・出力する時刻はすべて「音声ファイル先頭からの経過秒数」
    (例: 12.50 = 12.5秒地点)であり、壁時計時刻(例: 14:23:05)は一切使わない。
    目的が「検知がどのタイミングで発生しているか」を確認することなので、
    音声と正解ラベルに対して意味を持つのは経過秒数の方だからである。

    加えてこれは表示上の好みだけの話ではない。RealtimeDetector.step() のクールダウン判定
    ((now - self.last_detect_time) >= COOLDOWN_SECONDS) は、本番では self._clock() が
    time.time()(壁時計)を返すことを前提にしている。もしこの評価スクリプトが単純に
    time.time() をそのまま使うと、--no_realtime_pace 的に音声を「実時間を待たず」最速で
    流した場合、壁時計はほとんど進まないのに音声上の時間だけ進んでしまい、
    COOLDOWN_SECONDS の制限が「音声内で1秒以上離れている別々の正解イベント」まで
    誤って抑制してしまう(=FNの水増し)。これを避けるため、下記の SimClock を
    RealtimeDetector の clock 引数に渡し、「今どの経過秒数を処理中か」を明示的に
    差し込めるようにしている。

使い方:
    python test_realtime_detect.py --audio path/to/test.wav --ground_truth path/to/ground_truth.csv

    HOP_SECONDSを変えて比較したい場合(6.1節のパラメータスイープ相当):
    python test_realtime_detect.py --audio test.wav --ground_truth gt.csv --hop_seconds 0.1
    python test_realtime_detect.py --audio test.wav --ground_truth gt.csv --hop_seconds 0.2
    python test_realtime_detect.py --audio test.wav --ground_truth gt.csv --hop_seconds 0.5

出力(デフォルトは eval_runs/<実行日時>_<音声ファイル名>/ 以下にまとめる):
    - eval_detection_log.csv : 経過秒数ベースの検知ログ(TP/FP/Confused判定・遅延Δt付き)
    - eval_metrics.json      : クラスごとのPrecision/Recall/F1、Δt統計、TP/FP/FN件数
    - run_info.json          : test_frontend.py がタイムライン表示に使う一次データ一式

ground_truth.csv の形式:
    timestamp_sec, label
    12.50, shakapati_front
    30.10, Hindu_shuffle
    (labelは realtime_detect.py 側の実際のクラス名(suffixなし)を使うこと。
     学習データ由来の "_5seconds_cut" のようなsuffixが付いていた場合はこちらで自動的に除去する)
"""

import argparse
import csv
import json
import os
from contextlib import contextmanager
from datetime import datetime

import librosa
import numpy as np
import torch

import realtime_detect as rd  # 本番の検知ロジック(RealtimeDetector等)をそのまま再利用する[cite: 3]

TOLERANCE_SEC_DEFAULT = 0.5  # 正解イベントとのマッチングを許容する時間窓(±秒)[cite: 3]
GT_LABEL_SUFFIX_TO_STRIP = "_5seconds_cut"  # 学習データ由来のsuffix。ground_truth.csv側の事故防止用[cite: 3]


# ------------------------------------------------------------------------------
# 壁時計の代わりに「音声上の経過秒数」を保持するだけの入れ物[cite: 3]
# ------------------------------------------------------------------------------
class SimClock:
    """RealtimeDetectorのclock引数に渡すためのオブジェクト(callableにしてtime.timeの代役にする)。[cite: 3]
    シミュレーションループがHOPごとにset()で現在の経過秒数を更新し、
    RealtimeDetector.step()内の now = self._clock() はこの値を返すようになる。[cite: 3]"""

    def __init__(self):
        self._t = 0.0

    def set(self, elapsed_sec):
        self._t = elapsed_sec

    def __call__(self):
        return self._t


# ------------------------------------------------------------------------------
# 評価用の検知ロガー(本番のDetectionLoggerは継承しない・独立クラス)[cite: 3]
# ------------------------------------------------------------------------------
class EvalDetectionLogger:
    """RealtimeDetectorはこのオブジェクトのon_detect(predicted_class, confidence, audio_clip)を
    ダックタイピングで呼ぶだけなので、DetectionLoggerを継承する必要は無い。[cite: 3]
    継承すると__init__でのCSVヘッダー作成やclip_dir作成といった、本番向けの副作用まで
    引き継いでしまうため、あえて継承せずゼロから書いている。[cite: 3]

    本番との違い:
    - CSV/DB(detections.db)には一切書き込まない。結果はすべて self.records (メモリ上のリスト)に
      貯めるだけで、ファイルへの書き出しは呼び出し側がシミュレーション終了後にまとめて行う。[cite: 3]
    - 音声クリップ(.wav)はデフォルトでは保存しない(save_clips_dirがNoneなら何もしない)。
      同じ音声を何度も評価し直す(HOP_SECONDS比較実験など)ときに毎回大量の.wavが
      増え続けるのを防ぐため。距離推定自体はクリップ保存の有無に関わらず行う
      (推定にはメモリ上のaudio_clipがあれば十分で、ファイル保存は不要)。[cite: 3]
    """

    def __init__(self, sample_rate, distance_model=None, distance_sample_rate=None,
                 distance_clip_seconds=None, device="cpu", save_clips_dir=None):
        self.sample_rate = sample_rate
        self.distance_model = distance_model
        self.distance_sample_rate = distance_sample_rate
        self.distance_clip_seconds = distance_clip_seconds
        self.device = device
        self.save_clips_dir = save_clips_dir  # Noneなら.wav保存はしない(デフォルト)[cite: 3]
        if save_clips_dir:
            os.makedirs(save_clips_dir, exist_ok=True)

        # シミュレーションループがHOPごとに更新する。「今処理しているのは何秒地点か」[cite: 3]
        self.current_elapsed_sec = 0.0
        self.records = []  # 検知1件ごとに {elapsed_sec, predicted_class, confidence, ...} を積む[cite: 3]

    def on_detect(self, predicted_class, confidence, audio_clip=None):
        elapsed = self.current_elapsed_sec

        # 距離推定(本番のDetectionLogger.on_detectと同じ関数をそのまま再利用)[cite: 3]
        distance_cm = None
        if self.distance_model is not None and audio_clip is not None:
            distance_cm = rd.estimate_distance(
                self.distance_model, audio_clip, self.sample_rate,
                self.distance_sample_rate, self.distance_clip_seconds, self.device)

        clip_filename = ""
        if self.save_clips_dir and audio_clip is not None:
            import soundfile as sf  # 保存する場合だけ読み込む(依存を減らすため)[cite: 3]
            clip_filename = os.path.join(
                self.save_clips_dir, f"t{elapsed:08.2f}_{predicted_class}.wav")
            sf.write(clip_filename, audio_clip, self.sample_rate)

        self.records.append({
            "elapsed_sec": round(elapsed, 3),
            "predicted_class": predicted_class,
            "confidence": round(confidence, 3),
            "estimated_distance_cm": round(distance_cm, 1) if distance_cm is not None else None,
            "clip_file": clip_filename,
        })

        distance_msg = f", 推定距離={distance_cm:.1f}cm" if distance_cm is not None else ""
        print(f"  🔔 t={elapsed:7.2f}s 検知: {predicted_class} (確信度={confidence:.3f}{distance_msg})")


# ------------------------------------------------------------------------------
# realtime_detect.py のグローバル設定を一時的に上書きするためのヘルパー[cite: 3]
# ------------------------------------------------------------------------------
@contextmanager
def override_globals(**kwargs):
    """realtime_detect モジュールのグローバル変数(HOP_SECONDS等)を一時的に書き換える。[cite: 3]
    HOP_SECONDSスイープのようなパラメータ実験のために使う。
    with を抜けるとき(正常終了・例外どちらでも)必ず元の値に戻すので、
    このプロセス内で他の評価やimport元に影響を残さない。[cite: 3]"""
    original = {}
    for key, value in kwargs.items():
        original[key] = getattr(rd, key)
        setattr(rd, key, value)
    try:
        yield
    finally:
        for key, value in original.items():
            setattr(rd, key, value)


# ------------------------------------------------------------------------------
# ground_truth.csv の読み込み[cite: 3]
# ------------------------------------------------------------------------------
def load_ground_truth(path):
    """(timestamp_sec, label) のCSVを読み込み、経過秒数順に並べたリストを返す。[cite: 3]
    label は realtime_detect.py 側の実際のクラス名(suffixなし)を前提とするが、
    学習データ由来の "_5seconds_cut" のようなsuffixが付いていた場合は自動的に除去する
    (これを取り違えるとTPが一件も出ずRecallが常に0%になる事故につながるため、
    最低限の防御として入れてある)。[cite: 3]"""
    events = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            label = row["label"].strip()
            if label.endswith(GT_LABEL_SUFFIX_TO_STRIP):
                label = label[: -len(GT_LABEL_SUFFIX_TO_STRIP)]
            events.append({"elapsed_sec": float(row["timestamp_sec"]), "label": label})
    events.sort(key=lambda e: e["elapsed_sec"])
    return events


# ------------------------------------------------------------------------------
# 検知結果と正解ラベルの2段階マッチング[cite: 3]
# ------------------------------------------------------------------------------
def match_detections(detections, ground_truth, tolerance_sec):
    """まず時間窓(±tolerance_sec)だけを見て貪欲に対応付け、そのあとでクラス一致/不一致を見る
    2段階方式。時間差が小さい組み合わせから優先的に確定させることで、
    1つの正解イベントに複数の検知が競合した場合も一番近いものを優先する。[cite: 3]

    戻り値:
        results: detectionsと同じ順序・件数のリスト。各要素に
                 match_status("TP"/"Confused"/"FP"), matched_gt_elapsed_sec, delta_t_ms を追加したもの
        unmatched_gt_idx: どの検知からも対応付けられなかった正解イベントのインデックス一覧(=FN)[cite: 3]
    """
    candidate_pairs = []
    for di, d in enumerate(detections):
        for gi, g in enumerate(ground_truth):
            dt = d["elapsed_sec"] - g["elapsed_sec"]
            if abs(dt) <= tolerance_sec:
                candidate_pairs.append((abs(dt), di, gi))
    candidate_pairs.sort(key=lambda p: p[0])  # 時間差が小さい順に確定させる[cite: 3]

    matched_det_to_gt = {}
    matched_gt_to_det = {}
    for _, di, gi in candidate_pairs:
        if di in matched_det_to_gt or gi in matched_gt_to_det:
            continue  # どちらかが既に別の相手と確定済みならスキップ[cite: 3]
        matched_det_to_gt[di] = gi
        matched_gt_to_det[gi] = di

    results = []
    for di, d in enumerate(detections):
        r = dict(d)
        if di in matched_det_to_gt:
            gi = matched_det_to_gt[di]
            g = ground_truth[gi]
            r["matched_gt_idx"] = gi  # 内部計算用(CSVには書き出さない)[cite: 3]
            r["matched_gt_elapsed_sec"] = g["elapsed_sec"]
            r["delta_t_ms"] = round((d["elapsed_sec"] - g["elapsed_sec"]) * 1000, 1)
            r["match_status"] = "TP" if d["predicted_class"] == g["label"] else "Confused"
        else:
            r["matched_gt_idx"] = None
            r["matched_gt_elapsed_sec"] = None
            r["delta_t_ms"] = None
            r["match_status"] = "FP"
        results.append(r)

    unmatched_gt_idx = [gi for gi in range(len(ground_truth)) if gi not in matched_gt_to_det]
    return results, unmatched_gt_idx


# ------------------------------------------------------------------------------
# クラスごとのPrecision/Recall/F1、遅延Δtの集計[cite: 3]
# ------------------------------------------------------------------------------
def compute_metrics(results, ground_truth, unmatched_gt_idx, expected_classes):
    """混同行列の考え方でクラスごとのTP/FP/FNを積み上げる。[cite: 3]
    - TP: 時間窓内にマッチし、かつクラスも一致
    - Confused(時間は合うがクラス違い): 予測クラス側にFP、正解クラス側にFNを1件ずつ計上
    - 時間窓内に正解が無いFP: 予測クラスにFPを1件計上
    - 時間窓内に検知が無いFN(unmatched_gt_idx): 正解クラスにFNを1件計上[cite: 3]
    """
    tp = {c: 0 for c in expected_classes}
    fp = {c: 0 for c in expected_classes}
    fn = {c: 0 for c in expected_classes}

    for r in results:
        pc = r["predicted_class"]
        if r["match_status"] == "TP":
            tp[pc] = tp.get(pc, 0) + 1
        elif r["match_status"] == "Confused":
            gt_label = ground_truth[r["matched_gt_idx"]]["label"]
            fp[pc] = fp.get(pc, 0) + 1
            fn[gt_label] = fn.get(gt_label, 0) + 1
        else:  # "FP": 時間窓内に正解イベントが無かった検知[cite: 3]
            fp[pc] = fp.get(pc, 0) + 1

    for gi in unmatched_gt_idx:  # 時間窓内に検知が一件も無かった正解イベント[cite: 3]
        label = ground_truth[gi]["label"]
        fn[label] = fn.get(label, 0) + 1

    all_classes = sorted(set(expected_classes) | set(tp) | set(fp) | set(fn))
    per_class = {}
    for c in all_classes:
        t, f_p, f_n = tp.get(c, 0), fp.get(c, 0), fn.get(c, 0)
        precision = t / (t + f_p) if (t + f_p) > 0 else 0.0
        recall = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[c] = {"tp": t, "fp": f_p, "fn": f_n,
                         "precision": round(precision, 3), "recall": round(recall, 3),
                         "f1": round(f1, 3)}

    macro_precision = float(np.mean([v["precision"] for v in per_class.values()])) if per_class else 0.0
    macro_recall = float(np.mean([v["recall"] for v in per_class.values()])) if per_class else 0.0
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()])) if per_class else 0.0

    # 検知遅延Δt: 仕様通り「全TPの平均および最大遅延」(Confusedは含めない)[cite: 3]
    tp_delta = [r["delta_t_ms"] for r in results if r["match_status"] == "TP"]
    delay_stats = {
        "mean_ms": round(float(np.mean(tp_delta)), 1) if tp_delta else None,
        "max_ms": round(float(np.max(tp_delta)), 1) if tp_delta else None,
        "n_tp": len(tp_delta),
    }

    return {
        "per_class": per_class,
        "macro_precision": round(macro_precision, 3),
        "macro_recall": round(macro_recall, 3),
        "macro_f1": round(macro_f1, 3),
        "delay": delay_stats,
        "counts": {
            "tp": sum(1 for r in results if r["match_status"] == "TP"),
            "confused": sum(1 for r in results if r["match_status"] == "Confused"),
            "fp": sum(1 for r in results if r["match_status"] == "FP"),
            "fn": len(unmatched_gt_idx),
            "n_detections": len(results),
            "n_ground_truth": len(ground_truth),
        },
    }


# ------------------------------------------------------------------------------
# 音声ファイル1本に対する検知シミュレーション本体[cite: 3]
# ------------------------------------------------------------------------------
def run_simulation(audio_path, checkpoint_path, distance_checkpoint_path, hop_seconds=None,
                    save_clips_dir=None, device=None):
    """本番のRealtimeDetector/DetectionLoggerと同じ検知ロジック(step()の中身)をそのまま使い、
    以下の3点だけを評価用に差し替える。[cite: 3]
    - notifier: EvalDetectionLogger(本番のCSV/DB/クリップ保存を一切行わない)
    - clock: SimClock(壁時計の代わりに「音声上の経過秒数」を返す)
    - enable_live_status=False: detections.db / live_status.json を一切書き換えない

    戻り値: (検知結果のリスト(EvalDetectionLogger.records), RMS推移(タイムライン背景描画用))[cite: 3]
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    model, mel, classes, sample_rate = rd.load_model(checkpoint_path, device)
    distance_model, distance_sr, distance_clip_sec = rd.load_distance_model(
        distance_checkpoint_path, device)

    notifier = EvalDetectionLogger(sample_rate, distance_model=distance_model,
                                    distance_sample_rate=distance_sr,
                                    distance_clip_seconds=distance_clip_sec,
                                    device=device, save_clips_dir=save_clips_dir)
    clock = SimClock()
    detector = rd.RealtimeDetector(model, mel, classes, sample_rate, device, notifier,
                                    clock=clock, enable_live_status=False)

    y, sr = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    y = y.astype(np.float32)
    block_size = int(rd.BLOCK_SECONDS * sample_rate)
    n_blocks = len(y) // block_size
    hop_seconds = hop_seconds if hop_seconds is not None else rd.HOP_SECONDS
    hop_blocks = max(1, int(hop_seconds / rd.BLOCK_SECONDS))

    rms_trace = []  # [(elapsed_sec, rms), ...] タイムラインの背景描画に使う(壁時計文字列は使わない)[cite: 3]
    print(f"評価シミュレーション開始: {audio_path} "
          f"({len(y) / sample_rate:.1f}秒, HOP_SECONDS={hop_seconds}s, "
          f"クラス={classes}, 検知対象={rd.TARGET_CLASSES})")

    for i in range(n_blocks):
        chunk = y[i * block_size:(i + 1) * block_size]
        detector.push_audio(chunk)

        if i % hop_blocks == 0:
            elapsed = i * rd.BLOCK_SECONDS
            clock.set(elapsed)  # RealtimeDetector.step()内の now=self._clock() がこの値を返す[cite: 3]
            notifier.current_elapsed_sec = elapsed  # on_detect()が記録に使う経過秒数を更新[cite: 3]
            detector.step()
            window = detector.buffer[-detector.window_len:]
            rms_val = float(np.sqrt(np.mean(window.astype(np.float64) ** 2)))
            rms_trace.append((round(elapsed, 3), round(rms_val, 5)))

    print(f"評価シミュレーション終了: 検知件数={len(notifier.records)}")
    return notifier.records, rms_trace


# ------------------------------------------------------------------------------
# 出力ファイルの書き出し[cite: 3]
# ------------------------------------------------------------------------------
def save_outputs(output_dir, audio_path, ground_truth_path, ground_truth,
                  match_results, metrics, rms_trace, hop_seconds, tolerance_sec):
    os.makedirs(output_dir, exist_ok=True)

    # eval_detection_log.csv (経過秒数ベース)[cite: 3]
    log_path = os.path.join(output_dir, "eval_detection_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed_sec", "predicted_class", "confidence", "estimated_distance_cm",
                          "match_status", "matched_gt_elapsed_sec", "delta_t_ms"])
        for r in match_results:
            writer.writerow([
                r["elapsed_sec"], r["predicted_class"], r["confidence"],
                r["estimated_distance_cm"] if r["estimated_distance_cm"] is not None else "",
                r["match_status"],
                r["matched_gt_elapsed_sec"] if r["matched_gt_elapsed_sec"] is not None else "",
                r["delta_t_ms"] if r["delta_t_ms"] is not None else "",
            ])

    with open(os.path.join(output_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # run_info.json: test_frontend.py がタイムライン表示に使う一次データ一式[cite: 3]
    run_info = {
        "audio_path": audio_path,
        "ground_truth_path": ground_truth_path,
        "hop_seconds": hop_seconds,
        "tolerance_sec": tolerance_sec,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ground_truth": ground_truth,
        "detections": [
            {k: v for k, v in r.items() if k != "matched_gt_idx"}  # 内部用インデックスは出力しない[cite: 3]
            for r in match_results
        ],
        "rms_trace": rms_trace,
        "metrics": metrics,
    }
    with open(os.path.join(output_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False)

    print(f"\n出力先: {output_dir}")
    print(f"  - {os.path.basename(log_path)}")
    print(f"  - eval_metrics.json")
    print(f"  - run_info.json  (test_frontend.py --run_dir \"{output_dir}\" で開けます)")


def print_summary(metrics):
    print("\n=== 評価サマリ (すべて経過秒数ベース) ===")
    print(f"Macro Precision={metrics['macro_precision']:.3f}  "
          f"Recall={metrics['macro_recall']:.3f}  F1={metrics['macro_f1']:.3f}")
    c = metrics["counts"]
    print(f"TP={c['tp']}  Confused={c['confused']}  FP={c['fp']}  FN={c['fn']}  "
          f"(正解イベント数={c['n_ground_truth']}, 検知件数={c['n_detections']})")
    d = metrics["delay"]
    if d["n_tp"] > 0:
        print(f"検知遅延Δt(TPのみ, {d['n_tp']}件): 平均={d['mean_ms']}ms  最大={d['max_ms']}ms")
    else:
        print("検知遅延Δt: TPが0件のため算出不可")
    print("\nクラスごと:")
    for c_name, v in metrics["per_class"].items():
        print(f"  {c_name:25s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} "
              f"(TP={v['tp']} FP={v['fp']} FN={v['fn']})")


# ------------------------------------------------------------------------------
# スクリプト実行時のエントリーポイント[cite: 3]
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="realtime_detect.py の検知ロジックをオフラインで評価するテスト用デモ"
                     "(本番のdetections.db/live_status.json/CSV/クリップには一切書き込まない)")
    parser.add_argument("--audio", required=True, help="評価対象の音声ファイル(wav)")
    parser.add_argument("--ground_truth", required=True,
                         help="正解ラベルCSV(timestamp_sec, label)")
    parser.add_argument("--checkpoint_path", type=str, default=rd.CHECKPOINT_PATH)
    parser.add_argument("--distance_checkpoint_path", type=str, default=rd.DISTANCE_CHECKPOINT_PATH)
    parser.add_argument("--hop_seconds", type=float, default=None,
                         help="HOP_SECONDSを一時的に上書きして評価したい場合に指定"
                              "(例: 0.1, 0.2, 0.5 で比較する)。省略時はrealtime_detect.pyの設定値を使う")
    parser.add_argument("--tolerance_sec", type=float, default=TOLERANCE_SEC_DEFAULT,
                         help=f"正解とのマッチングを許容する時間窓(±秒)。デフォルト{TOLERANCE_SEC_DEFAULT}")
    parser.add_argument("--save_clips", action="store_true",
                         help="検知時の音声クリップを output_dir/clips/ にwavとして保存する"
                              "(デフォルトでは保存しない)")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="出力先フォルダ(省略時は eval_runs/<日時>_<音声ファイル名>/ )")
    args = parser.parse_args()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        base = os.path.splitext(os.path.basename(args.audio))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("eval_runs", f"{ts}_{base}")

    save_clips_dir = os.path.join(output_dir, "clips") if args.save_clips else None

    ground_truth = load_ground_truth(args.ground_truth)
    expected_classes = sorted(set(rd.TARGET_CLASSES) | {g["label"] for g in ground_truth})

    # SAVE_CLIP_ON_DETECTは「RealtimeDetector.step()がaudio_clipをon_detect()へ渡すかどうか」の
    # ゲートも兼ねているため、評価中は常にTrueにしておく(距離推定にはこのクリップが必要)。
    # 実際に.wavファイルへ書き出すかどうかは--save_clips(EvalDetectionLogger側)で別途制御しており、
    # こちらは常にFalseのままで良い。with を抜ければ元の値に自動で戻る。[cite: 3]
    with override_globals(SAVE_CLIP_ON_DETECT=True):
        detections, rms_trace = run_simulation(
            args.audio, args.checkpoint_path, args.distance_checkpoint_path,
            hop_seconds=args.hop_seconds, save_clips_dir=save_clips_dir)

    match_results, unmatched_gt_idx = match_detections(detections, ground_truth, args.tolerance_sec)
    metrics = compute_metrics(match_results, ground_truth, unmatched_gt_idx, expected_classes)

    hop_seconds_used = args.hop_seconds if args.hop_seconds is not None else rd.HOP_SECONDS
    save_outputs(output_dir, args.audio, args.ground_truth, ground_truth,
                 match_results, metrics, rms_trace, hop_seconds_used, args.tolerance_sec)
    print_summary(metrics)


if __name__ == "__main__":
    main()
