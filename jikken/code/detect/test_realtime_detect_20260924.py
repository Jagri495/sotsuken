# ==============================================================================
# モジュール全体の概要説明
# ==============================================================================
"""
realtime_detect.py の検知ロジック(RealtimeDetector)を、音声ファイル+正解ラベル(ground_truth.csv)
に対してオフラインで評価するためのテスト用デモスクリプト。

【9/24版での変更点(test_realtime_detect.py からの差分)】
    1. ground_truth.csv が「区間形式」(start_sec, end_sec, label)に対応した。「1回のシャッフルを
       どう定義するか」を数えなくてよいように、「この区間はずっとこの行動をしていた」という粒度で
       正解を書けるようにしている(点形式(timestamp_sec, label)も後方互換で引き続き使える)。
    2. 発火(fired)したかどうかに関わらず、毎HOPごとの生の予測(predicted_class, confidence)を
       全部 raw_predictions として記録するようにした。「0〜5秒はshakapati_front、5秒あたりから
       Deal_shuffleに変わる」のような、境界付近での予測の推移を後から確認できる。
       【重要】これは診断用の追加ログであり、realtime_detect.py本体の検知ロジック(エッジトリガー・
       MIN_CONSECUTIVE・COOLDOWN_SECONDS)は一切変更していない。実際の検知イベント(発火)の
       挙動は今まで通り。
    3. --window_seconds を追加。1秒/3秒/5秒など、異なる窓幅で学習したモデルを比較評価できる
       (--checkpoint_path と必ずセットで、学習時の窓幅と一致させて使うこと)。

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

    加えてこれは表示上の好みだけの話ではない。RealtimeDetector.step() のクールダウン判定
    ((now - self.last_detect_time) >= COOLDOWN_SECONDS)は、本番では self._clock() が
    time.time()(壁時計)を返すことを前提にしている。もしこの評価スクリプトが単純に
    time.time() をそのまま使うと、音声を「実時間を待たず」最速で流した場合、壁時計はほとんど
    進まないのに音声上の時間だけ進んでしまい、COOLDOWN_SECONDS の制限が誤って作動する
    (=FNの水増し)。これを避けるため、下記の SimClock を RealtimeDetector の clock 引数に渡し、
    「今どの経過秒数を処理中か」を明示的に差し込めるようにしている。

    【注意】このバージョンでは音声を実時間を待たず最速で処理する(--realtime_paceのような
    実時間再生オプションはまだ未実装)。「ラベルの誤差」と「実機の処理落ち」を切り分けたい場合は、
    このシミュレーションでの結果(原理的な遅延)と、本番実行時のコンソールログ(推論/step/DB書込み
    時間)を別々に見ること。両方を1回の実行で同時に測る仕組みは今後の課題。

使い方:
    python test_realtime_detect_20260924.py --audio test.wav --ground_truth ground_truth.csv

    HOP_SECONDSを変えて比較したい場合:
    python test_realtime_detect_20260924.py --audio test.wav --ground_truth gt.csv --hop_seconds 0.1

    窓幅の異なるモデルを比較したい場合(チェックポイントと窓幅は必ずセットで指定すること):
    python test_realtime_detect_20260924.py --audio test.wav --ground_truth gt.csv \\
        --checkpoint_path model_3s.pt --window_seconds 3.0

出力(デフォルトは eval_runs/<実行日時>_<音声ファイル名>/ 以下にまとめる):
    - eval_detection_log.csv     : 経過秒数ベースの検知ログ(TP/FP/Confused判定・遅延Δt付き)
    - eval_raw_predictions.csv   : 【新規】毎HOPごとの生の予測(発火有無に関わらず全件)
    - eval_metrics.json          : クラスごとのPrecision/Recall/F1、Δt統計、区間ごとの被覆状況
    - run_info.json              : test_frontend.py がタイムライン表示に使う一次データ一式

ground_truth.csv の形式(どちらも使える):
    【区間形式(推奨)】
        start_sec, end_sec, label
        0.0, 7.0, shakapati_front
        7.0, 12.0, none                 ← 「none」「background」等は「何も起きていない区間」として
                                            扱われ、そこに落ちた検知は素直にFPになる(Confusedにはしない)
        12.0, 19.0, Deal_shuffle
    【点形式(後方互換)】
        timestamp_sec, label
        12.50, shakapati_front
    (labelは realtime_detect.py 側の実際のクラス名(suffixなし)を使うこと。
     学習データ由来の "_5seconds_cut" のようなsuffixが付いていた場合はこちらで自動的に除去する)

    【注意】ground_truth.csvで覆っていない時間帯(区間として書かれていない隙間)に検知が起きた場合、
    そこは「未知」ではなく素直にFPとして扱われる。音声の全区間を「実際の行動」か「none」のどちらかで
    切れ目なく書いておくことを推奨する(以前話した、既知のクリップを結合してテスト音声を作る方法なら、
    結合の境界がそのまま区間の境目になるので、この網羅がほぼ自動的にできる)。
"""

# ==============================================================================
# ライブラリのインポート
# ==============================================================================
import argparse
import csv
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime

import librosa
import numpy as np
import torch

import realtime_detect as rd  # 本番の検知ロジック(RealtimeDetector等)をそのまま再利用する

TOLERANCE_SEC_DEFAULT = 0.5  # 正解区間の前後、マッチングを許容する時間(±秒)
GT_LABEL_SUFFIX_PATTERN = re.compile(r"_\d+seconds?_cut$", re.IGNORECASE)  # 学習データ由来のsuffix。
# 【9/25更新】1秒/3秒/5秒窓モデルの比較評価に備え、"_5seconds_cut"固定文字列ではなく正規表現化
# (realtime_detect.py側のstrip_clip_suffix()と同じロジック)。
NONE_LABELS = {"none", "background", "nothing", "no_action", ""}  # 「何も起きていない」区間の予約語


# ------------------------------------------------------------------------------
# 壁時計の代わりに「音声上の経過秒数」を保持するだけの入れ物
# ------------------------------------------------------------------------------
class SimClock:
    """RealtimeDetectorのclock引数に渡すためのオブジェクト(callableにしてtime.timeの代役にする)。
    シミュレーションループがHOPごとにset()で現在の経過秒数を更新し、
    RealtimeDetector.step()内の now = self._clock() はこの値を返すようになる。"""

    def __init__(self):
        self._t = 0.0

    def set(self, elapsed_sec):
        self._t = elapsed_sec

    def __call__(self):
        return self._t


# ------------------------------------------------------------------------------
# 評価用の検知ロガー(本番のDetectionLoggerは継承しない・独立クラス)
# ------------------------------------------------------------------------------
class EvalDetectionLogger:
    """RealtimeDetectorはこのオブジェクトのon_detect(predicted_class, confidence, audio_clip)を
    ダックタイピングで呼ぶだけなので、DetectionLoggerを継承する必要は無い。
    継承すると__init__でのCSVヘッダー作成やclip_dir作成といった、本番向けの副作用まで
    引き継いでしまうため、あえて継承せずゼロから書いている。

    本番との違い:
    - CSV/DB(detections.db)には一切書き込まない。結果はすべて self.records (メモリ上のリスト)に
      貯めるだけで、ファイルへの書き出しは呼び出し側がシミュレーション終了後にまとめて行う。
    - 音声クリップ(.wav)はデフォルトでは保存しない(save_clips_dirがNoneなら何もしない)。
      距離推定自体はクリップ保存の有無に関わらず行う(推定にはメモリ上のaudio_clipがあれば十分)。
    """

    def __init__(self, sample_rate, distance_model=None, distance_sample_rate=None,
                 distance_clip_seconds=None, device="cpu", save_clips_dir=None):
        self.sample_rate = sample_rate
        self.distance_model = distance_model
        self.distance_sample_rate = distance_sample_rate
        self.distance_clip_seconds = distance_clip_seconds
        self.device = device
        self.save_clips_dir = save_clips_dir
        if save_clips_dir:
            os.makedirs(save_clips_dir, exist_ok=True)

        self.current_elapsed_sec = 0.0  # シミュレーションループがHOPごとに更新する
        self.records = []  # 検知1件ごとに {elapsed_sec, predicted_class, confidence, ...} を積む

    def on_detect(self, predicted_class, confidence, audio_clip=None):
        elapsed = self.current_elapsed_sec

        distance_cm = None
        if self.distance_model is not None and audio_clip is not None:
            distance_cm = rd.estimate_distance(
                self.distance_model, audio_clip, self.sample_rate,
                self.distance_sample_rate, self.distance_clip_seconds, self.device)

        clip_filename = ""
        if self.save_clips_dir and audio_clip is not None:
            import soundfile as sf  # 保存する場合だけ読み込む(依存を減らすため)
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
# realtime_detect.py のグローバル設定を一時的に上書きするためのヘルパー
# ------------------------------------------------------------------------------
@contextmanager
def override_globals(**kwargs):
    """realtime_detect モジュールのグローバル変数(HOP_SECONDS, WINDOW_SECONDS等)を一時的に
    書き換える。パラメータ実験(HOP_SECONDSスイープ、窓幅比較)のために使う。
    with を抜けるとき(正常終了・例外どちらでも)必ず元の値に戻すので、
    このプロセス内で他の評価やimport元に影響を残さない。"""
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
# ground_truth.csv の読み込み(区間形式・点形式の両対応)
# ------------------------------------------------------------------------------
def load_ground_truth(path):
    """区間形式(start_sec, end_sec, label)と点形式(timestamp_sec, label)の両方に対応する。
    点形式は内部的に start_sec == end_sec == timestamp_sec の「幅0の区間」として扱われるため、
    以降のマッチングロジックはどちらの形式でも共通で動く。

    label は realtime_detect.py 側の実際のクラス名(suffixなし)を前提とするが、
    学習データ由来の "_5seconds_cut" のようなsuffixが付いていた場合は自動的に除去する。
    label が NONE_LABELS(none/background等)に該当する行は「何も起きていない区間」として
    is_background=True でマークし、マッチング対象(TP/Confusedの相手)からは除外する
    (そこに落ちた検知は素直にFPとして扱われる)。"""
    events = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        is_interval_format = {"start_sec", "end_sec"}.issubset(fieldnames)
        for row in reader:
            label = row["label"].strip()
            label = GT_LABEL_SUFFIX_PATTERN.sub("", label)
            if is_interval_format:
                start_sec = float(row["start_sec"])
                end_sec = float(row["end_sec"])
            else:
                start_sec = end_sec = float(row["timestamp_sec"])
            events.append({
                "start_sec": start_sec,
                "end_sec": end_sec,
                "label": label,
                "is_background": label.lower() in NONE_LABELS,
            })
    events.sort(key=lambda e: e["start_sec"])
    return events


# ------------------------------------------------------------------------------
# 検知結果と正解区間のマッチング
# ------------------------------------------------------------------------------
def match_detections(detections, ground_truth, tolerance_sec):
    """検知結果と正解区間を対応付ける。

    マッチング方針:
      1. まず「許容誤差なしの厳密な区間内」に検知があるかを見る
      2. 厳密に入っていなければ、区間の前後 tolerance_sec 以内かを見る
         (境界のズレ・ウィンドウ由来の遅延を許容するため)
      3. 候補が複数ある場合、predicted_classと一致するものを優先。一致するものが無ければ、
         区間までの距離が一番近いものを選ぶ
      4. is_background(none等)の区間は候補に含めない(そこに落ちた検知は素直にFPになる)
      5. 1つの区間には最大1件の検知しか「代表として」対応付けない(同じ区間内に複数の検知が
         あっても、2件目以降は別の未使用の区間を探しにいく)。これにより、1つの長い区間に
         検知が複数回起きても「区間を見つけられたかどうか」の判定(interval_coverage、
         compute_metrics参照)が壊れない。

    戻り値:
        results: detectionsと同じ順序・件数のリスト。各要素に
                 match_status("TP"/"Confused"/"FP"), matched_gt_idx, matched_gt_start_sec,
                 matched_gt_end_sec, matched_gt_elapsed_sec(=区間の開始秒、CSV互換用),
                 delta_t_ms(=検知時刻-区間の開始秒。区間内のどのタイミングで検知したかを表す) を追加
        unmatched_gt_idx: どの検知からも対応付けられなかった正解区間(背景区間を除く)のインデックス
    """
    matchable = [(gi, g) for gi, g in enumerate(ground_truth) if not g["is_background"]]

    def distance_to_interval(t, g):
        if g["start_sec"] <= t <= g["end_sec"]:
            return 0.0
        return min(abs(t - g["start_sec"]), abs(t - g["end_sec"]))

    matched_det_to_gt = {}
    matched_gt_to_det = {}

    # 検知は時刻順に処理する(先着優先。どちらから処理しても大勢に影響は無いが、結果を安定させるため)
    for di, d in sorted(enumerate(detections), key=lambda x: x[1]["elapsed_sec"]):
        t = d["elapsed_sec"]
        strict = [(gi, g) for gi, g in matchable
                  if g["start_sec"] <= t <= g["end_sec"] and gi not in matched_gt_to_det]
        candidates = strict if strict else [
            (gi, g) for gi, g in matchable
            if g["start_sec"] - tolerance_sec <= t <= g["end_sec"] + tolerance_sec
            and gi not in matched_gt_to_det
        ]
        if not candidates:
            continue
        exact_class = [(gi, g) for gi, g in candidates if g["label"] == d["predicted_class"]]
        pool = exact_class if exact_class else candidates
        gi, g = min(pool, key=lambda pair: distance_to_interval(t, pair[1]))
        matched_det_to_gt[di] = gi
        matched_gt_to_det[gi] = di

    results = []
    for di, d in enumerate(detections):
        r = dict(d)
        if di in matched_det_to_gt:
            gi = matched_det_to_gt[di]
            g = ground_truth[gi]
            r["matched_gt_idx"] = gi
            r["matched_gt_start_sec"] = g["start_sec"]
            r["matched_gt_end_sec"] = g["end_sec"]
            r["matched_gt_elapsed_sec"] = g["start_sec"]  # CSV互換用。区間の開始秒を代表値にする
            r["delta_t_ms"] = round((d["elapsed_sec"] - g["start_sec"]) * 1000, 1)
            r["match_status"] = "TP" if d["predicted_class"] == g["label"] else "Confused"
        else:
            r["matched_gt_idx"] = None
            r["matched_gt_start_sec"] = None
            r["matched_gt_end_sec"] = None
            r["matched_gt_elapsed_sec"] = None
            r["delta_t_ms"] = None
            r["match_status"] = "FP"
        results.append(r)

    unmatched_gt_idx = [gi for gi, _ in matchable if gi not in matched_gt_to_det]
    return results, unmatched_gt_idx


# ------------------------------------------------------------------------------
# クラスごとのPrecision/Recall/F1、遅延Δt、区間ごとの被覆状況の集計
# ------------------------------------------------------------------------------
def compute_metrics(results, ground_truth, unmatched_gt_idx, expected_classes):
    """混同行列の考え方でクラスごとのTP/FP/FNを積み上げる(detection単位、従来通り)。
    - TP: 区間内(または許容誤差内)にマッチし、かつクラスも一致
    - Confused(区間には合うがクラス違い): 予測クラス側にFP、正解クラス側にFNを1件ずつ計上
    - 区間外のFP: 予測クラスにFPを1件計上
    - どの検知からも見つけられなかった区間(unmatched_gt_idx): 正解クラスにFNを1件計上

    加えて interval_coverage(区間ごとの被覆状況、9/24新規)を返す。こちらはdetection単位ではなく
    「区間(=1つの行動のまとまり)を見つけられたか」という粒度の集計で、境界での挙動を見るための
    診断情報。P/R/F1の算出には使わない(detection単位の集計とは独立)。
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
        else:  # "FP": 区間外(または背景区間)に落ちた検知
            fp[pc] = fp.get(pc, 0) + 1

    for gi in unmatched_gt_idx:
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

    # 検知遅延Δt: 区間ごとに「最初のTP」だけを使う(1つの長い区間で複数回発火した場合、
    # 2回目以降は「区間の開始からの経過時間」が単調に増えるだけで、遅延の実態を表さないため、
    # 平均・最大の算出からは除外する。TPの件数自体(counts.tp)には引き続きすべて計上される)。
    first_tp_per_interval = {}
    for r in results:
        if r["match_status"] != "TP":
            continue
        gi = r["matched_gt_idx"]
        if gi not in first_tp_per_interval or r["elapsed_sec"] < first_tp_per_interval[gi]["elapsed_sec"]:
            first_tp_per_interval[gi] = r
    tp_delta = [r["delta_t_ms"] for r in first_tp_per_interval.values()]
    delay_stats = {
        "mean_ms": round(float(np.mean(tp_delta)), 1) if tp_delta else None,
        "max_ms": round(float(np.max(tp_delta)), 1) if tp_delta else None,
        "n_tp": len(tp_delta),  # 「最初のTPのみ」を数えた件数。全TP件数は counts.tp を参照
    }

    # 区間ごとの被覆状況(9/24新規、診断用)。
    # 【重要】ここだけは、上のTP/FP/FN集計(1区間につき1検知しか代表登録しない matched_gt_idx)とは
    # 独立に、「時間的にその区間に入っている検知」を全部拾い直している。理由: 例えば0〜7秒の区間で
    # 2秒地点の検知が正しくTPとして代表登録された後、6.5秒地点で別クラスの誤検知が起きた場合、
    # matched_gt_idxベースの集計だと(区間は既に2秒の検知で「見つかった」ことになっているため)6.5秒の
    # 誤検知はどの区間にも属さない単純なFPとして処理される。それ自体はP/R/F1の計算としては正しいが、
    # 「0〜7秒の区間の**中で**何が起きていたか」を見たい境界診断の目的には合わないため、
    # ここでは時刻の範囲だけで独立に再集計している(ユーザー報告の「5秒あたりで別クラスに変わる」
    # 現象を見逃さないようにするため)。
    interval_coverage = []
    for gi, g in enumerate(ground_truth):
        if g["is_background"]:
            continue
        in_interval = [r for r in results if g["start_sec"] <= r["elapsed_sec"] <= g["end_sec"]]
        tp_count = sum(1 for r in in_interval if r["predicted_class"] == g["label"])
        others = [r for r in in_interval if r["predicted_class"] != g["label"]]
        interval_coverage.append({
            "start_sec": g["start_sec"],
            "end_sec": g["end_sec"],
            "label": g["label"],
            "found": tp_count > 0,
            "tp_count": tp_count,
            "confused_count": len(others),
            "confused_with": sorted(set(r["predicted_class"] for r in others)),
        })

    return {
        "per_class": per_class,
        "macro_precision": round(macro_precision, 3),
        "macro_recall": round(macro_recall, 3),
        "macro_f1": round(macro_f1, 3),
        "delay": delay_stats,
        "interval_coverage": interval_coverage,
        "counts": {
            "tp": sum(1 for r in results if r["match_status"] == "TP"),
            "confused": sum(1 for r in results if r["match_status"] == "Confused"),
            "fp": sum(1 for r in results if r["match_status"] == "FP"),
            "fn": len(unmatched_gt_idx),
            "n_detections": len(results),
            "n_ground_truth": sum(1 for g in ground_truth if not g["is_background"]),
        },
    }


# ------------------------------------------------------------------------------
# 音声ファイル1本に対する検知シミュレーション本体
# ------------------------------------------------------------------------------
def run_simulation(audio_path, checkpoint_path, distance_checkpoint_path, hop_seconds=None,
                    save_clips_dir=None, device=None):
    """本番のRealtimeDetector/DetectionLoggerと同じ検知ロジック(step()の中身)をそのまま使い、
    以下の3点だけを評価用に差し替える。
    - notifier: EvalDetectionLogger(本番のCSV/DB/クリップ保存を一切行わない)
    - clock: SimClock(壁時計の代わりに「音声上の経過秒数」を返す)
    - enable_live_status=False: detections.db / live_status.json を一切書き換えない

    戻り値: (検知結果のリスト, RMS推移, 毎HOPごとの生の予測リスト(raw_predictions))
    raw_predictions は発火の有無に関わらず全HOP分を記録する診断用ログ(9/24新規)。
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

    rms_trace = []       # [(elapsed_sec, rms), ...] タイムラインの背景描画に使う
    raw_predictions = []  # [(elapsed_sec, predicted_class, confidence, fired), ...] 9/24新規

    print(f"評価シミュレーション開始: {audio_path} "
          f"({len(y) / sample_rate:.1f}秒, WINDOW_SECONDS={rd.WINDOW_SECONDS}s, "
          f"HOP_SECONDS={hop_seconds}s, クラス={classes}, 検知対象={rd.TARGET_CLASSES})")

    for i in range(n_blocks):
        chunk = y[i * block_size:(i + 1) * block_size]
        detector.push_audio(chunk)

        if i % hop_blocks == 0:
            elapsed = i * rd.BLOCK_SECONDS
            clock.set(elapsed)
            notifier.current_elapsed_sec = elapsed
            pred_class, conf, fired = detector.step()

            # 発火の有無に関わらず、この瞬間の生の予測を残しておく(境界分析用)
            raw_predictions.append({
                "elapsed_sec": round(elapsed, 3),
                "predicted_class": pred_class,
                "confidence": round(conf, 3) if pred_class else None,
                "fired": fired,
            })

            window = detector.buffer[-detector.window_len:]
            rms_val = float(np.sqrt(np.mean(window.astype(np.float64) ** 2)))
            rms_trace.append((round(elapsed, 3), round(rms_val, 5)))

    print(f"評価シミュレーション終了: 検知件数={len(notifier.records)}  "
          f"生の予測ログ件数={len(raw_predictions)}")
    return notifier.records, rms_trace, raw_predictions


# ------------------------------------------------------------------------------
# 出力ファイルの書き出し
# ------------------------------------------------------------------------------
def save_outputs(output_dir, audio_path, ground_truth_path, ground_truth,
                  match_results, metrics, rms_trace, raw_predictions,
                  hop_seconds, window_seconds, tolerance_sec):
    os.makedirs(output_dir, exist_ok=True)

    # eval_detection_log.csv (経過秒数ベース)
    log_path = os.path.join(output_dir, "eval_detection_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed_sec", "predicted_class", "confidence", "estimated_distance_cm",
                          "match_status", "matched_gt_start_sec", "matched_gt_end_sec", "delta_t_ms"])
        for r in match_results:
            writer.writerow([
                r["elapsed_sec"], r["predicted_class"], r["confidence"],
                r["estimated_distance_cm"] if r["estimated_distance_cm"] is not None else "",
                r["match_status"],
                r["matched_gt_start_sec"] if r["matched_gt_start_sec"] is not None else "",
                r["matched_gt_end_sec"] if r["matched_gt_end_sec"] is not None else "",
                r["delta_t_ms"] if r["delta_t_ms"] is not None else "",
            ])

    # eval_raw_predictions.csv (9/24新規: 発火の有無に関わらず毎HOPの生の予測)
    raw_path = os.path.join(output_dir, "eval_raw_predictions.csv")
    with open(raw_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed_sec", "predicted_class", "confidence", "fired"])
        for p in raw_predictions:
            writer.writerow([p["elapsed_sec"], p["predicted_class"] or "", p["confidence"], p["fired"]])

    with open(os.path.join(output_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # run_info.json: test_frontend.py がタイムライン表示に使う一次データ一式
    run_info = {
        "audio_path": audio_path,
        "ground_truth_path": ground_truth_path,
        "hop_seconds": hop_seconds,
        "window_seconds": window_seconds,
        "tolerance_sec": tolerance_sec,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ground_truth": ground_truth,
        "detections": [
            {k: v for k, v in r.items() if k != "matched_gt_idx"}  # 内部用インデックスは出力しない
            for r in match_results
        ],
        "rms_trace": rms_trace,
        "raw_predictions": raw_predictions,
        "metrics": metrics,
    }
    with open(os.path.join(output_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False)

    print(f"\n出力先: {output_dir}")
    print(f"  - {os.path.basename(log_path)}")
    print(f"  - {os.path.basename(raw_path)}")
    print(f"  - eval_metrics.json")
    print(f"  - run_info.json  (test_frontend_20260924.py --run_dir \"{output_dir}\" で開けます)")


def print_summary(metrics):
    print("\n=== 評価サマリ (すべて経過秒数ベース) ===")
    print(f"Macro Precision={metrics['macro_precision']:.3f}  "
          f"Recall={metrics['macro_recall']:.3f}  F1={metrics['macro_f1']:.3f}")
    c = metrics["counts"]
    print(f"TP={c['tp']}  Confused={c['confused']}  FP={c['fp']}  FN={c['fn']}  "
          f"(正解区間数={c['n_ground_truth']}, 検知件数={c['n_detections']})")
    d = metrics["delay"]
    if d["n_tp"] > 0:
        print(f"検知遅延Δt(区間ごとの最初のTPのみ, {d['n_tp']}件): 平均={d['mean_ms']}ms  最大={d['max_ms']}ms")
    else:
        print("検知遅延Δt: TPが0件のため算出不可")

    print("\nクラスごと:")
    for c_name, v in metrics["per_class"].items():
        print(f"  {c_name:25s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} "
              f"(TP={v['tp']} FP={v['fp']} FN={v['fn']})")

    print("\n区間ごとの被覆状況(境界での挙動を見るための診断情報):")
    for iv in metrics["interval_coverage"]:
        mark = "✅" if iv["found"] else "❌"
        confused_msg = f"  ※誤って {'/'.join(iv['confused_with'])} とも判定" if iv["confused_with"] else ""
        print(f"  {mark} {iv['start_sec']:7.2f}〜{iv['end_sec']:7.2f}s  {iv['label']:20s} "
              f"(TP={iv['tp_count']}, Confused={iv['confused_count']}){confused_msg}")


# ------------------------------------------------------------------------------
# スクリプト実行時のエントリーポイント
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="realtime_detect.py の検知ロジックをオフラインで評価するテスト用デモ"
                     "(本番のdetections.db/live_status.json/CSV/クリップには一切書き込まない)")
    parser.add_argument("--audio", required=True, help="評価対象の音声ファイル(wav)")
    parser.add_argument("--ground_truth", required=True,
                         help="正解ラベルCSV(区間形式: start_sec,end_sec,label / 点形式: timestamp_sec,label)")
    parser.add_argument("--checkpoint_path", type=str, default=rd.CHECKPOINT_PATH)
    parser.add_argument("--distance_checkpoint_path", type=str, default=rd.DISTANCE_CHECKPOINT_PATH)
    parser.add_argument("--hop_seconds", type=float, default=None,
                         help="HOP_SECONDSを一時的に上書きして評価したい場合に指定"
                              "(例: 0.1, 0.2, 0.5 で比較する)。省略時はrealtime_detect.pyの設定値を使う")
    parser.add_argument("--window_seconds", type=float, default=None,
                         help="WINDOW_SECONDSを一時的に上書きする(1秒/3秒/5秒モデルの比較用)。"
                              "--checkpoint_pathで指定するモデルの学習時の窓幅と必ず一致させること。"
                              "窓幅の違うチェックポイントと組み合わせると結果が無意味になるので注意")
    parser.add_argument("--tolerance_sec", type=float, default=TOLERANCE_SEC_DEFAULT,
                         help=f"正解区間の前後、マッチングを許容する時間(±秒)。デフォルト{TOLERANCE_SEC_DEFAULT}")
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
    expected_classes = sorted(
        set(rd.TARGET_CLASSES) | {g["label"] for g in ground_truth if not g["is_background"]})

    hop_seconds_used = args.hop_seconds if args.hop_seconds is not None else rd.HOP_SECONDS
    window_seconds_used = args.window_seconds if args.window_seconds is not None else rd.WINDOW_SECONDS

    # SAVE_CLIP_ON_DETECTは「RealtimeDetector.step()がaudio_clipをon_detect()へ渡すかどうか」の
    # ゲートも兼ねているため、評価中は常にTrueにしておく(距離推定にはこのクリップが必要)。
    # HOP_SECONDS/WINDOW_SECONDSも、指定があれば実際に使う値に揃えておく
    # (stepの内部処理・遅延判定などがグローバル値を直接参照しているため、実際の値と一致させる)。
    overrides = {
        "SAVE_CLIP_ON_DETECT": True,
        "HOP_SECONDS": hop_seconds_used,
        "WINDOW_SECONDS": window_seconds_used,
    }
    with override_globals(**overrides):
        detections, rms_trace, raw_predictions = run_simulation(
            args.audio, args.checkpoint_path, args.distance_checkpoint_path,
            hop_seconds=args.hop_seconds, save_clips_dir=save_clips_dir)

    match_results, unmatched_gt_idx = match_detections(detections, ground_truth, args.tolerance_sec)
    metrics = compute_metrics(match_results, ground_truth, unmatched_gt_idx, expected_classes)

    save_outputs(output_dir, args.audio, args.ground_truth, ground_truth,
                 match_results, metrics, rms_trace, raw_predictions,
                 hop_seconds_used, window_seconds_used, args.tolerance_sec)
    print_summary(metrics)


if __name__ == "__main__":
    main()