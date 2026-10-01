# ==============================================================================
# モジュール全体の概要説明
# ==============================================================================
"""
シャカパチ検知の「フィードバック」部分のプロトタイプ(周辺視野の発光UI + 非侵入的な効果音)。

【位置づけ(重要)】
    アップロードいただいた「シャカパチ検知＆発光フィードバックUI 技術仕様書」のうち、
    検知アルゴリズム部分(GMM/SmartSifterによる教師なし異常検知)は採用していない
    (検知は引き続きrealtime_detect.py側のEfficientAT分類器がそのまま担当する)。
    このファイルが実装しているのは、あくまで仕様書の「3. 視覚フィードバック要件」
    「4. 聴覚フィードバック要件」「5. フロントエンド構成」の部分だけを、今の実システムに
    翻訳したものである。

    仕様書の概念と、このファイルでの実際の対応は以下の通り(要調整の可能性あり、実際に
    動かしてみてズレがあれば直す前提):
      - 外れ値スコア(μ+3σ)              → 分類の confidence で代用
      - 環境音学習の収束状況(GMM前提)     → 該当なし。代わりに「realtime_detect.pyが
                                            稼働中か/最終更新はいつか」を表示
      - 学習リセットボタン(GMM前提)       → 該当なし。今回は実装していない
      - ILDによる左右の発光マッピング     → 現状未実装(README 3章「検討中」のまま)。
                                            左右非対応で全周を光らせる形にしてあり、
                                            方向推定ができた時点で後付けする想定
      - 緑→黄→橙のグラデーション、
        フェードイン/アウト               → そのまま踏襲(confidenceの高さで色を決定)
      - 聴覚フィードバック(DAFX)         → 「今聴いている音声にリアルタイムでエフェクトを
                                            かける」のはシステム音声のキャプチャ+DSP処理が
                                            必要で難易度が高いため見送り、代わりに短い
                                            非侵入的な効果音(帯域を絞ったホワイトノイズの
                                            盛り上がり、フェードイン/アウト付き、約0.7秒)を
                                            追加で鳴らす方式にしている
      - 感度調整スライダー                → 表示・効果音を鳴らすかどうかのフィルタとしてのみ
                                            機能する。realtime_detect.py側のCONFIDENCE_
                                            THRESHOLD等は一切変更しない(このファイルを
                                            分離させたいという方針のため)

【他ファイルとの関係(重要)】
    realtime_detect.py / frontend_app.py には一切手を加えていない。db.py の読み取り専用関数
    (get_live_status, get_recent_detections)だけを「参照」して使っている。このファイル単体を
    今後どれだけ改造しても、検知本体やdb.pyには影響しない。

    取りこぼし防止の設計: 一瞬の状態を表す live_status ではなく、追記専用の detections テーブル
    (get_recent_detections)をポーリングしている。live_status は毎step()で上書きされるため、
    ポーリング間隔によっては「発火した瞬間」を取りこぼす可能性があるが、detections テーブルは
    発火したときだけ追記される記録なので、新しい行が増えたかどうかを見れば取りこぼさない。

事前準備:
    pip install flask numpy sounddevice

使い方:
    1. 別のターミナルで realtime_detect.py を起動しておく
    2. python habit_feedback_ui.py  (既定ポート 5002。frontend_app.py の5000、
       test_frontend.py の5001と衝突しないよう分けてある)
    3. ブラウザで http://127.0.0.1:5002 を開き、画面端に配置する(周辺視野に入る位置)
"""

# ==============================================================================
# ライブラリのインポート
# ==============================================================================
import argparse
import threading
import time
from datetime import datetime

import numpy as np
from flask import Flask, jsonify, render_template_string, request

from db import get_live_status, get_recent_detections  # 読み取り専用の参照のみ。書き込みはしない

# ============================ 設定 (ここを変更) ============================
HOST = "127.0.0.1"
PORT = 5002  # frontend_app.py(5000)・test_frontend.py(5001)と衝突しないよう分けてある
POLL_INTERVAL_SECONDS = 0.4     # detections テーブルを何秒おきに確認するか
STALE_AFTER_SECONDS = 3.0       # live_statusの最終更新がこれより古いと「未接続/停止中」表示にする
GLOW_DECAY_SECONDS = 3.0        # 発光がフェードアウトしきるまでの秒数
DEFAULT_SENSITIVITY = 0.6       # 感度スライダーの初期値(この値未満のconfidenceは表示・音を出さない)
SOUND_DURATION_SECONDS = 0.7    # 効果音の長さ(仕様書の「持続時間1秒以内」に準拠)
SOUND_BAND_HZ = (300, 1400)     # 効果音の帯域(バンドパス)
MAX_LOG_ENTRIES = 30            # ログエリアに保持する件数
# ===========================================================================

app = Flask(__name__)

# アプリ全体で共有する状態。単一ユーザー・ローカル利用のツールなので、簡易ロックのみで保護する。
_state_lock = threading.Lock()
app_state = {
    "sensitivity": DEFAULT_SENSITIVITY,
    "connected": False,
    "last_update": None,          # realtime_detect.py側の最終更新時刻(文字列)
    "last_event": None,           # 直近に(感度を超えて)フィードバックした検知1件
    "log": [],                    # 直近の検知ログ(表示用、フィルタ後)
}


# ------------------------------------------------------------------------------
# 効果音(非侵入的な「盛り上がり」、単純なアラーム音にしないための処理)
# ------------------------------------------------------------------------------
def generate_swell_sound(sample_rate=44100, duration=SOUND_DURATION_SECONDS, band=SOUND_BAND_HZ):
    """帯域を絞ったホワイトノイズに、なだらかな音量エンベロープ(フェードイン/アウト)をかけた
    短い効果音を生成する。単純なビープ音ではなく「音質の変化」に近い、控えめな通知を狙ったもの。"""
    n = int(sample_rate * duration)
    noise = np.random.uniform(-1.0, 1.0, n)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    spectrum[~band_mask] = 0  # 帯域外の周波数成分を除去(簡易バンドパスフィルター)
    filtered = np.fft.irfft(spectrum, n)
    peak = np.max(np.abs(filtered))
    if peak > 0:
        filtered = filtered / peak
    envelope = np.hanning(n)  # 急激な音量変化を避けるための、なだらかな山形の包絡線
    return (filtered * envelope * 0.35).astype(np.float32)  # 0.35で控えめな音量に抑える


_SWELL_CACHE = {}  # (band幅, 継続時間)ごとに生成済みの音を使い回す(毎回生成しなくて済むように)


def play_feedback_sound(confidence_band):
    """検知時に効果音を鳴らす。音自体は確信度帯によらず同じ(方向・強さの作り分けは今回は行わない)。
    音が鳴らせない環境でもクラッシュせず、フィードバック処理自体は継続する。"""
    try:
        import sounddevice as sd  # 環境依存のため遅延import(realtime_detect.pyと同じ方針)
        key = "default"
        if key not in _SWELL_CACHE:
            _SWELL_CACHE[key] = generate_swell_sound()
        sd.play(_SWELL_CACHE[key], samplerate=44100)  # 非同期再生。ポーリングループをブロックしない
    except Exception as e:  # noqa: BLE001 - 音が原因でフィードバック全体が止まらないようにする
        print(f"  ℹ️ 効果音を再生できませんでした({e})。フィードバック処理自体は継続します。", flush=True)


# ------------------------------------------------------------------------------
# confidenceから発光色を決める(仕様書の 緑(低)→黄(中)→橙(高) を踏襲)
# ------------------------------------------------------------------------------
def confidence_to_band(confidence):
    if confidence >= 0.9:
        return "high"    # 橙
    elif confidence >= 0.75:
        return "mid"     # 黄
    return "low"          # 緑


# ------------------------------------------------------------------------------
# バックグラウンドでdetectionsテーブルをポーリングするスレッド
# ------------------------------------------------------------------------------
def poll_loop():
    """detections テーブルを定期的に確認し、新しい行(=新しい検知)が増えていたら、
    感度スライダーの閾値を超えているものだけフィードバック(発光+効果音)を発火させる。
    live_statusは別途ポーリングし、「realtime_detect.pyが今動いているか」の判定にのみ使う。"""
    last_seen_id = None  # 起動時点で存在する履歴は「新規」として扱わないよう、最初にIDだけ記録する

    while True:
        try:
            recent = get_recent_detections(limit=10)  # 新しい順(id DESC)で返る
        except Exception as e:  # noqa: BLE001 - DBが一時的に無くても監視は続ける
            print(f"  ⚠️ detections取得に失敗しました: {e}", flush=True)
            recent = []

        if recent:
            if last_seen_id is None:
                last_seen_id = recent[0]["id"]  # 起動時の最新IDを基準にする(過去分は再生しない)
            else:
                new_rows = [r for r in recent if r["id"] > last_seen_id]
                new_rows.sort(key=lambda r: r["id"])  # 古い→新しいの順に処理する
                for row in new_rows:
                    _handle_new_detection(row)
                if new_rows:
                    last_seen_id = new_rows[-1]["id"]

        # live_status側は「今動いているか」の判定だけに使う(取りこぼしを気にしなくてよい用途)
        try:
            live = get_live_status()
        except Exception:
            live = None
        with _state_lock:
            if live and live.get("timestamp"):
                app_state["last_update"] = live["timestamp"]
                try:
                    last_dt = datetime.strptime(live["timestamp"], "%Y-%m-%d %H:%M:%S.%f")
                    app_state["connected"] = (datetime.now() - last_dt).total_seconds() < STALE_AFTER_SECONDS
                except ValueError:
                    app_state["connected"] = False
            else:
                app_state["connected"] = False

        time.sleep(POLL_INTERVAL_SECONDS)


def _handle_new_detection(row):
    """新しい検知1件を、感度スライダーの閾値と照らし合わせてフィードバックするかどうかを決める。
    ここで弾かれた検知も、db.py側の履歴(detections.db)には残ったままなので消えるわけではなく、
    あくまで「今回の発光・効果音の対象にするかどうか」だけの、表示側のフィルタである。"""
    confidence = row.get("confidence") or 0.0
    with _state_lock:
        sensitivity = app_state["sensitivity"]

    entry = {
        "elapsed_id": row["id"],
        "timestamp": row["timestamp"],
        "predicted_class": row["predicted_class"],
        "confidence": confidence,
        "estimated_distance_cm": row.get("estimated_distance_cm"),
        "band": confidence_to_band(confidence),
        "shown": confidence >= sensitivity,
    }

    with _state_lock:
        app_state["log"].insert(0, entry)
        app_state["log"] = app_state["log"][:MAX_LOG_ENTRIES]
        if entry["shown"]:
            app_state["last_event"] = {**entry, "server_time": time.time()}

    if entry["shown"]:
        play_feedback_sound(entry["band"])


# ------------------------------------------------------------------------------
# Web API エンドポイント
# ------------------------------------------------------------------------------
@app.route("/api/status")
def api_status():
    with _state_lock:
        return jsonify({
            "connected": app_state["connected"],
            "last_update": app_state["last_update"],
            "last_event": app_state["last_event"],
            "log": app_state["log"],
            "sensitivity": app_state["sensitivity"],
            "glow_decay_seconds": GLOW_DECAY_SECONDS,
            "server_time": time.time(),
        })


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json(silent=True) or {}
    sensitivity = data.get("sensitivity")
    if sensitivity is not None:
        try:
            sensitivity = max(0.0, min(1.0, float(sensitivity)))
            with _state_lock:
                app_state["sensitivity"] = sensitivity
        except (TypeError, ValueError):
            return jsonify({"error": "invalid_sensitivity"}), 400
    with _state_lock:
        return jsonify({"sensitivity": app_state["sensitivity"]})


# ==============================================================================
# フロントエンド（HTML / CSS / JavaScript）
# ==============================================================================
PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>habit feedback</title>
<style>
  :root {
    --glow-color: 76,175,80; /* デフォルトは緑(low) */
    --glow-opacity: 0;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; width: 100%; height: 100%;
    background: #0c0d10; color: #cfd3d8;
    font-family: -apple-system, "Segoe UI", "Hiragino Kaku Gothic ProN", sans-serif;
    overflow: hidden;
  }
  /* 周辺視野向けの発光: 画面端をぼかしたinset box-shadowで表現し、急な点滅ではなく
     opacityのなめらかな変化だけで存在を知らせる */
  #glow {
    position: fixed; inset: 0; pointer-events: none;
    box-shadow: inset 0 0 120px 40px rgba(var(--glow-color), var(--glow-opacity));
    transition: box-shadow 0.15s linear;
  }
  #header {
    position: fixed; top: 10px; left: 10px; font-size: 12px; color: #8a8f98;
    display: flex; align-items: center; gap: 8px;
  }
  #header .dot { width: 8px; height: 8px; border-radius: 50%; background: #555; }
  #header .dot.on { background: #4caf50; }
  #logArea {
    position: fixed; bottom: 10px; left: 10px; width: 300px; max-height: 160px;
    overflow-y: auto; font-size: 11px; color: #7d828b; line-height: 1.6;
  }
  #logArea .row { border-left: 3px solid #444; padding-left: 6px; margin-bottom: 3px; }
  #logArea .row.low { border-color: #4caf50; }
  #logArea .row.mid { border-color: #ffc107; }
  #logArea .row.high { border-color: #ff9800; }
  #panel {
    position: fixed; bottom: 10px; right: 10px; width: 220px;
    background: rgba(30, 33, 38, 0.85); border: 1px solid #2f333a; border-radius: 8px;
    padding: 10px; font-size: 11px;
  }
  #panel label { display: block; margin-bottom: 4px; color: #8a8f98; }
  #panel input[type=range] { width: 100%; }
  #sensitivityValue { color: #cfd3d8; }
</style>
</head>
<body>
  <div id="glow"></div>

  <div id="header">
    <span class="dot" id="connDot"></span>
    <span id="connLabel">接続確認中...</span>
  </div>

  <div id="logArea"></div>

  <div id="panel">
    <label>感度(この値未満は発光・効果音を出さない): <span id="sensitivityValue">0.60</span></label>
    <input type="range" id="sensitivitySlider" min="0.5" max="1.0" step="0.01" value="0.6">
    <div style="margin-top:6px; color:#6a6f78;">
      表示側のフィルタのみ。realtime_detect.py側の検知閾値は変更されません。
    </div>
  </div>

<script>
let sensitivity = 0.6;
let lastEventServerTime = null;
let lastEventBand = 'low';
let glowDecaySeconds = 3.0;

const bandColor = { low: '76,175,80', mid: '255,193,7', high: '255,152,0' };

function renderLog(entries) {
  const area = document.getElementById('logArea');
  area.innerHTML = entries.filter(e => e.shown).slice(0, 15).map(e => `
    <div class="row ${e.band}">
      ${e.timestamp.split(' ')[1] ? e.timestamp.split(' ')[1].slice(0,8) : e.timestamp}
      ${e.predicted_class} (${e.confidence.toFixed(2)})
      ${e.estimated_distance_cm != null ? ' / ' + e.estimated_distance_cm.toFixed(1) + 'cm' : ''}
    </div>
  `).join('');
}

async function pollStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();

    const dot = document.getElementById('connDot');
    const label = document.getElementById('connLabel');
    if (data.connected) {
      dot.classList.add('on');
      label.textContent = 'realtime_detect.py 稼働中';
    } else {
      dot.classList.remove('on');
      label.textContent = data.last_update ? '未接続(最終更新: ' + data.last_update + ')' : '未接続';
    }

    renderLog(data.log || []);
    glowDecaySeconds = data.glow_decay_seconds || 3.0;

    if (data.last_event) {
      // サーバー時刻とクライアント時刻のズレを吸収するため、サーバー基準の経過時間で計算する
      const elapsedSinceEvent = data.server_time - data.last_event.server_time;
      lastEventServerTime = Date.now() - elapsedSinceEvent * 1000;
      lastEventBand = data.last_event.band;
    }

    // スライダーは他タブ操作等で外部から変わる可能性もあるため、表示だけ同期しておく
    if (!sliderDragging) {
      sensitivity = data.sensitivity;
      document.getElementById('sensitivitySlider').value = sensitivity;
      document.getElementById('sensitivityValue').textContent = sensitivity.toFixed(2);
    }
  } catch (e) {
    document.getElementById('connLabel').textContent = 'サーバーに接続できません';
  }
}

// なめらかなフェードイン/アウトのため、発光の不透明度は毎フレーム計算する(点滅を避ける)
function animateGlow() {
  let opacity = 0;
  if (lastEventServerTime !== null) {
    const elapsed = (Date.now() - lastEventServerTime) / 1000;
    if (elapsed < 0.3) {
      opacity = elapsed / 0.3;  // フェードイン
    } else if (elapsed < glowDecaySeconds) {
      opacity = 1 - (elapsed - 0.3) / (glowDecaySeconds - 0.3);  // フェードアウト
    } else {
      opacity = 0;
    }
    opacity = Math.max(0, Math.min(1, opacity)) * 0.5;  // 主張しすぎないよう最大でも0.5に抑える
  }
  document.getElementById('glow').style.setProperty('--glow-opacity', opacity.toFixed(3));
  document.getElementById('glow').style.setProperty('--glow-color', bandColor[lastEventBand] || bandColor.low);
  requestAnimationFrame(animateGlow);
}

let sliderDragging = false;
let sliderDebounceTimer = null;
const slider = document.getElementById('sensitivitySlider');
slider.addEventListener('input', () => {
  sliderDragging = true;
  document.getElementById('sensitivityValue').textContent = parseFloat(slider.value).toFixed(2);
  clearTimeout(sliderDebounceTimer);
  sliderDebounceTimer = setTimeout(async () => {
    await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sensitivity: parseFloat(slider.value) }),
    });
    sliderDragging = false;
  }, 250);
});

setInterval(pollStatus, 500);
pollStatus();
requestAnimationFrame(animateGlow);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE_TEMPLATE)


# ------------------------------------------------------------------------------
# スクリプト実行時のエントリーポイント
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="シャカパチ検知のフィードバックUI(発光+効果音、開発・検証用プロトタイプ)")
    parser.add_argument("--host", type=str, default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    poll_thread = threading.Thread(target=poll_loop, daemon=True)
    poll_thread.start()

    print(f"起動します: http://{args.host}:{args.port}")
    print("  (realtime_detect.py を別プロセスで起動しておいてください)")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
