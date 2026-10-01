# ==============================================================================
# モジュール全体の概要説明
# ==============================================================================
"""
test_realtime_detect.py が出力する評価結果(eval_runs/<実行日時>_<音声ファイル名>/run_info.json)
をブラウザ上でタイムライン表示するための、テスト用デモの簡易フロントエンド。

【frontend_app.py との違い(重要)】
    frontend_app.py は本番の detections.db(現在進行中のリアルタイム状態)を見るためのもので、
    このスクリプトは test_realtime_detect.py が生成した「1回分の評価実行結果」を後から見返す
    ためのものである。本番の detections.db / live_status.json には一切アクセスしない
    (読み込むのは eval_runs/ 以下の run_info.json だけ)。ポート番号も frontend_app.py
    (5000番)と衝突しないよう既定で5001番にしている。

【目的(重要)】
    「検知がどのタイミングで発生しているか」を、壁時計ではなく音声先頭からの経過秒数で
    確認できるようにすること。そのため、タイムラインのx軸・テーブルの時刻列は
    すべて経過秒数(例: 12.50s)で統一しており、壁時計時刻は一切表示しない。

事前準備:
    pip install flask

使い方:
    1. 先に test_realtime_detect.py を実行し、eval_runs/ 以下に評価結果を作っておく
    2. このスクリプトを起動する: python test_frontend.py
    3. ブラウザで http://127.0.0.1:5001 を開く
    (複数回評価を実行していれば、画面上部のプルダウンで実行結果を切り替えられる)

    特定の1件だけを見たい場合:
    python test_frontend.py --run_dir "eval_runs/20260921_153000_test"
"""

# ==============================================================================
# ライブラリのインポート
# ==============================================================================
import argparse
import json
import os

from flask import Flask, jsonify, render_template_string, request

# ============================ 設定 (ここを変更) ============================
EVAL_RUNS_DIR_DEFAULT = "eval_runs"  # test_realtime_detect.py のデフォルト出力先と揃えてある
HOST = "127.0.0.1"
PORT = 5001  # frontend_app.py(本番用、5000番)とポートが衝突しないよう変えてある
# ===========================================================================

app = Flask(__name__)


# ------------------------------------------------------------------------------
# 評価結果フォルダの一覧・読み込み
# ------------------------------------------------------------------------------
def list_runs(base_dir):
    """base_dir直下で run_info.json を持つフォルダを、新しいものから順に並べて返す。"""
    if not base_dir or not os.path.isdir(base_dir):
        return []
    runs = []
    for name in sorted(os.listdir(base_dir)):
        info_path = os.path.join(base_dir, name, "run_info.json")
        if os.path.isfile(info_path):
            runs.append({"name": name, "path": os.path.join(base_dir, name),
                         "mtime": os.path.getmtime(info_path)})
    runs.sort(key=lambda r: r["mtime"], reverse=True)  # 新しい実行結果を先頭に[説明]
    return runs


def resolve_run_dir(name):
    """--run_dir で1件だけ指定されている場合はそれを最優先で使う。
    それ以外は eval_runs_dir 配下から探す(nameが無ければ最新の1件)。"""
    single = app.config.get("SINGLE_RUN_DIR")
    if single:
        return single
    base_dir = app.config.get("EVAL_RUNS_DIR", EVAL_RUNS_DIR_DEFAULT)
    if not name:
        runs = list_runs(base_dir)
        return runs[0]["path"] if runs else None
    return os.path.join(base_dir, name)


# ------------------------------------------------------------------------------
# Web API エンドポイント
# ------------------------------------------------------------------------------
@app.route("/api/runs")
def api_runs():
    """プルダウンに出す、選択可能な評価結果フォルダ名の一覧を返す。"""
    single = app.config.get("SINGLE_RUN_DIR")
    if single:
        return jsonify({"runs": [os.path.basename(os.path.normpath(single))]})
    base_dir = app.config.get("EVAL_RUNS_DIR", EVAL_RUNS_DIR_DEFAULT)
    return jsonify({"runs": [r["name"] for r in list_runs(base_dir)]})


@app.route("/api/run")
def api_run():
    """指定された(または最新の)評価結果1件の中身(run_info.json)をそのまま返す。"""
    run_dir = resolve_run_dir(request.args.get("dir"))
    if not run_dir:
        return jsonify({"error": "no_runs_found",
                         "message": "eval_runsフォルダに評価結果が見つかりません。"
                                    "先に test_realtime_detect.py を実行してください。"}), 404
    info_path = os.path.join(run_dir, "run_info.json")
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return jsonify({"error": "load_failed", "message": str(e)}), 404
    return jsonify(info)


# ==============================================================================
# フロントエンド（HTML / CSS / JavaScript）のテンプレート定義
# ==============================================================================
PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>評価結果ビューア(テスト用デモ)</title>
<style>
  :root {
    --bg: #14161a; --panel: #1e2126; --border: #2f333a;
    --text: #e6e8eb; --muted: #9aa0aa;
    --tp: #4caf50; --confused: #ff9800; --fp: #f44336;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 16px; background: var(--bg); color: var(--text);
    font-family: -apple-system, "Segoe UI", "Hiragino Kaku Gothic ProN", sans-serif;
  }
  h1 { font-size: 18px; margin: 0 0 4px 0; }
  .sub { color: var(--muted); font-size: 12px; margin-bottom: 14px; }
  .panel {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 14px; margin-bottom: 14px;
  }
  .toolbar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  select, button {
    background: #262a31; color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 10px; font-size: 13px;
  }
  .meta-line { font-size: 12px; color: var(--muted); margin-top: 8px; }
  .metrics-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
    gap: 10px;
  }
  .metric { background: #262a31; border-radius: 6px; padding: 8px 10px; }
  .metric .label { display: block; font-size: 11px; color: var(--muted); }
  .metric .value { display: block; font-size: 18px; font-weight: 600; margin-top: 2px; }
  .legend { display: flex; gap: 16px; font-size: 12px; color: var(--muted); margin-top: 10px; flex-wrap: wrap; }
  .legend span.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }
  #timelineWrap { width: 100%; overflow-x: auto; }
  #timelineCanvas { width: 100%; display: block; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); white-space: nowrap; }
  th { color: var(--muted); font-weight: 500; position: sticky; top: 0; background: var(--panel); }
  .table-wrap { max-height: 320px; overflow-y: auto; }
  #emptyMsg { display: none; color: var(--muted); padding: 20px; text-align: center; }
</style>
</head>
<body>
  <h1>評価結果ビューア(テスト用デモ)</h1>
  <div class="sub">test_realtime_detect.py の出力を表示しています。時刻はすべて「音声先頭からの経過秒数」です(壁時計ではありません)。</div>

  <div class="panel">
    <div class="toolbar">
      <label for="runSelect">評価結果:</label>
      <select id="runSelect"></select>
      <span class="meta-line" id="audioPathLabel"></span>
      <span class="meta-line" id="hopLabel"></span>
      <span class="meta-line" id="toleranceLabel"></span>
    </div>
  </div>

  <div id="emptyMsg" class="panel">評価結果がまだありません。先に test_realtime_detect.py を実行してください。</div>

  <div class="panel" id="metricsPanel">
    <div class="metrics-grid" id="metricsBox"></div>
  </div>

  <div class="panel">
    <div id="timelineWrap">
      <canvas id="timelineCanvas" height="280"></canvas>
    </div>
    <div class="legend">
      <span><span class="dot" style="background:var(--tp)"></span>TP(時間・クラスとも一致)</span>
      <span><span class="dot" style="background:var(--confused)"></span>Confused(時間は一致・クラス違い)</span>
      <span><span class="dot" style="background:var(--fp)"></span>FP(対応する正解が窓内に無い)</span>
      <span><span class="dot" style="background:rgba(150,150,150,0.5)"></span>正解イベント + 許容窓(±tolerance_sec)</span>
    </div>
  </div>

  <div class="panel">
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>経過秒数</th><th>予測クラス</th><th>確信度</th><th>推定距離</th>
            <th>判定</th><th>対応する正解の時刻</th><th>遅延Δt</th>
          </tr>
        </thead>
        <tbody id="detTableBody"></tbody>
      </table>
    </div>
  </div>

<script>
let lastRunData = null;

function statusColor(status) {
  if (status === 'TP') return getComputedStyle(document.documentElement).getPropertyValue('--tp').trim();
  if (status === 'Confused') return getComputedStyle(document.documentElement).getPropertyValue('--confused').trim();
  return getComputedStyle(document.documentElement).getPropertyValue('--fp').trim();
}

function renderMetrics(m) {
  const fmt = v => (v === null || v === undefined) ? '-' : v;
  document.getElementById('metricsBox').innerHTML = `
    <div class="metric"><span class="label">Macro Precision</span><span class="value">${m.macro_precision.toFixed(3)}</span></div>
    <div class="metric"><span class="label">Macro Recall</span><span class="value">${m.macro_recall.toFixed(3)}</span></div>
    <div class="metric"><span class="label">Macro F1</span><span class="value">${m.macro_f1.toFixed(3)}</span></div>
    <div class="metric"><span class="label">TP</span><span class="value" style="color:var(--tp)">${m.counts.tp}</span></div>
    <div class="metric"><span class="label">Confused</span><span class="value" style="color:var(--confused)">${m.counts.confused}</span></div>
    <div class="metric"><span class="label">FP</span><span class="value" style="color:var(--fp)">${m.counts.fp}</span></div>
    <div class="metric"><span class="label">FN</span><span class="value">${m.counts.fn}</span></div>
    <div class="metric"><span class="label">遅延Δt 平均/最大(TPのみ)</span><span class="value">${fmt(m.delay.mean_ms)} / ${fmt(m.delay.max_ms)} ms</span></div>
  `;
}

function renderTable(dets) {
  document.getElementById('detTableBody').innerHTML = dets.map(d => `
    <tr style="border-left: 4px solid ${statusColor(d.match_status)}">
      <td>${d.elapsed_sec.toFixed(2)}s</td>
      <td>${d.predicted_class}</td>
      <td>${d.confidence.toFixed(3)}</td>
      <td>${d.estimated_distance_cm != null ? d.estimated_distance_cm.toFixed(1) + 'cm' : '-'}</td>
      <td>${d.match_status}</td>
      <td>${d.matched_gt_elapsed_sec != null ? d.matched_gt_elapsed_sec.toFixed(2) + 's' : '-'}</td>
      <td>${d.delta_t_ms != null ? Math.round(d.delta_t_ms) + 'ms' : '-'}</td>
    </tr>
  `).join('');
}

function drawTimeline(data) {
  const canvas = document.getElementById('timelineCanvas');
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.parentElement.clientWidth;
  const cssHeight = 280;
  canvas.style.width = cssWidth + 'px';
  canvas.style.height = cssHeight + 'px';
  canvas.width = cssWidth * dpr;
  canvas.height = cssHeight * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const marginLeft = 8, marginRight = 8, marginTop = 24, marginBottom = 26;
  const plotW = cssWidth - marginLeft - marginRight;
  const plotH = cssHeight - marginTop - marginBottom;

  const rms = data.rms_trace || [];
  const gt = data.ground_truth || [];
  const dets = data.detections || [];
  const tol = data.tolerance_sec || 0.5;

  let maxT = 1;
  rms.forEach(p => { maxT = Math.max(maxT, p[0]); });
  gt.forEach(e => { maxT = Math.max(maxT, e.elapsed_sec); });
  dets.forEach(d => { maxT = Math.max(maxT, d.elapsed_sec); });
  maxT = maxT * 1.02 || 1;
  const xOf = t => marginLeft + (t / maxT) * plotW;

  // 背景: RMS推移(面グラフ、壁時計ではなく経過秒数で描画)
  let maxRms = 0.001;
  rms.forEach(p => { maxRms = Math.max(maxRms, p[1]); });
  if (rms.length > 0) {
    ctx.beginPath();
    ctx.moveTo(xOf(rms[0][0]), marginTop + plotH);
    rms.forEach(p => {
      ctx.lineTo(xOf(p[0]), marginTop + plotH - (p[1] / maxRms) * plotH * 0.75);
    });
    ctx.lineTo(xOf(rms[rms.length - 1][0]), marginTop + plotH);
    ctx.closePath();
    ctx.fillStyle = 'rgba(100, 150, 220, 0.22)';
    ctx.fill();
  }

  // 正解イベント: 許容窓(shaded area)+縦線+ラベル
  gt.forEach(e => {
    const xC = xOf(e.elapsed_sec);
    const xL = xOf(Math.max(0, e.elapsed_sec - tol));
    const xR = xOf(e.elapsed_sec + tol);
    ctx.fillStyle = 'rgba(150, 150, 150, 0.15)';
    ctx.fillRect(xL, marginTop, xR - xL, plotH);
    ctx.strokeStyle = 'rgba(190, 190, 190, 0.9)';
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(xC, marginTop);
    ctx.lineTo(xC, marginTop + plotH);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#cfd3d8';
    ctx.font = '10px sans-serif';
    ctx.save();
    ctx.translate(xC + 3, marginTop + 10);
    ctx.fillText(e.label, 0, 0);
    ctx.restore();
  });

  // 検知マーカー(色は判定結果ごと)
  dets.forEach(d => {
    const x = xOf(d.elapsed_sec);
    const y = marginTop + plotH * 0.55;
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = statusColor(d.match_status);
    ctx.fill();
    ctx.strokeStyle = '#14161a';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });

  // 時間軸(経過秒数のみ。壁時計は表示しない)
  ctx.strokeStyle = '#555';
  ctx.beginPath();
  ctx.moveTo(marginLeft, marginTop + plotH);
  ctx.lineTo(marginLeft + plotW, marginTop + plotH);
  ctx.stroke();
  const tickCount = Math.min(10, Math.max(4, Math.round(plotW / 90)));
  ctx.fillStyle = '#9aa0aa';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  for (let i = 0; i <= tickCount; i++) {
    const t = (maxT / tickCount) * i;
    const x = xOf(t);
    ctx.beginPath();
    ctx.moveTo(x, marginTop + plotH);
    ctx.lineTo(x, marginTop + plotH + 4);
    ctx.stroke();
    ctx.fillText(t.toFixed(1) + 's', x, marginTop + plotH + 16);
  }
  ctx.textAlign = 'left';
}

async function loadRun(name) {
  const res = await fetch('/api/run' + (name ? ('?dir=' + encodeURIComponent(name)) : ''));
  if (!res.ok) {
    document.getElementById('emptyMsg').style.display = 'block';
    return;
  }
  document.getElementById('emptyMsg').style.display = 'none';
  const data = await res.json();
  lastRunData = data;
  document.getElementById('audioPathLabel').textContent = '音声: ' + data.audio_path;
  document.getElementById('hopLabel').textContent = 'HOP_SECONDS: ' + data.hop_seconds + 's';
  document.getElementById('toleranceLabel').textContent = '許容窓: ±' + data.tolerance_sec + 's';
  renderMetrics(data.metrics);
  drawTimeline(data);
  renderTable(data.detections);
}

async function loadRuns() {
  const res = await fetch('/api/runs');
  const data = await res.json();
  const select = document.getElementById('runSelect');
  select.innerHTML = data.runs.map(r => `<option value="${r}">${r}</option>`).join('');
  if (data.runs.length > 0) {
    await loadRun(data.runs[0]);
  } else {
    document.getElementById('emptyMsg').style.display = 'block';
  }
}

document.getElementById('runSelect').addEventListener('change', e => loadRun(e.target.value));
window.addEventListener('resize', () => { if (lastRunData) drawTimeline(lastRunData); });

loadRuns();
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
        description="test_realtime_detect.py の評価結果をブラウザで見るビューア"
                     "(本番のdetections.db/live_status.jsonには一切アクセスしません)")
    parser.add_argument("--run_dir", type=str, default=None,
                         help="単一の評価結果フォルダ(run_info.jsonがある場所)を直接指定する場合。"
                              "指定すると画面上部のプルダウンはその1件のみになる")
    parser.add_argument("--eval_runs_dir", type=str, default=EVAL_RUNS_DIR_DEFAULT,
                         help="複数の評価結果フォルダをまとめて置いてある親フォルダ"
                              "(test_realtime_detect.pyのデフォルト出力先と同じ)")
    parser.add_argument("--host", type=str, default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    if args.run_dir:
        app.config["SINGLE_RUN_DIR"] = args.run_dir
    app.config["EVAL_RUNS_DIR"] = args.eval_runs_dir

    print(f"起動します: http://{args.host}:{args.port}")
    if args.run_dir:
        print(f"  表示対象: {args.run_dir} (この1件のみ)")
    else:
        print(f"  表示対象: {args.eval_runs_dir}/ 以下の評価結果一覧")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
