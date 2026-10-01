# ==============================================================================
# モジュール全体の概要説明（ドキュメント文字列）[cite: 1]
# ==============================================================================
"""
realtime_detect.py の検知状況をブラウザで見えるようにする、簡易フロントエンド。

realtime_detect.py はリアルタイム状態(現在の音量・予測クラス・検知中かどうか)と
検知履歴を db.py 経由で SQLite (detections.db) に書き込みます。
このスクリプトはその detections.db を定期的に読み込んで画面に表示するだけの、
シンプルなダッシュボードです。

なお realtime_detect.py は live_status.json / realtime_detections.csv にも
同じ内容をバックアップとして書き出していますが、このダッシュボードは
それらのファイルは読みません(DBのみを参照します)。

事前準備:
    pip install flask

使い方:
    1. 別のターミナルで realtime_detect.py を起動しておく
    2. このスクリプトを起動する: python frontend_app.py
    3. ブラウザで http://127.0.0.1:5000 を開く

【設計方針】
    realtime_detect.py 本体には一切手を加えず、db.py 経由で detections.db を
    読むだけの完全に独立したプロセスとして動きます(検知ロジックと表示ロジックの分離)。
    どちらのプロセスを先に止めても、もう片方に影響しません。
    2つのプロセスは同じ作業ディレクトリから起動し、同じ相対パスの
    detections.db (db.DB_PATH) を共有することを前提としています。
"""

# ==============================================================================
# ライブラリのインポート[cite: 1]
# ==============================================================================
import argparse  # コマンドライン引数（起動オプション）を処理する標準ライブラリ[cite: 1]

from flask import (  # Webアプリケーションを作成するためのFlaskフレームワークから機能を呼び出し[cite: 1]
    Flask,  # Flaskアプリケーション本体を作成するクラス[cite: 1]
    jsonify,  # 辞書型などのデータをJSON形式のHTTPレスポンスに変換する関数[cite: 1]
    render_template_string,  # Python上の文字列をHTMLテンプレートとして描画する関数[cite: 1]
)
from db import (  # 別ファイルの db.py からデータベース操作用の関数をインポート[cite: 1]
    get_live_status,  # 現在のリアルタイムステータスを取得する関数[cite: 1]
    get_recent_detections,  # 過去の検知履歴を取得する関数[cite: 1]
)

# ============================ 設定 (ここを変更) ============================
# リアルタイム状態・検知履歴は db.py 経由で detections.db (db.DB_PATH) から読む。
# realtime_detect.py がバックアップとして書き出す live_status.json /
# realtime_detections.csv は、このダッシュボードでは使用しない。
HOST = "127.0.0.1"  # サーバーを起動するIPアドレス (ローカルホスト)[cite: 1]
PORT = 5000  # サーバーを起動するポート番号[cite: 1]
POLL_INTERVAL_MS = 1000  # ブラウザ側がAPIを呼び出して自動更新する間隔 (1000ミリ秒 = 1秒)[cite: 1]
MAX_RECENT_DETECTIONS = 200  # 画面のテーブルに表示する最新履歴の最大件数(仮の値。表の枠は元々スクロール可能なので、
                              # まずは件数だけ増やしてログを追いやすくしている)[cite: 1]
# ===========================================================================

app = Flask(__name__)  # Flask のアプリケーションインスタンスを作成[cite: 1]


# ------------------------------------------------------------------------------
# Web API エンドポイント: /api/status[cite: 1]
# ------------------------------------------------------------------------------
@app.route("/api/status")  # ブラウザからの Ajax(fetch) リクエストを受け付けるURLパスを定義[cite: 1]
def api_status():  # 上記URLにアクセスがあった時に実行される関数[cite: 1]
    return jsonify({  # 辞書データを JSON 形式に変換してブラウザへレスポンス返却[cite: 1]
        "live": get_live_status(),  # 現在のリアルタイム情報を取得[cite: 1]
        "recent_detections": get_recent_detections(
            MAX_RECENT_DETECTIONS
        ),  # 最新の履歴件数を取得[cite: 1]
    })


# ==============================================================================
# フロントエンド（HTML / CSS / JavaScript）のテンプレート定義[cite: 1]
# ==============================================================================
PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8"> <!-- 文字コードをUTF-8に設定 -->
<title>シャカパチ検知モニター</title> <!-- ブラウザのタブに表示されるタイトル -->
<style>
    /* 画面全体のスタイル設定 */
    body { font-family: -apple-system, "Yu Gothic", "Meiryo", sans-serif; background: #111; color: #eee;
           margin: 0; padding: 24px; }
    h1 { font-size: 20px; margin-bottom: 20px; color: #aaa; }

    /* 波形グラフ領域のスタイル */
    .waveform-panel { background: #0a0a0a; border: 1px solid #2a2a2a; border-radius: 12px;
                       padding: 12px 16px 4px; margin-bottom: 24px; }
    #time-axis { width: 100%; height: 20px; display: block; }
    #waveform-canvas { width: 100%; height: 90px; display: block; }

    /* 検知ステータス表示カードのスタイル */
    .status-card { border-radius: 16px; padding: 32px; text-align: center; margin-bottom: 24px;
                   transition: background-color 0.2s; }
    .status-idle { background: #1e2a1e; } /* 監視中（緑系） */
    .status-detecting { background: #4a1414; box-shadow: 0 0 30px rgba(255,60,60,0.5); } /* 検知中（赤色に発光） */
    .status-silence { background: #1a1a1a; } /* 無音時（ダークグレー） */
    .status-label { font-size: 28px; font-weight: bold; margin-bottom: 8px; }
    .status-sub { font-size: 14px; color: #999; }

    /* 数値指標を表示するグリッドレイアウト */
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
    .metric { background: #1a1a1a; border-radius: 12px; padding: 16px; }
    .metric-label { font-size: 12px; color: #888; margin-bottom: 4px; }
    .metric-value { font-size: 22px; font-weight: bold; }

    /* 音量（RMS）メーターのバーのスタイル */
    .rms-bar-bg { background: #222; border-radius: 6px; height: 12px; margin-top: 8px; overflow: hidden; }
    .rms-bar-fill { background: #4a9eff; height: 100%; width: 0%; transition: width 0.2s; }

    /* 履歴テーブルのスタイル */
    .table-wrap { max-height: 320px; overflow-y: auto; border: 1px solid #2a2a2a; border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #2a2a2a; font-size: 13px; }
    th { color: #888; font-weight: normal; position: sticky; top: 0; background: #161616; }
    .no-data { color: #666; font-size: 13px; padding: 16px; text-align: center; }
    .footer-note { color: #555; font-size: 11px; margin-top: 24px; }
</style>
</head>
<body>
    <h1>🎙 シャカパチ検知モニター</h1>

    <!-- リアルタイム音声波形表示用のキャンバスエリア -->
    <div class="waveform-panel">
        <canvas id="time-axis"></canvas>
        <canvas id="waveform-canvas"></canvas>
    </div>

    <!-- メインの検知ステータス表示カード -->
    <div id="status-card" class="status-card status-idle">
        <div id="status-label" class="status-label">読み込み中...</div>
        <div id="status-sub" class="status-sub"></div>
    </div>

    <!-- メトリクス（音量・現在の分類）表示エリア -->
    <div class="grid">
        <div class="metric">
            <div class="metric-label">音量 (RMS)</div>
            <div id="rms-value" class="metric-value">-</div>
            <div class="rms-bar-bg"><div id="rms-bar" class="rms-bar-fill"></div></div>
        </div>
        <div class="metric">
            <div class="metric-label">現在の予測クラス / 確信度</div>
            <div id="pred-value" class="metric-value">-</div>
        </div>
    </div>

    <!-- 検知履歴テーブル -->
    <h2 style="font-size:16px; color:#aaa;">検知履歴(最新{{ max_rows }}件・スクロールで表示)</h2>
    <div class="table-wrap">
        <table id="history-table">
            <thead>
                <tr><th>時刻</th><th>クラス</th><th>確信度</th><th>推定距離</th></tr>
            </thead>
            <tbody id="history-body"></tbody>
        </table>
    </div>
    <div id="no-history" class="no-data" style="display:none;">まだ検知履歴がありません</div>

    <!-- フッター情報 -->
    <div class="footer-note">
        {{ poll_interval }}msごとに自動更新 / detections.db を読み取り中
    </div>

<script>
// キャンバス要素の取得
const waveformCanvas = document.getElementById("waveform-canvas");
const timeAxisCanvas = document.getElementById("time-axis");

// 高解像度ディスプレイ(Retina等)に対応させるためのキャンバス解像度調整関数
function resizeCanvases() {
    for (const c of [waveformCanvas, timeAxisCanvas]) {
        const rect = c.getBoundingClientRect();
        c.width = rect.width * devicePixelRatio;
        c.height = rect.height * devicePixelRatio;
    }
}
window.addEventListener("resize", resizeCanvases); // 画面リサイズ時に実行

// 音声波形をキャンバス上に描画する関数
function drawWaveform(rmsHistory) {
    const ctx = waveformCanvas.getContext("2d");
    const w = waveformCanvas.width, h = waveformCanvas.height;
    ctx.clearRect(0, 0, w, h); // Canvasのクリア
    if (!rmsHistory || rmsHistory.length === 0) return;

    const maxRms = Math.max(0.02, ...rmsHistory.map(p => p.rms)); // スケーリング用最大音量計算
    const barW = w / rmsHistory.length; // 棒グラフ1本あたりの幅
    const mid = h / 2; // Y軸の中央値

    rmsHistory.forEach((point, i) => {
        const barH = Math.max(2, (point.rms / maxRms) * (h * 0.9)); // 音量に応じた高さ
        ctx.fillStyle = point.in_event ? "#ff5555" : "#4a9eff"; // 検知中なら赤、通常時は青
        ctx.fillRect(i * barW, mid - barH / 2, Math.max(1, barW - 1), barH); // 描画
    });
}

// 波形グラフの上部に時間軸の目盛りを描画する関数
function drawTimeAxis(rmsHistory) {
    const ctx = timeAxisCanvas.getContext("2d");
    const w = timeAxisCanvas.width, h = timeAxisCanvas.height;
    ctx.clearRect(0, 0, w, h); // Canvasのクリア
    if (!rmsHistory || rmsHistory.length === 0) return;

    ctx.strokeStyle = "#444";
    ctx.fillStyle = "#888";
    ctx.font = (11 * devicePixelRatio) + "px sans-serif";
    ctx.textBaseline = "top";

    const n = rmsHistory.length;
    const tickEvery = Math.max(1, Math.floor(n / 8));  // 目盛りを間引いて表示
    for (let i = 0; i < n; i += tickEvery) {
        const x = (i / n) * w;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h * 0.4);
        ctx.stroke();
        ctx.fillText(rmsHistory[i].t, x + 2, h * 0.4); // 時刻テキストを描画
    }
}

// サーバー(/api/status)から最新データを取得して画面(DOM)を更新する非同期関数
async function refresh() {
    let res;
    try {
        res = await fetch("/api/status"); // APIを呼び出してレスポンスを待機
    } catch (e) {
        return; // サーバー側が一時的に読めなくても表示は維持する
    }
    const data = await res.json(); // レスポンスJSONを解析
    const live = data.live;

    // 各表示エレメントの取得
    const card = document.getElementById("status-card");
    const label = document.getElementById("status-label");
    const sub = document.getElementById("status-sub");
    const rmsValue = document.getElementById("rms-value");
    const rmsBar = document.getElementById("rms-bar");
    const predValue = document.getElementById("pred-value");

    // live データが存在しない場合 (バックエンド非起動時)
    if (!live) {
        label.textContent = "待機中(realtime_detect.py が起動していません)";
        card.className = "status-card status-idle";
        sub.textContent = "";
        drawWaveform([]);
        drawTimeAxis([]);
        return;
    }

    // 検知ステータスに応じて表示カードのCSSクラスとテキストを変更
    if (live.in_event) {
        card.className = "status-card status-detecting";
        label.textContent = "🔔 検知中: " + live.predicted_class;
    } else if (live.is_silence) {
        card.className = "status-card status-silence";
        label.textContent = "無音(待機中)";
    } else {
        card.className = "status-card status-idle";
        label.textContent = "監視中";
    }
    sub.textContent = "最終更新: " + live.timestamp;

    // 音量(RMS)数値およびゲージバーの表示更新
    rmsValue.textContent = live.rms.toFixed(4);
    const barPct = Math.min(100, (live.rms / (live.silence_threshold * 5)) * 100);
    rmsBar.style.width = barPct + "%";

    // 予測クラス・確信度の表示更新
    predValue.textContent = live.predicted_class
        ? live.predicted_class + " (" + (live.confidence !== null ? live.confidence.toFixed(3) : "-") + ")"
        : "-";

    // グラフの再描画
    drawWaveform(live.rms_history);
    drawTimeAxis(live.rms_history);

    // 検知履歴テーブルの更新
    const tbody = document.getElementById("history-body");
    const noHistory = document.getElementById("no-history");
    tbody.innerHTML = ""; // テーブル内容を一旦クリア
    if (data.recent_detections.length === 0) {
        noHistory.style.display = "block"; // データがない場合メッセージを表示
    } else {
        noHistory.style.display = "none";
        for (const row of data.recent_detections) { // 各行をテーブル要素として動的生成
            const tr = document.createElement("tr");
            tr.innerHTML = `<td>${row.timestamp || ""}</td>
                            <td>${row.predicted_class || ""}</td>
                            <td>${row.confidence || ""}</td>
                            <td>${row.estimated_distance_cm ? row.estimated_distance_cm + "cm" : "-"}</td>`;
            tbody.appendChild(tr);
        }
    }
}

// 初期化処理
resizeCanvases(); // 初回キャンバスサイズ調整
setInterval(refresh, {{ poll_interval }}); // 指定間隔 (ミリ秒) ごとに refresh 関数を繰り返し実行
refresh(); // 初回データロード実行
</script>
</body>
</html>
"""


# ------------------------------------------------------------------------------
# ルートURL ( http://127.0.0.1:5000/ ) にアクセスされた時の処理[cite: 1]
# ------------------------------------------------------------------------------
@app.route("/")  # トップページへのアクセスを定義[cite: 1]
def index():  # トップページにアクセスされた時の処理関数[cite: 1]
    return render_template_string(  # 定義済みの HTML テンプレート文字列をレンダリングして返却[cite: 1]
        PAGE_TEMPLATE,
        poll_interval=POLL_INTERVAL_MS,  # テンプレート内の {{ poll_interval }} を置換[cite: 1]
        max_rows=MAX_RECENT_DETECTIONS,  # テンプレート内の {{ max_rows }} を置換[cite: 1]
    )


# ==============================================================================
# メイン処理 (このスクリプトが直接実行された時のみ動く)[cite: 1]
# ==============================================================================
if __name__ == "__main__":  # `python frontend_app.py` として実行された場合に成立[cite: 1]
    parser = argparse.ArgumentParser(  # コマンドライン引数のパーサー（解析器）を生成[cite: 1]
        description="シャカパチ検知モニター(簡易フロントエンド)"
    )
    # 起動オプション `--host` の定義[cite: 1]
    parser.add_argument("--host", type=str, default=HOST)
    # 起動オプション `--port` の定義[cite: 1]
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()  # コマンドライン引数を実際に解析して値を取得[cite: 1]

    # 起動メッセージを標準出力に出力[cite: 1]
    print(f"ダッシュボードを起動します: http://{args.host}:{args.port}")

    # Flask 開発用 Web サーバーを起動[cite: 1]
    app.run(host=args.host, port=args.port, debug=False)