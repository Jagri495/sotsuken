"""
シャカパチ検知システム用 データベースモジュール (SQLite)

【機能】
1. テーブルの初期化 (init_db)
2. 検知データの保存 (insert_detection)
3. リアルタイム状態の更新/取得 (save_live_status / get_live_status)
4. 最新の検知履歴の取得 (get_recent_detections)
"""

import json
import sqlite3
from typing import Any, Dict, List, Optional

DB_PATH = "detections.db"


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """データベース接続を取得（辞書型で値を取得できる row_factory を設定）"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """テーブルが存在しない場合に初期化作成する"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. 検知履歴テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL,
                estimated_distance_cm REAL,
                rms REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. リアルタイム状態管理用テーブル（最新1件のみ保持）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS live_status (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                data_json TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()


def insert_detection(
    timestamp: str,
    predicted_class: str,
    confidence: Optional[float] = None,
    estimated_distance_cm: Optional[float] = None,
    rms: Optional[float] = None,
    db_path: str = DB_PATH,
) -> int:
    """検知レコードを1件挿入し、発行されたIDを返す"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO detections (timestamp, predicted_class, confidence, estimated_distance_cm, rms)
            VALUES (?, ?, ?, ?, ?)
            """,
            (timestamp, predicted_class, confidence, estimated_distance_cm, rms),
        )
        conn.commit()
        return cursor.lastrowid


def get_recent_detections(limit: int = 15, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """最新の検知履歴を取得（フロントエンド表示用）"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, timestamp, predicted_class, confidence, estimated_distance_cm, rms
            FROM detections
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def save_live_status(status_dict: Dict[str, Any], db_path: str = DB_PATH) -> None:
    """リアルタイム状態（live_status.jsonの代わり）をDBに保存"""
    data_json = json.dumps(status_dict, ensure_ascii=False)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO live_status (id, data_json, updated_at)
            VALUES (1, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                data_json = excluded.data_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (data_json,),
        )
        conn.commit()


def get_live_status(db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """リアルタイム状態（live_status）を取得"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT data_json FROM live_status WHERE id = 1")
        row = cursor.fetchone()
        if row and row["data_json"]:
            return json.loads(row["data_json"])
        return None


if __name__ == "__main__":
    # 単体テスト用動作確認
    init_db()
    print("データベース初期化成功: detections.db")

    # テスト挿入
    new_id = insert_detection(
        timestamp="2026-09-16 12:00:00",
        predicted_class="shakapachi",
        confidence=0.985,
        estimated_distance_cm=45.2,
        rms=0.035,
    )
    print(f"テストデータ挿入完了 (ID: {new_id})")

    # 取得テスト
    recent = get_recent_detections(limit=5)
    print("最新履歴:", recent)