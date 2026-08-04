"""
AgriVision System A — 数据层（SQLite）
=======================================
管理三张表：
  - diagnose_history: 诊断历史记录（类别、置信度、检测框、概率分布）
  - async_tasks:      异步诊断任务状态（pending → processing → done/failed）
  - result_cache:     图片级结果缓存（MD5 哈希 → JSON 结果）

被 app_fastapi.py 导入使用，数据库文件 diagnose_history.db 与本脚本同目录。
"""

import sqlite3
import json
import os
import uuid
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnose_history.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                total_objects INTEGER DEFAULT 0,
                boxes_json TEXT,
                probabilities_json TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                result_json TEXT,
                image_hash TEXT,
                created_at TEXT NOT NULL,
                finished_at TEXT
            )
        """)
        # 缓存表：图片MD5 → 诊断结果
        c.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                image_hash TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def insert_diagnosis(class_name, confidence, boxes, probabilities):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO history (created_at, class_name, confidence, total_objects, boxes_json, probabilities_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                class_name,
                confidence,
                len(boxes),
                json.dumps(boxes, ensure_ascii=False),
                json.dumps(probabilities, ensure_ascii=False),
            ),
        )
        return cur.lastrowid


def fetch_history(limit=20):
    with get_conn() as c:
        rows = c.execute(
            "SELECT id, created_at, class_name, confidence, total_objects, boxes_json, probabilities_json "
            "FROM history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "id": r["id"],
            "time": r["created_at"],
            "class": r["class_name"],
            "confidence": r["confidence"],
            "total_objects": r["total_objects"],
            "boxes": json.loads(r["boxes_json"] or "[]"),
            "probabilities": json.loads(r["probabilities_json"] or "{}"),
        }
        for r in rows
    ]


def clear_history():
    with get_conn() as c:
        c.execute("DELETE FROM history")


# ======================== 异步任务管理 ========================

def create_task(image_hash: str = None) -> str:
    """创建异步任务，返回 task_id"""
    task_id = uuid.uuid4().hex[:12]
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO tasks (task_id, status, image_hash, created_at) VALUES (?, 'pending', ?, ?)",
            (task_id, image_hash, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
    return task_id


def get_task(task_id: str) -> dict | None:
    """查询任务状态和结果"""
    with get_conn() as c:
        row = c.execute(
            "SELECT task_id, status, result_json, created_at, finished_at FROM tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "task_id": row["task_id"],
        "status": row["status"],
        "result": json.loads(row["result_json"]) if row["result_json"] else None,
        "created_at": row["created_at"],
        "finished_at": row["finished_at"],
    }


def update_task(task_id: str, status: str, result: dict = None):
    """更新任务状态和结果"""
    finished = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if status in ("done", "failed") else None
    with get_conn() as conn:
        conn.execute(
            "UPDATE tasks SET status = ?, result_json = ?, finished_at = ? WHERE task_id = ?",
            (status, json.dumps(result, ensure_ascii=False) if result else None, finished, task_id),
        )


# ======================== 图片哈希缓存 ========================

def cache_get(image_hash: str) -> dict | None:
    """从缓存中获取诊断结果"""
    with get_conn() as c:
        row = c.execute(
            "SELECT result_json FROM cache WHERE image_hash = ?", (image_hash,)
        ).fetchone()
    if row:
        return json.loads(row["result_json"])
    return None


def cache_set(image_hash: str, result: dict):
    """将诊断结果写入缓存"""
    with get_conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO cache (image_hash, result_json, created_at) VALUES (?, ?, ?)",
            (image_hash, json.dumps(result, ensure_ascii=False), datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )


def clear_cache():
    """清空结果缓存（启动时调用，防止旧模型结果的缓存污染新模型）"""
    with get_conn() as c:
        c.execute("DELETE FROM cache")


init_db()
