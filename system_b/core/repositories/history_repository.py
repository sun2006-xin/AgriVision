"""SQLite repository for System B detection history."""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager


class HistoryRepository:
    """Persist detection rows with indexed, parameterized queries."""

    LEVELS = ("正常", "注意", "警告", "严重")

    def __init__(self, database_path):
        self.database_path = str(database_path)
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self):
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS detection_records (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    level TEXT NOT NULL,
                    level_code INTEGER NOT NULL,
                    disease_count INTEGER NOT NULL,
                    disease_ratio REAL NOT NULL,
                    white_count INTEGER NOT NULL,
                    white_ratio REAL NOT NULL,
                    green_ratio REAL NOT NULL,
                    image_path TEXT,
                    yolo_disease_count INTEGER NOT NULL DEFAULT 0,
                    yolo_bug_count INTEGER NOT NULL DEFAULT 0,
                    dual_confidence TEXT NOT NULL DEFAULT '',
                    dual_agree_disease INTEGER NOT NULL DEFAULT 0,
                    dual_agree_bug INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_detection_date ON detection_records(date)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_detection_timestamp ON detection_records(timestamp DESC)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_detection_level_date ON detection_records(level, date)")
            connection.execute("PRAGMA user_version = 1")

    @classmethod
    def _row_to_record(cls, row):
        if row is None:
            return None
        record = dict(row)
        record["dual_agree_disease"] = bool(record["dual_agree_disease"])
        record["dual_agree_bug"] = bool(record["dual_agree_bug"])
        return record

    @staticmethod
    def _record_values(record):
        return (
            record["id"],
            record.get("timestamp", ""),
            record.get("date", ""),
            record.get("time", ""),
            record.get("level", "正常"),
            int(record.get("level_code", 0)),
            int(record.get("disease_count", 0)),
            float(record.get("disease_ratio", 0.0)),
            int(record.get("white_count", 0)),
            float(record.get("white_ratio", 0.0)),
            float(record.get("green_ratio", 0.0)),
            record.get("image_path"),
            int(record.get("yolo_disease_count", 0)),
            int(record.get("yolo_bug_count", 0)),
            str(record.get("dual_confidence", "")),
            int(bool(record.get("dual_agree_disease", False))),
            int(bool(record.get("dual_agree_bug", False))),
        )

    def insert(self, record):
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO detection_records (
                    id, timestamp, date, time, level, level_code,
                    disease_count, disease_ratio, white_count, white_ratio,
                    green_ratio, image_path, yolo_disease_count, yolo_bug_count,
                    dual_confidence, dual_agree_disease, dual_agree_bug
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._record_values(record),
            )

    def migrate(self, records):
        """Insert legacy JSON records once, ignoring already migrated IDs."""
        inserted = 0
        with self._connection() as connection:
            for record in records:
                if not isinstance(record, dict) or not record.get("id"):
                    continue
                try:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO detection_records (
                            id, timestamp, date, time, level, level_code,
                            disease_count, disease_ratio, white_count, white_ratio,
                            green_ratio, image_path, yolo_disease_count, yolo_bug_count,
                            dual_confidence, dual_agree_disease, dual_agree_bug
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        self._record_values(record),
                    )
                    inserted += connection.execute("SELECT changes()").fetchone()[0]
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue
        return inserted

    @staticmethod
    def _where(date_from=None, date_to=None, level=None):
        clauses = ["1 = 1"]
        values = []
        if date_from:
            clauses.append("date >= ?")
            values.append(date_from)
        if date_to:
            clauses.append("date <= ?")
            values.append(date_to)
        if level:
            clauses.append("level = ?")
            values.append(level)
        return " AND ".join(clauses), values

    def query(self, date_from=None, date_to=None, level=None, page=1, page_size=20):
        page = max(1, int(page))
        page_size = min(100, max(1, int(page_size)))
        where, values = self._where(date_from, date_to, level)
        offset = (page - 1) * page_size
        with self._connection() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM detection_records WHERE {where}", values
            ).fetchone()[0]
            rows = connection.execute(
                f"SELECT * FROM detection_records WHERE {where} ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?",
                [*values, page_size, offset],
            ).fetchall()
        return {
            "records": [self._row_to_record(row) for row in rows],
            "total": total,
            "pages": (total + page_size - 1) // page_size,
            "current_page": page,
            "page_size": page_size,
        }

    def list_records(self, date_from=None, date_to=None, level=None):
        where, values = self._where(date_from, date_to, level)
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM detection_records WHERE {where} ORDER BY timestamp DESC, id DESC",
                values,
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_by_id(self, record_id):
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM detection_records WHERE id = ?", (record_id,)
            ).fetchone()
        return self._row_to_record(row)

    def count(self):
        with self._connection() as connection:
            return connection.execute("SELECT COUNT(*) FROM detection_records").fetchone()[0]

    def statistics(self, date_from=None, date_to=None):
        where, values = self._where(date_from, date_to)
        with self._connection() as connection:
            aggregate = connection.execute(
                f"""
                SELECT COUNT(*) AS total_records,
                       COALESCE(SUM(disease_count), 0) AS total_disease,
                       COALESCE(SUM(white_count), 0) AS total_pest,
                       COALESCE(AVG(disease_ratio), 0.0) AS avg_disease_ratio,
                       COALESCE(AVG(white_ratio), 0.0) AS avg_pest_ratio,
                       COALESCE(AVG(green_ratio), 0.0) AS avg_green_ratio
                FROM detection_records WHERE {where}
                """,
                values,
            ).fetchone()
            levels = connection.execute(
                f"SELECT level, COUNT(*) AS count FROM detection_records WHERE {where} GROUP BY level",
                values,
            ).fetchall()
        level_counts = {level: 0 for level in self.LEVELS}
        for row in levels:
            if row["level"] in level_counts:
                level_counts[row["level"]] = row["count"]
        return {
            "total_records": aggregate["total_records"],
            "level_counts": level_counts,
            "total_disease": aggregate["total_disease"],
            "total_pest": aggregate["total_pest"],
            "avg_disease_ratio": aggregate["avg_disease_ratio"],
            "avg_pest_ratio": aggregate["avg_pest_ratio"],
            "avg_green_ratio": aggregate["avg_green_ratio"],
        }

    def trend(self, days=7):
        days = min(366, max(1, int(days)))
        end = datetime.now().date()
        start = end - timedelta(days=days - 1)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT date, COALESCE(SUM(disease_count), 0) AS disease_count,
                       COALESCE(SUM(white_count), 0) AS pest_count
                FROM detection_records WHERE date >= ? AND date <= ?
                GROUP BY date
                """,
                (start.isoformat(), end.isoformat()),
            ).fetchall()
            level_rows = connection.execute(
                """
                SELECT date, level, COUNT(*) AS count FROM detection_records
                WHERE date >= ? AND date <= ? GROUP BY date, level
                """,
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        totals = {row["date"]: row for row in rows}
        levels = {}
        for row in level_rows:
            levels.setdefault(row["date"], {level: 0 for level in self.LEVELS})
            if row["level"] in levels[row["date"]]:
                levels[row["date"]][row["level"]] = row["count"]
        dates = [(start + timedelta(days=index)).isoformat() for index in range(days)]
        return {
            "dates": dates,
            "disease_counts": [totals.get(date, {"disease_count": 0})["disease_count"] for date in dates],
            "pest_counts": [totals.get(date, {"pest_count": 0})["pest_count"] for date in dates],
            "level_stats": [levels.get(date, {level: 0 for level in self.LEVELS}) for date in dates],
        }

    def delete_older_than(self, cutoff_date):
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id, image_path FROM detection_records WHERE date < ?", (cutoff_date,)
            ).fetchall()
            connection.execute("DELETE FROM detection_records WHERE date < ?", (cutoff_date,))
        return [dict(row) for row in rows]
