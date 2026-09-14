"""History service facade backed by the SQLite repository."""

import json
import os
import shutil
import sqlite3
import tempfile
import threading
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

from repositories.history_repository import HistoryRepository


HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_logs")
HISTORY_FILE = os.path.join(HISTORY_DIR, "history.json")
IMAGE_DIR = os.path.join(HISTORY_DIR, "images")


class HistoryManager:
    """Application-facing history service; normal reads never load JSON or all rows."""

    def __init__(self, history_dir=None, database_path=None, image_dir=None):
        self.history_dir = Path(history_dir or HISTORY_DIR)
        self.database_path = Path(database_path or self.history_dir / "history.db")
        self.image_dir = Path(image_dir or self.history_dir / "images")
        self.legacy_file = self.history_dir / "history.json"
        self._lock = threading.RLock()
        self._id_counter = 0
        self._last_id_second = ""
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.repository = HistoryRepository(self.database_path)
        self.repository.initialize()
        self._migrate_legacy_json()

    def _migrate_legacy_json(self):
        if self.repository.count() or not self.legacy_file.exists():
            return
        try:
            with self.legacy_file.open("r", encoding="utf-8") as stream:
                records = json.load(stream)
            if isinstance(records, list):
                sanitized = []
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    migrated = dict(record)
                    if migrated.get("image_path"):
                        migrated["image_path"] = self._safe_image_path(migrated["image_path"])
                    sanitized.append(migrated)
                self.repository.migrate(sanitized)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return

    def _safe_image_path(self, image_path):
        if not image_path:
            return None
        try:
            candidate = Path(image_path).resolve()
            image_root = self.image_dir.resolve()
        except (OSError, RuntimeError, TypeError):
            return None
        if candidate.parent != image_root:
            return None
        return str(candidate)

    def _public_record(self, record):
        public = dict(record)
        safe_path = self._safe_image_path(public.get("image_path"))
        public["image_path"] = Path(safe_path).name if safe_path else None
        return public

    def _public_records(self, records):
        return [self._public_record(record) for record in records]

    @property
    def records(self):
        """Compatibility escape hatch; normal API paths use indexed repository queries."""
        return self._public_records(self.repository.list_records())

    def _next_record_id(self):
        now = datetime.now()
        second = now.strftime("%Y%m%d_%H%M%S")
        if second == self._last_id_second:
            self._id_counter += 1
        else:
            self._last_id_second = second
            self._id_counter = 0
        return second if self._id_counter == 0 else f"{second}_{self._id_counter}"

    @staticmethod
    def _base_record(record_id, result, dual_result):
        now = datetime.now()
        record = {
            "id": record_id,
            "timestamp": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "level": result.get("level", "正常"),
            "level_code": result.get("level_code", 0),
            "disease_count": result.get("disease_count", 0),
            "disease_ratio": result.get("disease_ratio", 0.0),
            "white_count": result.get("white_count", 0),
            "white_ratio": result.get("white_ratio", 0.0),
            "green_ratio": result.get("green_ratio", 0.0),
            "image_path": None,
            "yolo_disease_count": 0,
            "yolo_bug_count": 0,
            "dual_confidence": "",
            "dual_agree_disease": False,
            "dual_agree_bug": False,
        }
        if dual_result is not None:
            yolo_result = dual_result.get("yolo_result", {})
            agreement = dual_result.get("agreement", {})
            record.update({
                "yolo_disease_count": yolo_result.get("disease_count", 0),
                "yolo_bug_count": yolo_result.get("bug_count", 0),
                "dual_confidence": dual_result.get("confidence", ""),
                "dual_agree_disease": agreement.get("disease", False),
                "dual_agree_bug": agreement.get("bug", False),
            })
        return record

    def add_record(self, result, image=None, dual_result=None):
        with self._lock:
            for _ in range(5):
                record_id = self._next_record_id()
                if self.repository.get_by_id(record_id) is not None:
                    continue
                record = self._base_record(record_id, result, dual_result)
                if int(record["level_code"]) >= 1 and image is not None:
                    image_path = self.image_dir / f"{record_id}.jpg"
                    try:
                        import cv2
                        if not cv2.imwrite(str(image_path), image):
                            raise OSError("image write failed")
                        record["image_path"] = str(image_path)
                    except Exception:
                        record["image_path"] = None
                try:
                    self.repository.insert(record)
                    self._cleanup_old_records(days=30)
                    return record_id
                except sqlite3.IntegrityError:
                    if record.get("image_path"):
                        try:
                            Path(record["image_path"]).unlink(missing_ok=True)
                        except OSError:
                            pass
                except Exception:
                    if record.get("image_path"):
                        try:
                            Path(record["image_path"]).unlink(missing_ok=True)
                        except OSError:
                            pass
                    raise
            raise RuntimeError("unable to allocate unique history record id")

    def _cleanup_old_records(self, days=30):
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        for record in self.repository.delete_older_than(cutoff):
            image_path = record.get("image_path")
            if not image_path:
                continue
            try:
                candidate = Path(image_path).resolve()
                image_root = self.image_dir.resolve()
                if candidate.parent == image_root:
                    candidate.unlink(missing_ok=True)
            except (OSError, RuntimeError):
                continue

    def query_records(self, date_from=None, date_to=None, level=None, page=1, page_size=20):
        result = self.repository.query(date_from, date_to, level or None, page, page_size)
        result["records"] = self._public_records(result["records"])
        return result

    def get_trend_data(self, days=7):
        return self.repository.trend(days)

    def get_statistics(self, date_from=None, date_to=None):
        return self.repository.statistics(date_from, date_to)

    def export_data(self, date_from=None, date_to=None):
        filtered = self.repository.list_records(date_from, date_to)
        public_records = self._public_records(filtered)
        export_dir = Path(tempfile.mkdtemp(prefix="history-export-", dir=self.history_dir))
        try:
            (export_dir / "images").mkdir()
            (export_dir / "detection_data.json").write_text(
                json.dumps(public_records, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (export_dir / "statistics.json").write_text(
                json.dumps(self.repository.statistics(date_from, date_to), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            for record in filtered:
                image_path = self._safe_image_path(record.get("image_path"))
                if image_path and Path(image_path).is_file():
                    shutil.copy2(image_path, export_dir / "images")
            zip_path = self.history_dir / f"detection_export_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in export_dir.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(export_dir))
            return str(zip_path)
        finally:
            shutil.rmtree(export_dir, ignore_errors=True)

    def get_record_by_id(self, record_id):
        record = self.repository.get_by_id(record_id)
        if record and record.get("image_path"):
            record["image_path"] = self._safe_image_path(record["image_path"])
        return record

    def get_all_levels(self):
        return list(HistoryRepository.LEVELS)

    def get_total_count(self):
        return self.repository.count()
