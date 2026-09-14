"""Bounded atomic JSON cache for detection events while the network is offline."""

import json
import os
from pathlib import Path
import re
import tempfile


_SAFE_EVENT_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class OfflineEventCache:
    def __init__(self, directory, max_items=1000, max_bytes=50 * 1024 * 1024):
        if max_items <= 0 or max_bytes <= 0:
            raise ValueError("cache limits must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_items = max_items
        self.max_bytes = max_bytes

    def _path(self, event_id):
        if not isinstance(event_id, str) or not _SAFE_EVENT_ID.fullmatch(event_id):
            raise ValueError("event_id contains unsupported characters")
        return self.directory / f"{event_id}.json"

    def put(self, event):
        event_id = event.get("event_id") if isinstance(event, dict) else None
        target = self._path(event_id)
        serialized = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        if len(serialized.encode("utf-8")) > self.max_bytes:
            raise ValueError("event exceeds offline cache byte limit")
        fd, temp_name = tempfile.mkstemp(prefix="event.", suffix=".tmp", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        finally:
            if os.path.exists(temp_name):
                os.remove(temp_name)
        self._trim()

    def list_pending(self):
        pending = []
        for path in sorted(self.directory.glob("*.json"), key=lambda item: (item.stat().st_mtime_ns, item.name)):
            try:
                pending.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return pending

    def ack(self, event_id):
        path = self._path(event_id)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _trim(self):
        paths = sorted(self.directory.glob("*.json"), key=lambda item: (item.stat().st_mtime_ns, item.name))
        total_bytes = sum(path.stat().st_size for path in paths)
        while paths and (len(paths) > self.max_items or total_bytes > self.max_bytes):
            oldest = paths.pop(0)
            total_bytes -= oldest.stat().st_size
            try:
                oldest.unlink()
            except FileNotFoundError:
                pass
