"""Non-blocking cross-process lock backed by SQLite's atomic transaction lock."""

import sqlite3
import threading
from pathlib import Path


class ProcessSyncLock:
    """A small lock with the same acquire/release shape as threading.Lock."""

    def __init__(self, database_path):
        self.database_path = str(database_path)
        self._state_lock = threading.Lock()
        self._connection = None
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.close()

    def acquire(self, blocking=True):
        if blocking:
            raise ValueError("ProcessSyncLock only supports non-blocking acquire")
        with self._state_lock:
            if self._connection is not None:
                return False
            connection = sqlite3.connect(self.database_path, timeout=0, isolation_level=None)
            try:
                connection.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError:
                connection.close()
                return False
            self._connection = connection
            return True

    def release(self):
        with self._state_lock:
            connection = self._connection
            self._connection = None
        if connection is not None:
            connection.rollback()
            connection.close()

    def __enter__(self):
        if not self.acquire(blocking=False):
            raise RuntimeError("process sync lock is already held")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
