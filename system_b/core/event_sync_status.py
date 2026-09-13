"""Thread-safe, privacy-safe operational state for offline event sync."""

from datetime import datetime, timezone
import threading


class EventSyncStatus:
    """Keep bounded counters and timestamps without retaining transport secrets."""

    def __init__(self):
        self._lock = threading.Lock()
        self._batches_succeeded = 0
        self._batches_failed = 0
        self._events_sent = 0
        self._events_acked = 0
        self._last_outcome = "never"
        self._last_transport = "none"
        self._last_error = ""
        self._last_attempt_at = ""
        self._last_success_at = ""

    def record(self, outcome, transport, pending, sent=0, acked=0, error=""):
        if outcome not in {"success", "failure", "empty", "skipped"}:
            raise ValueError("unsupported sync outcome")
        if transport not in {"mqtt", "http", "none"}:
            raise ValueError("unsupported sync transport")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._last_outcome = outcome
            self._last_transport = transport
            self._last_attempt_at = now
            self._last_error = "sync failed" if outcome == "failure" else ""
            self._events_sent += max(0, int(sent))
            self._events_acked += max(0, int(acked))
            if outcome == "success":
                self._batches_succeeded += 1
                self._last_success_at = now
            elif outcome == "failure":
                self._batches_failed += 1

    def snapshot(self, enabled, interval_seconds, transport, running, pending):
        with self._lock:
            return {
                "enabled": bool(enabled),
                "interval_seconds": interval_seconds,
                "transport": transport,
                "running": bool(running),
                "pending": max(0, int(pending)),
                "batches_succeeded": self._batches_succeeded,
                "batches_failed": self._batches_failed,
                "events_sent": self._events_sent,
                "events_acked": self._events_acked,
                "last_outcome": self._last_outcome,
                "last_transport": self._last_transport,
                "last_error": self._last_error,
                "last_attempt_at": self._last_attempt_at,
                "last_success_at": self._last_success_at,
            }
