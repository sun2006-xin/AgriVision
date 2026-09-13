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
        self._last_failure_type = ""
        self._last_attempt_at = ""
        self._last_success_at = ""

    def record(self, outcome, transport, pending, sent=0, acked=0, error="", failure_type=""):
        if outcome not in {"success", "failure", "empty", "skipped"}:
            raise ValueError("unsupported sync outcome")
        if transport not in {"mqtt", "http", "none"}:
            raise ValueError("unsupported sync transport")
        if failure_type not in {"", "permanent", "retry_exhausted", "runtime"}:
            raise ValueError("unsupported sync failure type")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._last_outcome = outcome
            self._last_transport = transport
            self._last_attempt_at = now
            self._last_error = "sync failed" if outcome == "failure" else ""
            self._last_failure_type = failure_type if outcome == "failure" else ""
            self._events_sent += max(0, int(sent))
            self._events_acked += max(0, int(acked))
            if outcome == "success":
                self._batches_succeeded += 1
                self._last_success_at = now
            elif outcome == "failure":
                self._batches_failed += 1

    def snapshot(
        self,
        enabled,
        interval_seconds,
        transport,
        running,
        pending,
        mqtt_connack_timeout=None,
        mqtt_publish_timeout=None,
    ):
        with self._lock:
            return {
                "enabled": bool(enabled),
                "interval_seconds": interval_seconds,
                "transport": transport,
                "running": bool(running),
                "pending": max(0, int(pending)),
                "mqtt_connack_timeout_seconds": mqtt_connack_timeout,
                "mqtt_publish_timeout_seconds": mqtt_publish_timeout,
                "batches_succeeded": self._batches_succeeded,
                "batches_failed": self._batches_failed,
                "events_sent": self._events_sent,
                "events_acked": self._events_acked,
                "last_outcome": self._last_outcome,
                "last_transport": self._last_transport,
                "last_error": self._last_error,
                "last_failure_type": self._last_failure_type,
                "last_attempt_at": self._last_attempt_at,
                "last_success_at": self._last_success_at,
            }
