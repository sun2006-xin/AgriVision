"""Small, stoppable scheduler for opt-in offline event synchronization."""

import logging
import threading


class EventSyncScheduler:
    """Run a supplied sync callback at a bounded interval in a daemon thread."""

    def __init__(self, sync_once, interval_seconds, logger=None):
        if not callable(sync_once):
            raise TypeError("sync_once must be callable")
        if isinstance(interval_seconds, bool) or not isinstance(interval_seconds, (int, float)):
            raise ValueError("interval_seconds must be a number")
        if interval_seconds < 0:
            raise ValueError("interval_seconds must not be negative")
        self._sync_once = sync_once
        self._interval_seconds = interval_seconds
        self._logger = logger or logging.getLogger(__name__)
        self._stop_event = threading.Event()
        self._thread = None
        self._state_lock = threading.Lock()

    @property
    def running(self):
        with self._state_lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self):
        """Start once; return False when disabled or already running."""
        if self._interval_seconds == 0:
            return False
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True

    def stop(self, timeout=None):
        """Request stop and return whether the worker has exited."""
        self._stop_event.set()
        with self._state_lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        return not self.running

    def _run(self):
        while not self._stop_event.wait(self._interval_seconds):
            try:
                self._sync_once()
            except Exception:
                self._logger.exception("offline event scheduler callback failed")
