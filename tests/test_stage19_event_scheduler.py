import threading
import time
import unittest
from pathlib import Path


from system_b.core.event_scheduler import EventSyncScheduler


class Stage19EventSchedulerTests(unittest.TestCase):
    def test_system_b_scheduler_is_opt_in_and_stoppable(self):
        source = (Path(__file__).resolve().parents[1] / "system_b" / "core" / "app.py").read_text(encoding="utf-8")

        self.assertIn('AGRIVISION_EVENTS_SYNC_INTERVAL', source)
        self.assertIn('EVENTS_SYNC_INTERVAL = _read_nonnegative_float', source)
        self.assertIn('EventSyncScheduler(', source)
        self.assertIn('start_event_sync_scheduler()', source)
        self.assertIn('@atexit.register\ndef stop_event_sync_scheduler', source)

    def test_zero_interval_keeps_scheduler_disabled(self):
        calls = []
        scheduler = EventSyncScheduler(calls.append, interval_seconds=0)

        self.assertFalse(scheduler.start())
        self.assertFalse(scheduler.running)
        self.assertEqual(calls, [])

    def test_enabled_scheduler_runs_and_stops_cleanly(self):
        calls = []
        first_call = threading.Event()

        def sync_once():
            calls.append(time.monotonic())
            first_call.set()

        scheduler = EventSyncScheduler(sync_once, interval_seconds=0.01)
        self.assertTrue(scheduler.start())
        self.assertTrue(first_call.wait(1))
        self.assertTrue(scheduler.stop(timeout=1))
        call_count = len(calls)
        time.sleep(0.03)

        self.assertFalse(scheduler.running)
        self.assertEqual(len(calls), call_count)

    def test_callback_exception_does_not_kill_scheduler(self):
        calls = []

        def sync_once():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("transient")

        scheduler = EventSyncScheduler(sync_once, interval_seconds=0.01)
        scheduler.start()
        deadline = time.monotonic() + 1
        while len(calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        scheduler.stop(timeout=1)

        self.assertGreaterEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
