import unittest
from pathlib import Path


from system_b.core.event_sync_status import EventSyncStatus


class Stage20SyncObservabilityTests(unittest.TestCase):
    def test_system_b_exposes_safe_sync_status_endpoint(self):
        source = (Path(__file__).resolve().parents[1] / "system_b" / "core" / "app.py").read_text(encoding="utf-8")

        self.assertIn("@app.route('/api/offline_events/sync_status')", source)
        self.assertIn("event_sync_status.snapshot", source)
        self.assertIn('failure_type="runtime"', source)
        self.assertNotIn('"broker_url"', source)

    def test_snapshot_has_safe_initial_state(self):
        status = EventSyncStatus()

        snapshot = status.snapshot(enabled=False, interval_seconds=0, transport="none", running=False, pending=2)

        self.assertEqual(snapshot["last_outcome"], "never")
        self.assertEqual(snapshot["pending"], 2)
        self.assertFalse(snapshot["enabled"])
        self.assertNotIn("url", snapshot)
        self.assertNotIn("password", snapshot)

    def test_record_exposes_bounded_operational_counters(self):
        status = EventSyncStatus()

        status.record("success", "mqtt", pending=3, sent=4, acked=4)
        snapshot = status.snapshot(enabled=True, interval_seconds=30, transport="mqtt", running=True, pending=3)

        self.assertEqual(snapshot["last_outcome"], "success")
        self.assertEqual(snapshot["batches_succeeded"], 1)
        self.assertEqual(snapshot["events_sent"], 4)
        self.assertEqual(snapshot["events_acked"], 4)
        self.assertTrue(snapshot["last_attempt_at"])

    def test_failure_is_visible_without_exposing_exception_text(self):
        status = EventSyncStatus()

        status.record("failure", "http", pending=5, error="sink credentials=secret")
        snapshot = status.snapshot(enabled=True, interval_seconds=10, transport="http", running=True, pending=5)

        self.assertEqual(snapshot["last_outcome"], "failure")
        self.assertEqual(snapshot["batches_failed"], 1)
        self.assertEqual(snapshot["last_error"], "sync failed")
        self.assertNotIn("secret", str(snapshot))


if __name__ == "__main__":
    unittest.main()
