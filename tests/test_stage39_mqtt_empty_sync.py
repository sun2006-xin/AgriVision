import unittest
from pathlib import Path

from system_b.core.event_sync_status import EventSyncStatus


ROOT = Path(__file__).resolve().parents[1]


class Stage39MqttEmptySyncTests(unittest.TestCase):
    def test_empty_outcome_does_not_increment_success_counter(self):
        status = EventSyncStatus()
        status.record("empty", "mqtt", pending=0)

        snapshot = status.snapshot(True, 30, "mqtt", False, 0)

        self.assertEqual(snapshot["last_outcome"], "empty")
        self.assertEqual(snapshot["batches_succeeded"], 0)
        self.assertEqual(snapshot["batches_failed"], 0)

    def test_manual_route_handles_empty_queue_before_publishing(self):
        source = (ROOT / "system_b" / "core" / "routes" / "events.py").read_text(encoding="utf-8")
        start = source.index("def api_offline_events_sync_mqtt")
        end = source.index('@blueprint.get("/api/offline_events/sync_status")', start)
        route = source[start:end]

        self.assertIn("if not events:", route)
        self.assertIn('event_sync_status.record("empty", "mqtt", 0)', route)
        self.assertIn('"attempts": 0', route)
        self.assertLess(route.index("if not events:"), route.index("result = transport.sync"))


if __name__ == "__main__":
    unittest.main()
