import unittest

from system_b.core.event_sync_status import EventSyncStatus
from system_b.core.event_mqtt import MqttEventTransport


class Stage38MqttRuntimeStatusTests(unittest.TestCase):
    def test_runtime_state_is_fixed_and_secret_free(self):
        status = EventSyncStatus()
        status.record_mqtt_runtime("connection_failed", "runtime")
        snapshot = status.snapshot(True, 30, "mqtt", False, 2)

        self.assertEqual(snapshot["mqtt_runtime_state"], "connection_failed")
        self.assertEqual(snapshot["mqtt_failure_type"], "runtime")
        self.assertNotIn("broker", str(snapshot))
        self.assertNotIn("password", str(snapshot))

    def test_mqtt_publish_exhaustion_has_fixed_failure_type(self):
        transport = MqttEventTransport(
            "agrivision/events",
            lambda *_args: False,
            max_attempts=2,
            backoff_seconds=0,
        )

        result = transport.sync([{"event_id": "stage38-1"}], lambda _event_id: None)

        self.assertEqual(result["sent"], 0)
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["failure_type"], "retry_exhausted")

    def test_app_records_runtime_and_manual_sync_outcomes(self):
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "system_b" / "core" / "app.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('record_mqtt_runtime("connecting")', source)
        self.assertIn('record_mqtt_runtime("connected")', source)
        self.assertIn('record_mqtt_runtime("connection_failed", "runtime")', source)
        self.assertIn('failure_type=result.get("failure_type", "")', source)
        self.assertIn('event_sync_status.record(', source)
        start = source.index("function formatEventSyncStatus")
        end = source.index("async function loadEventSyncStatus", start)
        formatter = source[start:end]
        self.assertIn("runtimeLabels", formatter)
        self.assertIn("data.mqtt_runtime_state", formatter)
        self.assertNotIn("mqtt_failure_type", formatter)


if __name__ == "__main__":
    unittest.main()
