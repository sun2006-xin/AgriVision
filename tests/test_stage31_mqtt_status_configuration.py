import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Stage31MqttStatusConfigurationTests(unittest.TestCase):
    def test_status_endpoint_passes_safe_mqtt_timeout_configuration(self):
        source = (ROOT / "system_b" / "core" / "app.py").read_text(encoding="utf-8")
        self.assertIn("mqtt_connack_timeout=MQTT_CONNACK_TIMEOUT", source)
        self.assertIn("mqtt_publish_timeout=MQTT_PUBLISH_TIMEOUT", source)

    def test_snapshot_contains_timeout_numbers_but_no_transport_secrets(self):
        from system_b.core.event_sync_status import EventSyncStatus

        snapshot = EventSyncStatus().snapshot(
            enabled=True,
            interval_seconds=30,
            transport="mqtt",
            running=True,
            pending=1,
            mqtt_connack_timeout=5.0,
            mqtt_publish_timeout=10.0,
        )

        self.assertEqual(snapshot["mqtt_connack_timeout_seconds"], 5.0)
        self.assertEqual(snapshot["mqtt_publish_timeout_seconds"], 10.0)
        self.assertNotIn("broker_url", snapshot)
        self.assertNotIn("username", snapshot)
        self.assertNotIn("password", snapshot)


if __name__ == "__main__":
    unittest.main()
