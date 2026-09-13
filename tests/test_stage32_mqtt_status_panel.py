import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Stage32MqttStatusPanelTests(unittest.TestCase):
    def test_monitor_page_contains_safe_event_sync_panel(self):
        source = (ROOT / "system_b" / "core" / "app.py").read_text(encoding="utf-8")
        self.assertIn('id="eventSyncInfo"', source)
        self.assertIn("/api/offline_events/sync_status", source)
        self.assertIn("mqtt_connack_timeout_seconds", source)
        self.assertIn("mqtt_publish_timeout_seconds", source)

    def test_panel_formats_only_bounded_status_values(self):
        source = (ROOT / "system_b" / "core" / "app.py").read_text(encoding="utf-8")
        self.assertIn("formatEventSyncStatus", source)
        self.assertIn("batches_succeeded", source)
        self.assertIn("batches_failed", source)
        self.assertNotIn("MQTT_PASSWORD", source[source.find("function formatEventSyncStatus"):])
        self.assertNotIn("MQTT_USERNAME", source[source.find("function formatEventSyncStatus"):])


if __name__ == "__main__":
    unittest.main()
