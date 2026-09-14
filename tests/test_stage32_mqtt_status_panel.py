import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Stage32MqttStatusPanelTests(unittest.TestCase):
    def test_monitor_page_contains_safe_event_sync_panel(self):
        source = (ROOT / "system_b" / "core" / "templates" / "main.html").read_text(encoding="utf-8")
        self.assertIn('id="eventSyncInfo"', source)
        self.assertIn("/api/offline_events/sync_status", source)
        self.assertIn("mqtt_connack_timeout_seconds", source)
        self.assertIn("mqtt_publish_timeout_seconds", source)

    def test_panel_formats_only_bounded_status_values(self):
        source = (ROOT / "system_b" / "core" / "templates" / "main.html").read_text(encoding="utf-8")
        start = source.index("function formatEventSyncStatus")
        end = source.index("async function loadEventSyncStatus", start)
        formatter = source[start:end]
        self.assertIn("batches_succeeded", formatter)
        self.assertIn("batches_failed", formatter)
        self.assertNotIn("MQTT_PASSWORD", formatter)
        self.assertNotIn("MQTT_USERNAME", formatter)


if __name__ == "__main__":
    unittest.main()
