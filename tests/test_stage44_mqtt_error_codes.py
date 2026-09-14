import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Stage44MqttErrorCodeTests(unittest.TestCase):
    def test_mqtt_route_uses_fixed_codes_without_exception_text(self):
        source = (ROOT / "system_b" / "core" / "app.py").read_text(encoding="utf-8")
        start = source.index("def api_offline_events_sync_mqtt")
        end = source.index("@app.route('/api/offline_events/sync_status')", start)
        route = source[start:end]

        self.assertIn('"mqtt_not_configured"', route)
        self.assertIn('"mqtt_connection_failed"', route)
        self.assertIn('"mqtt sync unavailable"', route)
        self.assertNotIn("str(error)", route)
        self.assertNotIn("logger.exception", route)


if __name__ == "__main__":
    unittest.main()
