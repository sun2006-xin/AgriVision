import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Stage40MqttRequestValidationTests(unittest.TestCase):
    def test_mqtt_route_validates_request_before_opening_transport(self):
        source = (ROOT / "system_b" / "core" / "app.py").read_text(encoding="utf-8")
        start = source.index("def api_offline_events_sync_mqtt")
        end = source.index("@app.route('/api/offline_events/sync_status')", start)
        route = source[start:end]

        self.assertLess(route.index("params = request.get_json"), route.index("transport = get_mqtt_transport"))
        self.assertLess(route.index("limit = params.get"), route.index("transport = get_mqtt_transport"))
        self.assertIn('"request body must be a JSON object"', route)
        self.assertIn('"limit must be an integer from 1 to 100"', route)


if __name__ == "__main__":
    unittest.main()
