import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Stage40MqttRequestValidationTests(unittest.TestCase):
    def test_mqtt_route_validates_request_before_opening_transport(self):
        source = (ROOT / "system_b" / "core" / "routes" / "events.py").read_text(encoding="utf-8")
        start = source.index("def api_offline_events_sync_mqtt")
        end = source.index('@blueprint.get("/api/offline_events/sync_status")', start)
        route = source[start:end]
        helper_start = source.index("def _validate_sync_body")
        helper = source[helper_start:start]

        self.assertIn("payload = request.get_json(silent=True)", helper)
        self.assertLess(route.index("params = _validate_sync_body()"), route.index("transport = get_mqtt_transport"))
        self.assertIn('"request body must be a JSON object"', helper)
        self.assertIn('"limit must be an integer from 1 to 100"', helper)


if __name__ == "__main__":
    unittest.main()
