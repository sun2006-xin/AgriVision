import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class Stage11SystemBMqttTests(unittest.TestCase):
    def test_system_b_mqtt_sync_is_explicit_and_lazy(self):
        source = (ROOT / "system_b" / "core" / "app.py").read_text(encoding="utf-8")
        self.assertIn("@app.route('/api/offline_events/sync_mqtt', methods=['POST'])", source)
        self.assertIn("AGRIVISION_MQTT_BROKER_URL", source)
        self.assertIn("AGRIVISION_MQTT_TOPIC", source)
        self.assertIn("AGRIVISION_MQTT_CLIENT_ID", source)
        self.assertIn("AGRIVISION_MQTT_USERNAME", source)
        self.assertIn("AGRIVISION_MQTT_PASSWORD", source)
        self.assertIn("create_paho_transport", source)
        self.assertIn("event sink is not configured", source)
        self.assertIn("mqtt runtime is not configured", source)


if __name__ == "__main__":
    unittest.main()
