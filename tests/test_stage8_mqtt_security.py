import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system_b" / "core"))


class Stage8MqttSecurityTests(unittest.TestCase):
    def test_remote_broker_requires_mqtts(self):
        from mqtt_config import validate_broker_url

        self.assertEqual(validate_broker_url("mqtts://broker.example.org:8883"), "mqtts://broker.example.org:8883")
        with self.assertRaises(ValueError):
            validate_broker_url("mqtt://broker.example.org:1883")

    def test_local_broker_may_use_mqtt_without_embedded_credentials(self):
        from mqtt_config import validate_broker_url

        self.assertEqual(validate_broker_url("mqtt://127.0.0.1:1883"), "mqtt://127.0.0.1:1883")
        with self.assertRaises(ValueError):
            validate_broker_url("mqtts://user:password@broker.example.org:8883")
        with self.assertRaises(ValueError):
            validate_broker_url("mqtt://127.0.0.1:1883/events")


if __name__ == "__main__":
    unittest.main()
