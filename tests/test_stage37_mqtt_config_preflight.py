import unittest
from pathlib import Path

from system_b.core.mqtt_config_status import build_mqtt_config_status


ROOT = Path(__file__).resolve().parents[1]


class Stage37MqttConfigPreflightTests(unittest.TestCase):
    def test_empty_configuration_returns_fixed_missing_codes_only(self):
        status = build_mqtt_config_status("", "", "", None, None, None)
        self.assertFalse(status["configured"])
        self.assertIn("broker_url_missing", status["issues"])
        self.assertIn("topic_missing", status["issues"])
        self.assertNotIn("broker_url", status)
        self.assertNotIn("username", status)
        self.assertNotIn("password", status)
        self.assertNotIn("ca_certs", status)

    def test_local_development_broker_can_be_ready_without_tls(self):
        status = build_mqtt_config_status(
            "mqtt://127.0.0.1:1883", "agrivision/events", "system-b", None, None, None
        )
        self.assertTrue(status["configured"])
        self.assertEqual(status["scheme"], "mqtt")
        self.assertFalse(status["tls"])
        self.assertFalse(status["credentials_configured"])

    def test_remote_broker_requires_tls_but_never_returns_secrets(self):
        status = build_mqtt_config_status(
            "mqtt://broker.example.com:1883",
            "agrivision/events",
            "system-b",
            "user",
            "secret",
            None,
        )
        self.assertFalse(status["configured"])
        self.assertIn("remote_tls_required", status["issues"])
        self.assertTrue(status["credentials_configured"])
        self.assertNotIn("broker.example.com", str(status))
        self.assertNotIn("user", str(status))
        self.assertNotIn("secret", str(status))

    def test_sync_status_wires_preflight_without_raw_configuration(self):
        source = (ROOT / "system_b" / "core" / "routes" / "events.py").read_text(encoding="utf-8")
        frontend = (ROOT / "system_b" / "core" / "templates" / "main.html").read_text(encoding="utf-8")
        source += "\n" + frontend
        self.assertIn("build_mqtt_config_status", source)
        self.assertIn("mqtt_config=build_mqtt_config_status", source)
        self.assertIn("mqttConfig.configured", source)
        self.assertIn("mqttConfig.tls", source)
        self.assertNotIn('"broker_url": MQTT_BROKER_URL', source)
        self.assertNotIn('"password": MQTT_PASSWORD', source)


if __name__ == "__main__":
    unittest.main()
