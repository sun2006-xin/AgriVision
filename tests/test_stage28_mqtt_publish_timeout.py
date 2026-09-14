import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system_b" / "core"))


class Stage28MqttPublishTimeoutTests(unittest.TestCase):
    def test_system_b_exposes_bounded_publish_timeout_configuration(self):
        source = (ROOT / "system_b" / "core" / "app.py").read_text(encoding="utf-8")
        self.assertIn("AGRIVISION_MQTT_PUBLISH_TIMEOUT", source)
        self.assertIn("MQTT_PUBLISH_TIMEOUT", source)
        self.assertIn("publish_timeout=MQTT_PUBLISH_TIMEOUT", source)

    def test_publish_timeout_is_passed_to_paho_and_is_bounded(self):
        from mqtt_runtime import create_paho_transport

        waits = []

        class Info:
            rc = 0

            def wait_for_publish(self, timeout):
                waits.append(timeout)

            def is_published(self):
                return True

        class FakeClient:
            is_connected = True

            def __init__(self, **kwargs):
                self.on_connect = None
                self.reconnect_on_failure = True

            def connect(self, host, port, keepalive):
                pass

            def loop_start(self):
                self.on_connect(None, None, None, types.SimpleNamespace(is_failure=False), None)

            def publish(self, topic, payload, qos, retain):
                return Info()

            def disconnect(self):
                pass

            def loop_stop(self):
                pass

        fake_client_module = types.ModuleType("paho.mqtt.client")
        fake_client_module.Client = FakeClient
        fake_client_module.CallbackAPIVersion = types.SimpleNamespace(VERSION2=2)
        fake_client_module.MQTTv5 = 5
        fake_client_module.MQTT_ERR_SUCCESS = 0
        fake_mqtt_module = types.ModuleType("paho.mqtt")
        fake_paho_module = types.ModuleType("paho")
        fake_paho_module.mqtt = fake_mqtt_module
        fake_mqtt_module.client = fake_client_module

        with mock.patch.dict(
            sys.modules,
            {
                "paho": fake_paho_module,
                "paho.mqtt": fake_mqtt_module,
                "paho.mqtt.client": fake_client_module,
            },
        ):
            transport, close = create_paho_transport(
                "mqtt://127.0.0.1:1883",
                "agrivision/events",
                "agrivision-publish-timeout-test",
                publish_timeout=1.25,
            )
            try:
                result = transport.sync([{"event_id": "publish-timeout-1"}], lambda _: None)
            finally:
                close()

        self.assertEqual(result, {"sent": 1, "attempts": 1})
        self.assertEqual(waits, [1.25])

    def test_publish_timeout_rejects_unbounded_values(self):
        from mqtt_runtime import create_paho_transport

        for timeout in (0, -1, 60.01, "1"):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    create_paho_transport(
                        "mqtt://127.0.0.1:1883",
                        "agrivision/events",
                        "agrivision-publish-timeout-validation",
                        publish_timeout=timeout,
                    )


if __name__ == "__main__":
    unittest.main()
