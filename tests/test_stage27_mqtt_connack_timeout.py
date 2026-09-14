import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system_b" / "core"))


class _SuccessReasonCode:
    is_failure = False


class Stage27MqttConnackTimeoutTests(unittest.TestCase):
    def test_system_b_exposes_bounded_connack_timeout_configuration(self):
        source = (ROOT / "system_b" / "core" / "app.py").read_text(encoding="utf-8")
        self.assertIn("AGRIVISION_MQTT_CONNACK_TIMEOUT", source)
        self.assertIn("MQTT_CONNACK_TIMEOUT", source)
        self.assertIn("connack_timeout=MQTT_CONNACK_TIMEOUT", source)

    def test_connack_timeout_is_bounded_and_passed_to_wait(self):
        from mqtt_runtime import create_paho_transport

        waits = []

        class Event:
            def clear(self):
                pass

            def set(self):
                pass

            def wait(self, timeout):
                waits.append(timeout)
                return True

        class FakeClient:
            is_connected = True

            def __init__(self, **kwargs):
                self.on_connect = None
                self.reconnect_on_failure = True

            def connect(self, host, port, keepalive):
                pass

            def loop_start(self):
                self.on_connect(None, None, None, _SuccessReasonCode(), None)

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
        ), mock.patch("mqtt_runtime.threading.Event", Event):
            _transport, close = create_paho_transport(
                "mqtt://127.0.0.1:1883",
                "agrivision/events",
                "agrivision-timeout-test",
                connack_timeout=1.25,
            )
            close()

        self.assertEqual(waits, [1.25])

    def test_connack_timeout_rejects_unbounded_values(self):
        from mqtt_runtime import create_paho_transport

        for timeout in (0, -1, 30.01, "1"):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    create_paho_transport(
                        "mqtt://127.0.0.1:1883",
                        "agrivision/events",
                        "agrivision-timeout-validation",
                        connack_timeout=timeout,
                    )


if __name__ == "__main__":
    unittest.main()
