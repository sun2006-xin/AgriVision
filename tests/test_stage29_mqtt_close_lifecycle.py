import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system_b" / "core"))


class Stage29MqttCloseLifecycleTests(unittest.TestCase):
    def test_close_is_idempotent_and_stops_loop_when_disconnect_raises(self):
        from mqtt_runtime import create_paho_transport

        calls = []

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def connect(self, host, port, keepalive):
                pass

            def loop_start(self):
                calls.append("loop_start")

            def disconnect(self):
                calls.append("disconnect")
                raise OSError("already disconnected")

            def loop_stop(self):
                calls.append("loop_stop")

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
            _transport, close = create_paho_transport(
                "mqtt://127.0.0.1:1883",
                "agrivision/events",
                "agrivision-close-test",
            )
            close()
            close()

        self.assertEqual(calls, ["loop_start", "disconnect", "loop_stop"])


if __name__ == "__main__":
    unittest.main()
