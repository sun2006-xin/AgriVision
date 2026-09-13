import pathlib
import sys
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system_b" / "core"))


class Stage9PahoRuntimeTests(unittest.TestCase):
    def test_client_id_is_bounded_and_safe(self):
        from mqtt_runtime import validate_client_id

        self.assertEqual(validate_client_id("agrivision-system-b"), "agrivision-system-b")
        for client_id in ("", "bad client", "a" * 65, "client/one"):
            with self.subTest(client_id=client_id):
                with self.assertRaises(ValueError):
                    validate_client_id(client_id)

    def test_missing_optional_paho_dependency_is_explicit(self):
        from mqtt_runtime import create_paho_transport

        with self.assertRaises(ValueError):
            create_paho_transport("mqtt://127.0.0.1:1883", "agrivision/events", "bad client")

    def test_fake_paho_client_covers_connect_publish_and_close(self):
        from mqtt_runtime import create_paho_transport

        calls = []

        class Info:
            rc = 0

            def wait_for_publish(self, timeout):
                calls.append(("wait", timeout))

            def is_published(self):
                return True

        class FakeClient:
            def __init__(self, **kwargs):
                calls.append(("init", kwargs))

            def connect(self, host, port, keepalive):
                calls.append(("connect", host, port, keepalive))

            def loop_start(self):
                calls.append(("loop_start",))

            def publish(self, topic, payload, qos, retain):
                calls.append(("publish", topic, payload, qos, retain))
                return Info()

            def disconnect(self):
                calls.append(("disconnect",))

            def loop_stop(self):
                calls.append(("loop_stop",))

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
                "agrivision-test",
                max_attempts=1,
                backoff_seconds=0,
            )
            acknowledged = []
            result = transport.sync([{"event_id": "runtime-1"}], acknowledged.append)
            close()

        self.assertEqual(result, {"sent": 1, "attempts": 1})
        self.assertEqual(acknowledged, ["runtime-1"])
        self.assertEqual(calls[1], ("connect", "127.0.0.1", 1883, 30))
        self.assertEqual(calls[2], ("loop_start",))
        self.assertEqual(calls[-2:], [("disconnect",), ("loop_stop",)])

    def test_initial_connection_retries_with_a_bounded_limit(self):
        from mqtt_runtime import create_paho_transport

        calls = []

        class FlakyClient:
            def __init__(self, **kwargs):
                pass

            def connect(self, host, port, keepalive):
                calls.append((host, port, keepalive))
                if len(calls) == 1:
                    raise OSError("temporary broker outage")

            def loop_start(self):
                calls.append("loop_start")

            def disconnect(self):
                calls.append("disconnect")

            def loop_stop(self):
                calls.append("loop_stop")

        fake_client_module = types.ModuleType("paho.mqtt.client")
        fake_client_module.Client = FlakyClient
        fake_client_module.CallbackAPIVersion = types.SimpleNamespace(VERSION2=2)
        fake_client_module.MQTTv5 = 5
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
            _, close = create_paho_transport(
                "mqtt://127.0.0.1:1883",
                "agrivision/events",
                "agrivision-retry",
                connect_attempts=2,
                connect_backoff_seconds=0,
            )
            close()

        self.assertEqual(calls[:2], [("127.0.0.1", 1883, 30), ("127.0.0.1", 1883, 30)])
        self.assertEqual(calls[-3:], ["loop_start", "disconnect", "loop_stop"])


if __name__ == "__main__":
    unittest.main()
