import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system_b" / "core"))


class _ReasonCode:
    def __init__(self, is_failure):
        self.is_failure = is_failure


class Stage26MqttRetryLifecycleTests(unittest.TestCase):
    def test_failed_connack_stops_previous_loop_before_retry(self):
        from mqtt_runtime import create_paho_transport

        calls = []

        class Info:
            rc = 0

            def wait_for_publish(self, timeout):
                calls.append(("wait", timeout))

            def is_published(self):
                return True

        class FakeClient:
            is_connected = True
            connect_count = 0

            def __init__(self, **kwargs):
                self.on_connect = None
                self.reconnect_on_failure = True
                self.active_loops = 0

            def connect(self, host, port, keepalive):
                FakeClient.connect_count += 1
                calls.append(("connect", FakeClient.connect_count))

            def loop_start(self):
                self.active_loops += 1
                calls.append(("loop_start", self.active_loops))
                if self.active_loops > 1:
                    raise AssertionError("a second network loop started before the first stopped")
                self.on_connect(None, None, None, _ReasonCode(FakeClient.connect_count == 1), None)

            def loop_stop(self):
                calls.append(("loop_stop", self.active_loops))
                self.active_loops = 0

            def disconnect(self):
                calls.append(("disconnect",))

            def publish(self, topic, payload, qos, retain):
                return Info()

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
                "agrivision-retry-lifecycle",
                connect_attempts=2,
                connect_backoff_seconds=0,
                max_attempts=1,
                backoff_seconds=0,
            )
            try:
                result = transport.sync([{"event_id": "retry-lifecycle-1"}], lambda _: None)
            finally:
                close()

        self.assertEqual(result, {"sent": 1, "attempts": 1})
        self.assertEqual(calls[:5], [
            ("connect", 1),
            ("loop_start", 1),
            ("disconnect",),
            ("loop_stop", 1),
            ("connect", 2),
        ])
        self.assertEqual(calls[-2:], [("disconnect",), ("loop_stop", 1)])


if __name__ == "__main__":
    unittest.main()
