import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system_b" / "core"))


class Stage30MqttPublishRuntimeErrorTests(unittest.TestCase):
    def test_runtime_error_from_publisher_is_bounded_and_does_not_ack(self):
        from event_mqtt import MqttEventTransport

        calls = []
        acknowledged = []

        def publisher(topic, payload, qos, retain):
            calls.append((topic, qos, retain))
            raise RuntimeError("paho publish state is unavailable")

        transport = MqttEventTransport(
            "agrivision/events",
            publisher,
            max_attempts=2,
            backoff_seconds=0,
        )
        result = transport.sync(
            [{"event_id": "runtime-error-1", "schema_version": 1}],
            acknowledged.append,
        )

        self.assertEqual(result, {"sent": 0, "attempts": 2})
        self.assertEqual(len(calls), 2)
        self.assertEqual(acknowledged, [])


if __name__ == "__main__":
    unittest.main()
