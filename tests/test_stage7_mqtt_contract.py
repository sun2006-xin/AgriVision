import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system_b" / "core"))


class Stage7MqttContractTests(unittest.TestCase):
    def test_topic_rejects_wildcards_and_control_characters(self):
        from event_mqtt import validate_mqtt_topic

        self.assertEqual(validate_mqtt_topic("agrivision/events"), "agrivision/events")
        for topic in ("", "agrivision/#", "agrivision/+", "agrivision/\n"):
            with self.subTest(topic=topic):
                with self.assertRaises(ValueError):
                    validate_mqtt_topic(topic)

    def test_mqtt_batch_retries_and_acknowledges_after_publish(self):
        from event_mqtt import MqttEventTransport

        calls = []
        acknowledged = []

        def publisher(topic, payload, qos, retain):
            calls.append((topic, payload, qos, retain))
            if len(calls) < 2:
                return False
            return True

        transport = MqttEventTransport(
            "agrivision/events",
            publisher,
            max_attempts=2,
            backoff_seconds=0,
        )
        events = [{"event_id": "mqtt-1", "schema_version": 1, "payload": {"level_code": 1}}]
        result = transport.sync(events, acknowledged.append)

        self.assertEqual(result, {"sent": 1, "attempts": 2})
        self.assertEqual(acknowledged, ["mqtt-1"])
        self.assertEqual(calls[0][0], "agrivision/events")
        self.assertEqual(calls[0][2:], (1, False))
        message = json.loads(calls[0][1])
        self.assertEqual(message["schema_version"], 1)
        self.assertEqual(message["events"], events)
        self.assertEqual(len(message["batch_id"]), 64)


if __name__ == "__main__":
    unittest.main()
