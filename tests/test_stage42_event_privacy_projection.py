import json
import unittest

from system_b.core.event_mqtt import build_mqtt_payload
from system_b.core.event_schema import project_event


class Stage42EventPrivacyProjectionTests(unittest.TestCase):
    def setUp(self):
        self.event = {
            "schema_version": 1,
            "event_id": "privacy-42",
            "created_at": "2026-09-14T00:00:00+00:00",
            "camera_id": "cam1",
            "payload": {"level_code": 2, "disease_count": 1, "secret": "remove-me"},
            "url": "http://192.0.2.10/capture",
            "password": "remove-me",
            "image_path": "D:/private/image.jpg",
        }

    def test_projection_keeps_contract_fields_and_removes_unknown_fields(self):
        projected = project_event(self.event)

        self.assertNotIn("url", projected)
        self.assertNotIn("password", projected)
        self.assertNotIn("image_path", projected)
        self.assertNotIn("secret", projected["payload"])
        self.assertEqual(projected["payload"]["level_code"], 2)

    def test_mqtt_payload_is_privacy_projected(self):
        message = json.loads(build_mqtt_payload([self.event]))

        serialized = json.dumps(message, ensure_ascii=False)
        self.assertNotIn("192.0.2.10", serialized)
        self.assertNotIn("remove-me", serialized)
        self.assertNotIn("private/image.jpg", serialized)
        self.assertEqual(message["events"][0]["event_id"], "privacy-42")

    def test_http_transport_projects_before_sender_receives_events(self):
        from system_b.core.event_transport import EventTransport

        received = []
        transport = EventTransport(
            "http://127.0.0.1:9100/events",
            lambda _url, events: received.append(events) or 204,
            max_attempts=1,
            backoff_seconds=0,
        )

        result = transport.sync([self.event], lambda _event_id: None)

        self.assertEqual(result["sent"], 1)
        self.assertNotIn("password", received[0][0])
        self.assertNotIn("url", received[0][0])
        self.assertNotIn("secret", received[0][0]["payload"])


if __name__ == "__main__":
    unittest.main()
