import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "system_b" / "core"))


class Stage4EdgeTests(unittest.TestCase):
    def test_detection_event_is_versioned_and_does_not_include_private_config(self):
        from event_schema import build_detection_event

        event = build_detection_event(
            "cam1",
            {
                "level": "警告",
                "level_code": 2,
                "disease_count": 3,
                "white_count": 1,
                "disease_ratio": 0.12,
                "white_ratio": 0.01,
                "green_ratio": 0.61,
            },
        )

        self.assertEqual(event["schema_version"], 1)
        self.assertEqual(event["camera_id"], "cam1")
        self.assertEqual(event["payload"]["level_code"], 2)
        self.assertNotIn("url", event)
        self.assertNotIn("webhook", event)
        self.assertNotIn("image", event)

    def test_offline_cache_is_bounded_and_supports_acknowledgement(self):
        from offline_cache import OfflineEventCache

        with tempfile.TemporaryDirectory() as directory:
            cache = OfflineEventCache(directory, max_items=2, max_bytes=100_000)
            first = {"event_id": "event-1", "payload": {"level_code": 1}}
            second = {"event_id": "event-2", "payload": {"level_code": 2}}
            third = {"event_id": "event-3", "payload": {"level_code": 3}}
            cache.put(first)
            cache.put(second)
            cache.put(third)

            pending = cache.list_pending()
            self.assertEqual([item["event_id"] for item in pending], ["event-2", "event-3"])
            cache.ack("event-2")
            self.assertEqual([item["event_id"] for item in cache.list_pending()], ["event-3"])

    def test_offline_cache_rejects_path_traversal_event_ids(self):
        from offline_cache import OfflineEventCache

        with tempfile.TemporaryDirectory() as directory:
            cache = OfflineEventCache(directory)
            with self.assertRaises(ValueError):
                cache.put({"event_id": "../outside", "payload": {}})

    def test_system_b_exposes_offline_queue_contract(self):
        app_source = (ROOT / "system_b" / "core" / "app.py").read_text(encoding="utf-8")
        events_source = (ROOT / "system_b" / "core" / "routes" / "events.py").read_text(encoding="utf-8")
        source = app_source + "\n" + events_source
        self.assertIn("offline_event_cache.put(build_detection_event(camera_id, stable_result))", source)
        self.assertIn('@blueprint.post("/api/offline_events/ack")', events_source)
        self.assertIn('@blueprint.post("/api/offline_events/sync")', events_source)
        self.assertIn("AGRIVISION_EVENTS_SINK_URL", source)
        self.assertIn("offline_event_sync_lock", source)
        self.assertIn("validate_sync_request(payload)", events_source)
        transport_source = (ROOT / "system_b" / "core" / "event_transport.py").read_text(encoding="utf-8")
        self.assertIn("Idempotency-Key", transport_source)

    def test_sink_url_rejects_unsafe_or_ambiguous_destinations(self):
        from event_transport import validate_sink_url

        self.assertEqual(validate_sink_url("http://127.0.0.1:9100/events"), "http://127.0.0.1:9100/events")
        with self.assertRaises(ValueError):
            validate_sink_url("http://example.com/events")
        with self.assertRaises(ValueError):
            validate_sink_url("https://user:password@example.com/events")

    def test_transport_retries_and_acknowledges_only_after_success(self):
        from event_transport import EventTransport

        attempts = []
        acknowledged = []

        def sender(url, events):
            attempts.append((url, events))
            if len(attempts) < 3:
                raise OSError("offline")
            return 202

        transport = EventTransport("http://127.0.0.1:9100/events", sender, max_attempts=3)
        result = transport.sync([{"event_id": "event-1"}], acknowledged.append)
        self.assertEqual(result, {"sent": 1, "attempts": 3})
        self.assertEqual(acknowledged, ["event-1"])

    def test_batch_idempotency_key_is_stable_and_order_sensitive(self):
        from event_transport import build_batch_idempotency_key

        first = [{"event_id": "event-1"}, {"event_id": "event-2"}]
        second = [{"event_id": "event-1"}, {"event_id": "event-2"}]
        reordered = [{"event_id": "event-2"}, {"event_id": "event-1"}]
        self.assertEqual(build_batch_idempotency_key(first), build_batch_idempotency_key(second))
        self.assertNotEqual(build_batch_idempotency_key(first), build_batch_idempotency_key(reordered))
        self.assertEqual(len(build_batch_idempotency_key(first)), 64)


if __name__ == "__main__":
    unittest.main()
