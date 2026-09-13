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
        source = (ROOT / "system_b" / "core" / "app.py").read_text(encoding="utf-8")
        self.assertIn("offline_event_cache.put(build_detection_event(camera_id, stable_result))", source)
        self.assertIn("@app.route('/api/offline_events/ack', methods=['POST'])", source)


if __name__ == "__main__":
    unittest.main()
