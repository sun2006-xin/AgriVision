import tempfile
import unittest

from system_b.core.offline_cache import OfflineEventCache


class Stage41OfflineCacheSizeGuardTests(unittest.TestCase):
    def test_oversized_event_is_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = OfflineEventCache(directory, max_bytes=100)

            with self.assertRaisesRegex(ValueError, "event exceeds offline cache byte limit"):
                cache.put({"event_id": "too-large", "payload": {"text": "x" * 200}})

            self.assertEqual(cache.list_pending(), [])

    def test_oversized_replacement_does_not_remove_existing_event(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = OfflineEventCache(directory, max_bytes=180)
            cache.put({"event_id": "safe-1", "payload": {"level_code": 1}})

            with self.assertRaises(ValueError):
                cache.put({"event_id": "safe-1", "payload": {"text": "x" * 300}})

            self.assertEqual([item["event_id"] for item in cache.list_pending()], ["safe-1"])


if __name__ == "__main__":
    unittest.main()
