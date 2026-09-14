import json
import tempfile
import unittest
from pathlib import Path

from system_b.core.offline_cache import OfflineEventCache


class Stage43OfflineCacheRecordGuardTests(unittest.TestCase):
    def test_list_pending_ignores_non_object_and_invalid_event_records(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = OfflineEventCache(directory)
            root = Path(directory)
            (root / "scalar.json").write_text("[]", encoding="utf-8")
            (root / "missing-id.json").write_text(json.dumps({"payload": {}}), encoding="utf-8")
            (root / "unsafe-id.json").write_text(
                json.dumps({"event_id": "../outside", "payload": {}}), encoding="utf-8"
            )
            cache.put({"event_id": "valid-43", "payload": {"level_code": 1}})

            self.assertEqual(
                [event["event_id"] for event in cache.list_pending()],
                ["valid-43"],
            )

    def test_invalid_records_do_not_reach_sync_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = OfflineEventCache(directory)
            Path(directory, "invalid.json").write_text("null", encoding="utf-8")

            self.assertEqual(cache.list_pending(), [])


if __name__ == "__main__":
    unittest.main()
