import tempfile
import unittest
from pathlib import Path

from system_b.core.offline_cache import OfflineEventCache


class Stage45OfflineCacheFilesystemGuardTests(unittest.TestCase):
    def test_non_file_json_entry_does_not_break_cache_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = OfflineEventCache(directory)
            Path(directory, "directory.json").mkdir()
            cache.put({"event_id": "valid-45", "payload": {"level_code": 1}})

            self.assertEqual(
                [event["event_id"] for event in cache.list_pending()],
                ["valid-45"],
            )

    def test_cache_uses_safe_file_enumerator_for_read_and_trim(self):
        source = (Path(__file__).resolve().parents[1] / "system_b" / "core" / "offline_cache.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def _event_paths_by_mtime", source)
        self.assertGreaterEqual(source.count("_event_paths_by_mtime()"), 2)


if __name__ == "__main__":
    unittest.main()
