import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Stage33EventStatusIsolationTests(unittest.TestCase):
    def test_event_status_has_independent_refresh_handler(self):
        source = (ROOT / "system_b" / "core" / "app.py").read_text(encoding="utf-8")
        self.assertIn("async function loadEventSyncStatus()", source)
        self.assertIn("target.textContent = formatEventSyncStatus(data)", source)
        self.assertIn("事件同步状态暂不可用", source)

    def test_monitor_refresh_delegates_event_status_failures(self):
        source = (ROOT / "system_b" / "core" / "app.py").read_text(encoding="utf-8")
        update_start = source.index("async function updateStatus()")
        update_end = source.index("function renderAgree", update_start)
        update_body = source[update_start:update_end]
        self.assertIn("loadEventSyncStatus();", update_body)
        self.assertNotIn("await fetch('/api/offline_events/sync_status')", update_body)


if __name__ == "__main__":
    unittest.main()
