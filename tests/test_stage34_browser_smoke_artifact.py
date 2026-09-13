import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Stage34BrowserSmokeArtifactTests(unittest.TestCase):
    def test_browser_smoke_is_optional_and_points_at_safe_state(self):
        requirements = (ROOT / "requirements-ui-test.txt").read_text(encoding="utf-8")
        script = (ROOT / "tools" / "browser_smoke_event_status.py").read_text(encoding="utf-8")
        self.assertIn("playwright", requirements)
        self.assertIn("#eventSyncInfo", script)
        self.assertIn("事件同步状态暂不可用", script)
        self.assertNotIn("password", script.lower())
        self.assertNotIn("username", script.lower())


if __name__ == "__main__":
    unittest.main()
