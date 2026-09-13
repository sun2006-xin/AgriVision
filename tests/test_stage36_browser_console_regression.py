import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Stage36BrowserConsoleRegressionTests(unittest.TestCase):
    def test_smoke_collects_and_rejects_page_errors(self):
        source = (ROOT / "tools" / "browser_smoke_event_status.py").read_text(encoding="utf-8")
        self.assertIn("page_errors = []", source)
        self.assertIn("console_errors = []", source)
        self.assertIn("page.on(\"pageerror\"", source)
        self.assertIn("msg.type == \"error\"", source)
        self.assertIn("assert not page_errors", source)
        self.assertIn("assert not console_errors", source)


if __name__ == "__main__":
    unittest.main()
