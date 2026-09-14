import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Stage35CiBrowserSmokeTests(unittest.TestCase):
    def test_ci_has_isolated_browser_smoke_job(self):
        workflow = (ROOT / ".github" / "workflows" / "public-quality.yml").read_text(encoding="utf-8")
        smoke = (ROOT / "tools" / "browser_smoke_event_status.py").read_text(encoding="utf-8")
        self.assertIn("browser-smoke:", workflow)
        self.assertIn("requirements-ui-test.txt", workflow)
        self.assertIn("playwright install chromium", workflow)
        self.assertIn("tools/browser_smoke_event_status.py", workflow)
        self.assertIn("--disable-dev-shm-usage", smoke)
        self.assertIn("--no-sandbox", smoke)

    def test_browser_job_has_bounded_server_lifecycle(self):
        workflow = (ROOT / ".github" / "workflows" / "public-quality.yml").read_text(encoding="utf-8")
        job = workflow[workflow.index("browser-smoke:"):]
        self.assertIn("timeout-minutes:", job)
        self.assertIn("curl --fail --retry", job)
        self.assertIn("if: always()", job)
        self.assertIn("kill", job)


if __name__ == "__main__":
    unittest.main()
