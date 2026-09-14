import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Phase1StableRuntimeContractTests(unittest.TestCase):
    def test_system_a_report_and_system_b_proxy_use_file_field(self):
        system_a = (ROOT / "system_a" / "core" / "app_fastapi.py").read_text(encoding="utf-8")
        system_b = (ROOT / "system_b" / "core" / "app.py").read_text(encoding="utf-8")
        self.assertIn('async def report_endpoint(file: UploadFile = File(...))', system_a)
        self.assertIn('files={"file": ("frame.jpg", image_bytes, "image/jpeg")}', system_b)

    def test_system_b_params_and_health_contracts_are_wired(self):
        source = (ROOT / "system_b" / "core" / "app.py").read_text(encoding="utf-8")
        monitoring = (ROOT / "system_b" / "core" / "routes" / "monitoring.py").read_text(encoding="utf-8")
        self.assertIn("config_manager.validate_params(params)", source)
        self.assertIn("register_monitoring_routes", source)
        self.assertIn('@blueprint.get("/health/live")', monitoring)
        self.assertIn('@blueprint.get("/health/ready")', monitoring)

    def test_phase1_api_smoke_is_in_public_ci(self):
        workflow = (ROOT / ".github" / "workflows" / "public-quality.yml").read_text(encoding="utf-8")
        self.assertIn("tools/api_smoke_phase1.py", workflow)

    def test_public_requirements_are_exactly_pinned(self):
        for name in ("requirements-a.txt", "requirements-b.txt", "requirements-test.txt", "requirements-mqtt.txt", "requirements-ui-test.txt"):
            lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
            package_lines = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
            self.assertTrue(package_lines, name)
            self.assertTrue(all("==" in line for line in package_lines), name)


if __name__ == "__main__":
    unittest.main()
