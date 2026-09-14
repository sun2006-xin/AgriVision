import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SECURITY_PATH = ROOT / "system_a" / "core" / "security.py"


def load_system_a_security():
    spec = importlib.util.spec_from_file_location("agrivision_system_a_security_test", SECURITY_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SystemASecurityTests(unittest.TestCase):
    def test_system_a_protects_inference_history_and_async_results(self):
        module = load_system_a_security()
        security = module.ApiSecurity(api_token="a" * 32, rate_limit_max=10)
        headers = {"Authorization": "Bearer " + "a" * 32}
        self.assertEqual(security.authorize("/health/live", "10.0.0.8", {}).status, "allow")
        self.assertEqual(security.authorize("/docs", "10.0.0.8", {}).status, "allow")
        self.assertEqual(security.authorize("/report", "10.0.0.8", headers).status, "allow")
        self.assertEqual(security.authorize("/history/export/csv", "10.0.0.8", {}).status, "deny")
        self.assertEqual(security.authorize("/result/task-1", "10.0.0.8", {}).status, "deny")

    def test_system_a_fails_closed_for_remote_without_token(self):
        module = load_system_a_security()
        security = module.ApiSecurity(api_token="", auth_required=False)
        self.assertEqual(security.authorize("/diagnose", "127.0.0.1", {}).status, "allow")
        self.assertEqual(security.authorize("/diagnose", "10.0.0.8", {}).status, "deny")

    def test_rate_limit_source_tracking_is_bounded(self):
        module = load_system_a_security()
        security = module.ApiSecurity(api_token="a" * 32, rate_limit_max=10)
        for index in range(security.MAX_TRACKED_CLIENTS + 20):
            security.authorize("/predict", f"192.0.2.{index}", {"Authorization": "Bearer " + "a" * 32})
        self.assertLessEqual(len(security._windows), security.MAX_TRACKED_CLIENTS)

    def test_system_a_and_b_use_same_auth_boundary_and_proxy_header(self):
        system_a = (ROOT / "system_a" / "core" / "app_fastapi.py").read_text(encoding="utf-8")
        security_source = SECURITY_PATH.read_text(encoding="utf-8")
        system_b = (ROOT / "system_b" / "core" / "app.py").read_text(encoding="utf-8")
        frontend = (ROOT / "system_a" / "core" / "frontend.html").read_text(encoding="utf-8")
        self.assertIn("api_security_middleware", system_a)
        self.assertIn('AGRIVISION_API_TOKEN', security_source)
        self.assertIn('headers={"Authorization": f"Bearer {SYSTEM_A_API_TOKEN}"}', system_b)
        self.assertIn("sessionStorage.getItem('agrivision_api_token')", frontend)
        self.assertNotIn("agrivision_api_token", system_a)

    def test_system_a_deployment_script_requires_token_for_remote_binding(self):
        launcher = (ROOT / "deploy" / "start_system_a.ps1").read_text(encoding="utf-8")
        self.assertIn("AGRIVISION_SYSTEM_A_BIND_HOST", launcher)
        self.assertIn("AGRIVISION_API_TOKEN", launcher)
        self.assertIn("--workers 1", launcher)


if __name__ == "__main__":
    unittest.main()
