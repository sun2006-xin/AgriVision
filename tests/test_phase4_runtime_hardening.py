import pathlib
import re
import shutil
import sys
import subprocess
import tempfile
import unittest
import threading
import time
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE = ROOT / "system_b" / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


class Phase4ConfigValidationTests(unittest.TestCase):
    def _manager(self):
        from config import ConfigManager, DEFAULT_CONFIG

        manager = ConfigManager.__new__(ConfigManager)
        manager.current_config = DEFAULT_CONFIG.to_dict()
        return manager

    def test_integer_parameters_reject_fractional_values(self):
        manager = self._manager()
        valid, errors = manager.validate_params({"GREEN_ADV": 12.5})
        self.assertFalse(valid)
        self.assertTrue(any("必须为整数" in error for error in errors))

    def test_related_thresholds_are_validated_as_a_single_configuration(self):
        manager = self._manager()
        valid, errors = manager.validate_params({
            "NOTICE_DISEASE_COUNT": 4,
            "WARNING_DISEASE_COUNT": 3,
        })
        self.assertFalse(valid)
        self.assertTrue(any("病斑数量阈值" in error for error in errors))

    def test_invalid_patch_is_rejected_before_persistence(self):
        manager = self._manager()
        with mock.patch.object(manager, "save_config") as save_config:
            with self.assertRaises(ValueError):
                manager.update_params({"MORPH_KERNEL_SIZE": 4})
        save_config.assert_not_called()

    def test_invalid_persisted_config_falls_back_to_defaults(self):
        from config import ConfigManager, DEFAULT_CONFIG

        with tempfile.TemporaryDirectory() as directory:
            config_file = pathlib.Path(directory) / "detection_params.json"
            config_file.write_text('{"GREEN_ADV": 999}', encoding="utf-8")
            with mock.patch("config.CONFIG_DIR", directory), mock.patch(
                "config.CONFIG_FILE", str(config_file)
            ):
                manager = ConfigManager()
            self.assertEqual(manager.get_current_config(), DEFAULT_CONFIG.to_dict())


class Phase4TaskLifecycleTests(unittest.TestCase):
    def test_failed_task_retries_with_bounded_attempts(self):
        from workers.camera_tasks import CameraTaskManager

        manager = CameraTaskManager(["cam1"])
        attempts = []

        def flaky_task():
            attempts.append(len(attempts) + 1)
            if len(attempts) < 2:
                raise RuntimeError("transient")

        self.assertTrue(manager.submit("cam1", "detection", flaky_task, max_retries=1))
        self.assertTrue(manager.wait_for_idle(timeout=2))
        status = manager.snapshot()["cameras"]["cam1"]["detection"]
        self.assertEqual(attempts, [1, 2])
        self.assertEqual(status["completed"], 1)
        self.assertEqual(status["failed"], 0)
        self.assertEqual(status["retries"], 1)
        manager.stop_all(timeout=2)

    def test_cancel_drains_queued_work_and_allows_clean_shutdown(self):
        from workers.camera_tasks import CameraTaskManager

        manager = CameraTaskManager(["cam1"])
        started = threading.Event()
        release = threading.Event()
        executed = []

        def running_task():
            started.set()
            release.wait(2)

        self.assertTrue(manager.submit("cam1", "detection", running_task))
        self.assertTrue(started.wait(1))
        self.assertFalse(manager.submit("cam1", "detection", lambda: executed.append(True)))
        result = manager.cancel("cam1", "detection")
        release.set()
        self.assertTrue(manager.wait_for_idle(timeout=2))
        self.assertEqual(result["queued_cancelled"], 0)
        self.assertEqual(executed, [])
        manager.stop_all(timeout=2)
        time.sleep(0.02)
        self.assertTrue(manager.submit("cam1", "detection", lambda: executed.append(True)))
        self.assertTrue(manager.wait_for_idle(timeout=2))
        manager.stop_all(timeout=2)


class Phase4MediaSecurityTests(unittest.TestCase):
    def test_media_and_history_paths_require_authentication_remotely(self):
        from services.security import ApiSecurity

        token = "s" * 32
        security = ApiSecurity(api_token=token, auth_required=True)
        headers = {"Authorization": "Bearer " + token}
        self.assertTrue(security.is_protected_path("/video_feed/cam1"))
        self.assertTrue(security.is_protected_path("/dataset/leaf.jpg"))
        self.assertTrue(security.is_protected_path("/history/image/r1"))
        self.assertEqual(security.authorize("/video_feed/cam1", "10.0.0.8", {}).status, "deny")
        session_id = security.issue_session("10.0.0.8", headers)
        self.assertTrue(session_id)
        self.assertEqual(
            security.authorize(
                "/video_feed/cam1",
                "10.0.0.8",
                {},
                {security.SESSION_COOKIE_NAME: session_id},
            ).status,
            "allow",
        )
        self.assertEqual(
            security.authorize(
                "/video_feed/cam1",
                "10.0.0.9",
                {},
                {security.SESSION_COOKIE_NAME: session_id},
            ).status,
            "deny",
        )


class Phase4RouteCompositionTests(unittest.TestCase):
    def test_system_b_uses_blueprint_layers_for_all_http_boundaries(self):
        app_source = (CORE / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("@app.route", app_source)
        for module_name in ("control", "events", "video", "diagnosis", "pages"):
            source = (CORE / "routes" / f"{module_name}.py").read_text(encoding="utf-8")
            self.assertIn("Blueprint", source)
            self.assertIn("register_", source)
        for marker in (
            "register_control_routes",
            "register_event_routes",
            "register_video_routes",
            "register_diagnosis_routes",
            "register_page_routes",
        ):
            self.assertIn(marker, app_source)

    def test_page_templates_have_parseable_inline_scripts(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not installed")
        for template_name in ("main.html", "dashboard.html"):
            source = (CORE / "templates" / template_name).read_text(encoding="utf-8")
            scripts = re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", source, flags=re.DOTALL)
            for index, script in enumerate(scripts, start=1):
                result = subprocess.run(
                    [node, "--check", "-"],
                    input=script.encode("utf-8"),
                    capture_output=True,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    f"{template_name} inline script {index} is invalid: "
                    f"{result.stderr.decode('utf-8', 'replace')}",
                )

if __name__ == "__main__":
    unittest.main()
