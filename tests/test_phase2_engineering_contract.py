import hashlib
import hmac
import json
import pathlib
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE = ROOT / "system_b" / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


class Phase2HistoryRepositoryTests(unittest.TestCase):
    def test_history_repository_persists_and_queries_sqlite_rows(self):
        from repositories.history_repository import HistoryRepository

        with tempfile.TemporaryDirectory() as directory:
            database_path = pathlib.Path(directory) / "history.db"
            repository = HistoryRepository(database_path)
            repository.initialize()
            repository.insert({
                "id": "phase2-1",
                "timestamp": "2026-09-14T10:00:00",
                "date": "2026-09-14",
                "time": "10:00:00",
                "level": "警告",
                "level_code": 2,
                "disease_count": 6,
                "disease_ratio": 0.1,
                "white_count": 1,
                "white_ratio": 0.02,
                "green_ratio": 0.7,
                "image_path": None,
                "yolo_disease_count": 4,
                "yolo_bug_count": 1,
                "dual_confidence": "medium",
                "dual_agree_disease": True,
                "dual_agree_bug": False,
            })

            page = repository.query(date_from="2026-09-14", page=1, page_size=20)
            self.assertEqual(page["total"], 1)
            self.assertEqual(page["records"][0]["id"], "phase2-1")
            self.assertEqual(repository.statistics()["total_records"], 1)
            self.assertTrue(database_path.exists())

    def test_legacy_history_json_is_migrated_once_without_runtime_full_load(self):
        from history import HistoryManager

        with tempfile.TemporaryDirectory() as directory:
            history_dir = pathlib.Path(directory) / "detection_logs"
            history_dir.mkdir()
            legacy_path = history_dir / "history.json"
            legacy_path.write_text(json.dumps([{
                "id": "legacy-1",
                "timestamp": "2026-09-13T10:00:00",
                "date": "2026-09-13",
                "time": "10:00:00",
                "level": "注意",
                "level_code": 1,
                "disease_count": 1,
                "disease_ratio": 0.01,
                "white_count": 0,
                "white_ratio": 0.0,
                "green_ratio": 0.8,
                "image_path": None,
            }], ensure_ascii=False), encoding="utf-8")

            manager = HistoryManager(history_dir=history_dir)
            result = manager.query_records(page=1)
            self.assertEqual(result["total"], 1)
            self.assertEqual(result["records"][0]["id"], "legacy-1")
            self.assertTrue((history_dir / "history.db").exists())

    def test_history_api_does_not_expose_absolute_or_external_image_paths(self):
        from history import HistoryManager

        with tempfile.TemporaryDirectory() as directory:
            history_dir = pathlib.Path(directory) / "detection_logs"
            history_dir.mkdir()
            outside = pathlib.Path(directory) / "private.jpg"
            outside.write_bytes(b"private")
            (history_dir / "history.json").write_text(json.dumps([{
                "id": "legacy-path",
                "timestamp": "2026-09-13T10:00:00",
                "date": "2026-09-13",
                "time": "10:00:00",
                "level": "注意",
                "level_code": 1,
                "image_path": str(outside),
            }], ensure_ascii=False), encoding="utf-8")

            manager = HistoryManager(history_dir=history_dir)
            record = manager.query_records(page=1)["records"][0]
            self.assertIsNone(record["image_path"])
            self.assertIsNone(manager.get_record_by_id("legacy-path")["image_path"])


class Phase2CameraTaskTests(unittest.TestCase):
    def test_camera_tasks_have_isolated_queues_and_reject_duplicate_active_work(self):
        from workers.camera_tasks import CameraTaskManager

        manager = CameraTaskManager(["cam1", "cam2"])
        first_started = threading.Event()
        release_first = threading.Event()
        second_done = threading.Event()

        def first_task():
            first_started.set()
            release_first.wait(2)

        self.assertTrue(manager.submit("cam1", "detection", first_task))
        self.assertTrue(first_started.wait(1))
        self.assertFalse(manager.submit("cam1", "detection", lambda: None))
        self.assertTrue(manager.submit("cam2", "detection", second_done.set))
        self.assertTrue(second_done.wait(1))
        release_first.set()
        manager.wait_for_idle(timeout=2)
        snapshot = manager.snapshot()
        self.assertEqual(snapshot["cameras"]["cam1"]["detection"]["completed"], 1)
        self.assertEqual(snapshot["cameras"]["cam2"]["detection"]["completed"], 1)
        manager.stop_all(timeout=2)


class Phase2StorageSafetyTests(unittest.TestCase):
    def test_camera_file_listing_rejects_paths_non_images_and_unbounded_input(self):
        from services.storage import MAX_CAMERA_FILE_LIST, filter_camera_filenames

        values = ["safe.jpg", "safe.jpg", "nested/escape.jpg", "..\\escape.jpg", "notes.txt", "safe.jpeg"]
        values.extend(f"extra-{index}.jpg" for index in range(MAX_CAMERA_FILE_LIST + 10))
        filenames = filter_camera_filenames(values)
        self.assertEqual(filenames[:2], ["safe.jpg", "safe.jpeg"])
        self.assertNotIn("nested/escape.jpg", filenames)
        self.assertNotIn("..\\escape.jpg", filenames)
        self.assertLessEqual(len(filenames), MAX_CAMERA_FILE_LIST)


class Phase2RouteModuleTests(unittest.TestCase):
    def test_monitoring_routes_are_isolated_and_camera_aware(self):
        from flask import Flask
        from routes.monitoring import create_monitoring_blueprint

        cameras = {}
        sync_states = {
            "cam1": {"syncing": False, "sync_progress": "cam1"},
            "cam2": {"syncing": True, "sync_progress": "cam2"},
        }
        for camera_id in ("cam1", "cam2"):
            cameras[camera_id] = {
                "id": camera_id,
                "name": camera_id,
                "enabled": True,
                "state": {
                    "frame_lock": threading.Lock(),
                    "latest_result": {"level": camera_id},
                    "latest_dual_result": None,
                    "last_error": "",
                    "comparison_history": [],
                    "latest_frame": b"frame",
                },
            }

        class Detector:
            def is_loaded(self):
                return True

            def get_stats(self):
                return {"loaded": True}

        flask_app = Flask(__name__)
        flask_app.register_blueprint(create_monitoring_blueprint(
            cameras=cameras,
            get_default_camera_id=lambda: "cam1",
            get_sd_sync_info=lambda camera_id: sync_states[camera_id],
            sd_lock=threading.Lock(),
            yolo_detector=Detector(),
            get_yolo_enabled=lambda: True,
            summarize_camera=lambda camera_id, state: {"id": camera_id, "has_frame": True},
            build_service_health=lambda service, checks: {"status": "ready", "checks": checks},
        ))
        client = flask_app.test_client()
        self.assertEqual(client.get("/health/live").status_code, 200)
        self.assertEqual(client.get("/api/status?camera_id=cam2").get_json()["sd_sync"]["sync_progress"], "cam2")
        self.assertEqual(client.get("/api/status?camera_id=cam1").get_json()["level"], "cam1")


class Phase2MetricsTests(unittest.TestCase):
    def test_metrics_expose_http_latency_inference_queue_and_alert_signals(self):
        from services.metrics import MetricsRegistry

        metrics = MetricsRegistry()
        metrics.observe_http("GET", "/api/status", 200, 0.125)
        metrics.observe_inference("yolo", 18.5)
        metrics.set_queue_depth("detection", 2, camera_id="cam1")
        metrics.inc_alert("success")
        rendered = metrics.prometheus_text()
        self.assertIn("agrivision_http_requests_total", rendered)
        self.assertIn('route="/api/status"', rendered)
        self.assertIn("agrivision_http_request_duration_seconds_bucket", rendered)
        self.assertIn('engine="yolo"', rendered)
        self.assertIn('task="detection"', rendered)
        self.assertIn('camera_id="cam1"', rendered)
        self.assertIn('outcome="success"', rendered)


class Phase2SecurityTests(unittest.TestCase):
    def test_api_security_requires_bearer_token_and_limits_requests(self):
        from services.security import ApiSecurity

        security = ApiSecurity(api_token="t" * 32, auth_required=True, rate_limit_max=2, rate_limit_window_seconds=60)
        headers = {"Authorization": "Bearer " + "t" * 32}
        self.assertEqual(security.authorize("/api/status", "127.0.0.1", headers).status, "allow")
        self.assertEqual(security.authorize("/api/status", "127.0.0.1", headers).status, "allow")
        self.assertEqual(security.authorize("/api/status", "127.0.0.1", headers).status, "rate_limited")
        invalid = ApiSecurity(api_token="t" * 32, auth_required=True, rate_limit_max=2, rate_limit_window_seconds=60)
        self.assertEqual(invalid.authorize("/api/status", "127.0.0.1", {}).status, "deny")
        self.assertEqual(invalid.authorize("/api/status", "127.0.0.1", {}).status, "deny")
        self.assertEqual(invalid.authorize("/api/status", "127.0.0.1", {}).status, "rate_limited")

    def test_missing_token_fails_closed_for_remote_but_allows_loopback_development(self):
        from services.security import ApiSecurity

        security = ApiSecurity(api_token="", auth_required=False, rate_limit_max=10, rate_limit_window_seconds=60)
        self.assertEqual(security.authorize("/api/status", "127.0.0.1", {}).status, "allow")
        self.assertEqual(security.authorize("/api/status", "10.0.0.8", {}).status, "deny")


class Phase2WebhookTests(unittest.TestCase):
    def test_webhook_config_does_not_echo_url_query_or_secret_and_signs_payload(self):
        from alert_notifier import AlertNotifier

        webhook_query = "access_token=" + "unit-test-token"
        secret_value = ("unit" + "-credential")
        notifier = AlertNotifier(
            webhook_url="https://oapi.dingtalk.com/robot/send?" + webhook_query,
            webhook_secret=secret_value,
        )
        config = notifier.get_config()
        rendered = json.dumps(config, ensure_ascii=False)
        self.assertNotIn(webhook_query, rendered)
        self.assertNotIn(secret_value, rendered)
        self.assertTrue(config["webhook_configured"])

        with mock.patch("alert_notifier.requests.post") as post:
            post.return_value.status_code = 200
            post.return_value.json.return_value = {"errcode": 0}
            self.assertTrue(notifier._send_dingtalk("payload"))
            headers = post.call_args.kwargs["headers"]
            self.assertTrue(headers["X-AgriVision-Signature"].startswith("sha256="))
            body = post.call_args.kwargs["data"]
            expected = hmac.new(secret_value.encode(), body, hashlib.sha256).hexdigest()
            self.assertEqual(headers["X-AgriVision-Signature"], "sha256=" + expected)

    def test_webhook_host_must_be_allowlisted(self):
        from alert_notifier import AlertNotifier

        notifier = AlertNotifier()
        with self.assertRaises(ValueError):
            notifier.configure(webhook_url="https://169.254.169.254/latest")

    def test_same_level_alerts_are_serialized_while_one_send_is_in_flight(self):
        from alert_notifier import AlertNotifier

        notifier = AlertNotifier(webhook_url="http://127.0.0.1/notify")
        notifier.configure(enabled=True, cooldown=0)
        entered = threading.Event()
        release = threading.Event()

        def send(_message):
            entered.set()
            release.wait(2)
            return True

        results = []
        with mock.patch.object(notifier, "_send_dingtalk", side_effect=send):
            first = threading.Thread(target=lambda: results.append(notifier.notify("警告", 2)))
            first.start()
            self.assertTrue(entered.wait(1))
            second = notifier.notify("警告", 2)
            release.set()
            first.join(2)

        self.assertEqual(second["reason"], "同等级告警正在发送")
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["sent"])


class Phase2ArchitectureTests(unittest.TestCase):
    def test_engineering_modules_are_wired_into_system_b(self):
        source = (CORE / "app.py").read_text(encoding="utf-8")
        for marker in (
            "register_engineering_routes",
            "register_monitoring_routes",
            "CameraTaskManager",
            "MetricsRegistry",
            "ApiSecurity",
            "schemas.api",
        ):
            self.assertIn(marker, source)
        for relative in (
            "routes/engineering.py",
            "routes/monitoring.py",
            "services/metrics.py",
            "services/security.py",
            "services/storage.py",
            "workers/camera_tasks.py",
            "repositories/history_repository.py",
            "schemas/api.py",
            "Dockerfile.system-b",
            "deploy/start_system_b.ps1",
        ):
            self.assertTrue((ROOT / "system_b" / "core" / relative).exists() if relative.startswith(("routes/", "services/", "workers/", "repositories/", "schemas/")) else (ROOT / relative).exists(), relative)
        main_source = source.split("if __name__ == '__main__':", 1)[1]
        self.assertLess(main_source.index("if bind_host not in"), main_source.index("run_detection_once(cid)"))

    def test_deployment_artifacts_have_safe_defaults(self):
        dockerfile = (ROOT / "Dockerfile.system-b").read_text(encoding="utf-8")
        compose = (ROOT / "docker-compose.system-b.yml").read_text(encoding="utf-8")
        launcher = (ROOT / "deploy" / "start_system_b.ps1").read_text(encoding="utf-8")
        self.assertIn("python:3.11-slim", dockerfile)
        self.assertIn("AGRIVISION_API_AUTH_REQUIRED=true", dockerfile)
        self.assertIn("AGRIVISION_API_TOKEN:?", compose)
        self.assertIn("/app/system_b/core/detection_logs", compose)
        self.assertIn("AGRIVISION_API_TOKEN", launcher)
        self.assertIn("127.0.0.1", launcher)


if __name__ == "__main__":
    unittest.main()
