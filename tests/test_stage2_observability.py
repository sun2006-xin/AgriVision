import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system_a" / "core"))
sys.path.insert(0, str(ROOT / "system_b" / "core"))


class Stage2ObservabilityTests(unittest.TestCase):
    def test_service_health_reports_ready_only_when_required_checks_pass(self):
        from health import build_service_health

        ready = build_service_health("system-b", {"yolo": True, "camera": True})
        degraded = build_service_health("system-b", {"yolo": False, "camera": True})

        self.assertEqual(ready["status"], "ready")
        self.assertEqual(degraded["status"], "degraded")
        self.assertEqual(degraded["failed_checks"], ["yolo"])

    def test_camera_health_exposes_only_operational_summary(self):
        from health import summarize_camera

        summary = summarize_camera("cam1", {"latest_frame": b"jpeg", "last_error": ""})
        self.assertEqual(summary, {"id": "cam1", "has_frame": True, "last_error": None})

    def test_request_id_accepts_safe_existing_value_or_generates_one(self):
        from observability import resolve_request_id

        self.assertEqual(resolve_request_id("request-123"), "request-123")
        generated = resolve_request_id(None)
        self.assertRegex(generated, r"^[a-f0-9]{32}$")
        self.assertNotEqual(resolve_request_id("bad value"), "bad value")


if __name__ == "__main__":
    unittest.main()
