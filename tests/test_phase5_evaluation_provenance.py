import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[1]
SYSTEM_A_CORE = ROOT / "system_a" / "core"
if str(SYSTEM_A_CORE) not in sys.path:
    sys.path.insert(0, str(SYSTEM_A_CORE))


def _sha256(content):
    return hashlib.sha256(content).hexdigest()


def _record(record_id="sample-001", *, image="images/sample-001.jpg", digest=None,
            lighting="daylight", group_id="capture-001"):
    return {
        "id": record_id,
        "image": image,
        "image_sha256": digest or ("0" * 64),
        "true": "健康",
        "pred": "健康",
        "confidence": 0.91,
        "probabilities": {"健康": 0.91, "番茄早疫病": 0.09},
        "crop": "番茄",
        "lighting": lighting,
        "device": "esp32-cam-v1",
        "source": "field",
        "split": "test",
        "group_id": group_id,
        "annotation_status": "verified",
        "annotation_source": "two-reviewer-adjudication",
    }


def _manifest(records=None):
    return {
        "schema_version": 1,
        "dataset": {
            "name": "provenance-fixture",
            "version": "2026-09-14",
            "label_policy": "one image, one primary diagnosis",
        },
        "model": {
            "name": "system-a-classifier",
            "version": "best_model.onnx",
            "weights_sha256": "1" * 64,
        },
        "labels": ["健康", "番茄早疫病"],
        "records": records or [_record()],
    }


class Phase5EvaluationProvenanceTests(unittest.TestCase):
    def test_strict_provenance_requires_reproducible_identity_and_evaluates(self):
        from evaluation import evaluate_manifest, validate_manifest

        manifest = _manifest()
        normalized = validate_manifest(manifest, require_metadata=True, require_provenance=True)
        self.assertEqual(normalized["model"]["name"], "system-a-classifier")

        result = evaluate_manifest(
            manifest,
            healthy_label="健康",
            require_provenance=True,
        )
        self.assertTrue(result["manifest"]["provenance"]["required"])
        self.assertEqual(result["manifest"]["record_count"], 1)

    def test_strict_provenance_rejects_missing_identity_fields(self):
        from evaluation import validate_manifest

        required = ("dataset", "model", "image", "image_sha256", "group_id", "annotation_status")
        for field in required:
            manifest = _manifest()
            if field in {"dataset", "model"}:
                manifest.pop(field)
            else:
                manifest["records"][0].pop(field)
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    validate_manifest(manifest, require_metadata=True, require_provenance=True)

    def test_strict_provenance_rejects_unsafe_paths_bad_hashes_and_unknown_split(self):
        from evaluation import validate_manifest

        for field, value in (
            ("image", "../outside.jpg"),
            ("image", "C:/absolute.jpg"),
            ("image_sha256", "not-a-sha256"),
            ("split", "tesst"),
        ):
            manifest = _manifest()
            manifest["records"][0][field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    validate_manifest(manifest, require_metadata=True, require_provenance=True)

    def test_strict_provenance_rejects_duplicate_content_and_cross_split_groups(self):
        from evaluation import validate_manifest

        duplicate = _manifest([
            _record("sample-001", digest="2" * 64),
            _record("sample-002", image="images/sample-002.jpg", digest="2" * 64, group_id="capture-002"),
        ])
        with self.assertRaises(ValueError):
            validate_manifest(duplicate, require_metadata=True, require_provenance=True)

        leaked = _manifest([
            _record("sample-001", group_id="capture-1"),
            dict(_record("sample-002", image="images/sample-002.jpg", digest="3" * 64,
                         group_id="capture-1"), split="validation"),
        ])
        with self.assertRaises(ValueError):
            validate_manifest(leaked, require_metadata=True, require_provenance=True)

    def test_audit_checks_files_hashes_and_slice_coverage(self):
        from dataset_audit import audit_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            first = b"first image"
            second = b"second image"
            (root / "images").mkdir()
            (root / "images" / "sample-001.jpg").write_bytes(first)
            (root / "images" / "sample-002.jpg").write_bytes(second)
            manifest = _manifest([
                _record(digest=_sha256(first)),
                _record("sample-002", image="images/sample-002.jpg", digest=_sha256(second),
                        lighting="low-light", group_id="capture-002"),
            ])

            report = audit_manifest(manifest, image_root=root, verify_files=True)

        self.assertTrue(report["valid"])
        self.assertEqual(report["file_checks"]["checked"], 2)
        self.assertEqual(report["coverage"]["lighting"]["low-light"], 1)
        self.assertEqual(report["coverage"]["split"]["test"], 2)

    def test_audit_reports_missing_and_hash_mismatch_without_silent_success(self):
        from dataset_audit import audit_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "images").mkdir()
            (root / "images" / "sample-001.jpg").write_bytes(b"changed")
            manifest = _manifest([
                _record(digest="4" * 64),
                _record("sample-002", image="images/missing.jpg", digest="5" * 64,
                        group_id="capture-002"),
            ])

            report = audit_manifest(manifest, image_root=root, verify_files=True)

        self.assertFalse(report["valid"])
        self.assertEqual(report["file_checks"]["hash_mismatch"], ["sample-001"])
        self.assertEqual(report["file_checks"]["missing"], ["sample-002"])

    def test_audit_rejects_files_above_classifier_input_bound(self):
        from dataset_audit import audit_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "images").mkdir()
            content = b"too large for this bound"
            (root / "images" / "sample-001.jpg").write_bytes(content)
            manifest = _manifest([_record(digest=_sha256(content))])
            with patch("dataset_audit.MAX_AUDIT_FILE_BYTES", 1):
                report = audit_manifest(manifest, image_root=root, verify_files=True)

        self.assertFalse(report["valid"])
        self.assertEqual(report["file_checks"]["too_large"], ["sample-001"])

    def test_audit_cli_emits_machine_readable_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            image_root = root / "images"
            image_root.mkdir()
            content = b"cli image"
            (image_root / "sample-001.jpg").write_bytes(content)
            manifest = _manifest([_record(digest=_sha256(content))])
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "audit_evaluation_dataset.py"),
                    str(manifest_path),
                    "--image-root",
                    str(root),
                    "--check-files",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertTrue(report["valid"])
        self.assertEqual(report["file_checks"]["checked"], 1)

    def test_prediction_cli_can_enforce_strict_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = pathlib.Path(directory) / "manifest.json"
            manifest_path.write_text(
                json.dumps(_manifest(), ensure_ascii=False),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "system_a" / "core" / "evaluate_predictions.py"),
                    str(manifest_path),
                    "--strict-provenance",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertTrue(result["manifest"]["provenance"]["required"])


if __name__ == "__main__":
    unittest.main()
