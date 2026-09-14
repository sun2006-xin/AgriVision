import hashlib
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
SYSTEM_A_CORE = ROOT / "system_a" / "core"
if str(SYSTEM_A_CORE) not in sys.path:
    sys.path.insert(0, str(SYSTEM_A_CORE))


class _Input:
    name = "input"


class _Output:
    name = "output"


class _FakeSession:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32)

    def get_inputs(self):
        return [_Input()]

    def get_outputs(self):
        return [_Output()]

    def run(self, output_names, inputs):
        self.last_inputs = inputs
        return [self.scores]


def _manifest(image_sha256):
    return {
        "schema_version": 1,
        "dataset": {
            "name": "runner-fixture",
            "version": "2026-09-14",
            "label_policy": "one image, one primary diagnosis",
        },
        "model": {
            "name": "fixture-classifier",
            "version": "fixture-1",
            "weights_sha256": "a" * 64,
        },
        "labels": ["healthy", "disease"],
        "records": [
            {
                "id": "sample-001",
                "image": "images/sample-001.jpg",
                "image_sha256": image_sha256,
                "true": "disease",
                "crop": "tomato",
                "lighting": "daylight",
                "device": "camera-v1",
                "source": "field",
                "split": "test",
                "group_id": "capture-001",
                "annotation_status": "verified",
                "annotation_source": "fixture-review",
            }
        ],
    }


class Phase6ClassifierRunnerTests(unittest.TestCase):
    def test_runner_converts_logits_to_stable_probability_evidence(self):
        from classifier_inference import ClassifierRunner

        runner = ClassifierRunner(
            _FakeSession([[1000.0, 1001.0]]),
            class_names=["healthy", "disease"],
            preprocess=lambda content: np.zeros((1, 3, 224, 224), dtype=np.float32),
        )
        prediction = runner.predict(b"image")

        self.assertEqual(prediction["pred"], "disease")
        self.assertEqual(set(prediction["probabilities"]), {"healthy", "disease"})
        self.assertAlmostEqual(sum(prediction["probabilities"].values()), 1.0, places=6)
        self.assertGreaterEqual(prediction["confidence"], 0.5)

    def test_runner_rejects_non_finite_or_wrong_class_count_outputs(self):
        from classifier_inference import ClassifierRunner

        preprocess = lambda content: np.zeros((1, 3, 224, 224), dtype=np.float32)
        with self.assertRaises(ValueError):
            ClassifierRunner(
                _FakeSession([[float("nan"), 0.0]]),
                class_names=["healthy", "disease"],
                preprocess=preprocess,
            ).predict(b"image")
        with self.assertRaises(ValueError):
            ClassifierRunner(
                _FakeSession([[0.0, 1.0, 2.0]]),
                class_names=["healthy", "disease"],
                preprocess=preprocess,
            ).predict(b"image")

    def test_build_manifest_verifies_input_files_and_adds_predictions(self):
        from classifier_inference import build_prediction_manifest

        content = b"runner image"
        digest = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "images").mkdir()
            (root / "images" / "sample-001.jpg").write_bytes(content)
            result = build_prediction_manifest(
                _manifest(digest),
                image_root=root,
                session=_FakeSession([[0.0, 2.0]]),
                preprocess=lambda image: np.zeros((1, 3, 224, 224), dtype=np.float32),
                expected_model_sha256="a" * 64,
            )

        record = result["records"][0]
        self.assertEqual(record["pred"], "disease")
        self.assertIn("probabilities", record)
        self.assertEqual(record["true"], "disease")

    def test_build_manifest_rejects_model_hash_mismatch(self):
        from classifier_inference import build_prediction_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "images").mkdir()
            content = b"runner image"
            digest = hashlib.sha256(content).hexdigest()
            (root / "images" / "sample-001.jpg").write_bytes(content)
            with self.assertRaises(ValueError):
                build_prediction_manifest(
                    _manifest(digest),
                    image_root=root,
                    session=_FakeSession([[0.0, 2.0]]),
                    preprocess=lambda image: np.zeros((1, 3, 224, 224), dtype=np.float32),
                    expected_model_sha256="b" * 64,
                )

    def test_build_manifest_rechecks_bytes_after_initial_audit(self):
        from classifier_inference import build_prediction_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "images").mkdir()
            (root / "images" / "sample-001.jpg").write_bytes(b"changed-after-audit")
            manifest = _manifest(hashlib.sha256(b"original").hexdigest())
            with patch("classifier_inference.audit_manifest", return_value={
                "valid": True,
                "file_checks": {},
            }):
                with self.assertRaisesRegex(ValueError, "changed after the provenance audit"):
                    build_prediction_manifest(
                        manifest,
                        image_root=root,
                        session=_FakeSession([[0.0, 2.0]]),
                        preprocess=lambda image: np.zeros((1, 3, 224, 224), dtype=np.float32),
                        expected_model_sha256="a" * 64,
                    )

    def test_system_a_reuses_shared_classifier_contract(self):
        source = (ROOT / "system_a" / "core" / "app_fastapi.py").read_text(encoding="utf-8")
        self.assertIn("from classifier_inference import", source)
        self.assertIn("predict_classifier(", source)
        self.assertNotIn("transform = transforms.Compose", source)

    def test_runner_cli_bounds_onnx_thread_count_before_loading_inputs(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "generate_prediction_manifest.py"),
                "missing.json",
                "--image-root",
                ".",
                "--model",
                "missing.onnx",
                "--output",
                "missing-output.json",
                "--threads",
                "0",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--threads must be between 1 and 8", completed.stderr)


if __name__ == "__main__":
    unittest.main()
