import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SYSTEM_A_CORE = ROOT / "system_a" / "core"
SYSTEM_B_CORE = ROOT / "system_b" / "core"
for path in (SYSTEM_A_CORE, SYSTEM_B_CORE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _record(record_id, truth, prediction, confidence, *, crop="番茄", lighting="daylight",
            device="esp32-cam", source="field", split="test", group_id=None,
            probabilities=None):
    record = {
        "id": record_id,
        "true": truth,
        "pred": prediction,
        "confidence": confidence,
        "crop": crop,
        "lighting": lighting,
        "device": device,
        "source": source,
        "split": split,
    }
    if group_id is not None:
        record["group_id"] = group_id
    if probabilities is not None:
        record["probabilities"] = probabilities
    return record


class Phase3ManifestAndMetricsTests(unittest.TestCase):
    def test_strict_manifest_supports_required_field_slices_and_field_error_rates(self):
        from evaluate_predictions import evaluate_manifest
        from evaluation import validate_manifest

        labels = ["健康", "番茄早疫病"]
        records = [
            _record("f1", "健康", "健康", 0.90,
                    probabilities={"健康": 0.90, "番茄早疫病": 0.10}),
            _record("f2", "健康", "番茄早疫病", 0.65,
                    probabilities={"健康": 0.35, "番茄早疫病": 0.65}, lighting="low-light"),
            _record("f3", "番茄早疫病", "健康", 0.60,
                    probabilities={"健康": 0.60, "番茄早疫病": 0.40}),
            _record("f4", "番茄早疫病", "番茄早疫病", 0.88,
                    probabilities={"健康": 0.12, "番茄早疫病": 0.88}, device="phone"),
        ]
        manifest = {
            "schema_version": 1,
            "dataset": {"name": "contract-fixture", "version": "2026-09"},
            "labels": labels,
            "records": records,
        }

        normalized = validate_manifest(manifest, require_metadata=True)
        result = evaluate_manifest(normalized, bins=2, healthy_label="健康", field_source="field")

        self.assertEqual(result["support"], 4)
        self.assertEqual(result["slices"]["crop"]["番茄"]["support"], 4)
        self.assertEqual(result["slices"]["lighting"]["low-light"]["support"], 1)
        self.assertEqual(result["slices"]["device"]["phone"]["support"], 1)
        self.assertEqual(result["slices"]["disease"]["健康"]["support"], 2)
        field = result["field_error_rates"]
        self.assertTrue(field["available"])
        self.assertEqual(field["false_positive"], 1)
        self.assertEqual(field["false_negative"], 1)
        self.assertAlmostEqual(field["false_positive_rate"], 0.5)
        self.assertAlmostEqual(field["false_negative_rate"], 0.5)
        self.assertEqual(
            sum(item["count"] for item in result["calibration"]["reliability_bins"]),
            4,
        )
        self.assertIn("brier_score", result["calibration"])

    def test_strict_manifest_rejects_missing_dimensions_and_split_group_leakage(self):
        from evaluation import validate_manifest

        base = _record("same-capture", "健康", "健康", 0.9, split="train", group_id="capture-1")
        missing_dimension = dict(base)
        missing_dimension.pop("lighting")
        with self.assertRaises(ValueError):
            validate_manifest({"schema_version": 1, "labels": ["健康"], "records": [missing_dimension]})

        leaked = [
            base,
            _record("same-capture-test", "健康", "健康", 0.9, split="test", group_id="capture-1"),
        ]
        with self.assertRaises(ValueError):
            validate_manifest({"schema_version": 1, "labels": ["健康"], "records": leaked})

    def test_probability_evidence_distinguishes_uncertain_from_ood_heuristic(self):
        from evaluation import assess_probability_vector

        result = assess_probability_vector(
            {"a": 0.34, "b": 0.33, "c": 0.33},
            threshold=0.55,
            margin_threshold=0.15,
            entropy_threshold=0.8,
            ood_max_probability=0.4,
        )
        self.assertTrue(result["uncertain"])
        self.assertTrue(result["ood_suspected"])
        self.assertEqual(result["band"], "ood")
        self.assertIn("high_entropy", result["uncertainty_reason"])

    def test_legacy_record_evaluation_remains_supported_but_marks_field_metrics_unavailable(self):
        from evaluate_predictions import evaluate_records

        result = evaluate_records(
            [{"true": "healthy", "pred": "healthy", "confidence": 0.9}],
            ["healthy", "disease"],
        )
        self.assertEqual(result["support"], 1)
        self.assertFalse(result["field_error_rates"]["available"])


class Phase3TemporalFusionTests(unittest.TestCase):
    def test_weighted_temporal_fusion_rejects_one_frame_outlier(self):
        from services.temporal_fusion import TemporalFusion

        fusion = TemporalFusion(window_size=5, min_consecutive=3, decay=0.8)
        for _ in range(4):
            result = fusion.update({"level_code": 0, "level": "正常", "disease_count": 0})
        result = fusion.update({"level_code": 3, "level": "严重", "disease_count": 10})

        self.assertEqual(result["candidate_level_code"], 0)
        self.assertEqual(result["stable_level_code"], 0)
        self.assertGreater(result["weighted_support"], 0.5)
        self.assertEqual(result["method"], "decayed_weighted_vote")

    def test_temporal_fusion_reports_pending_evidence_before_switching(self):
        from services.temporal_fusion import TemporalFusion

        fusion = TemporalFusion(window_size=5, min_consecutive=3, decay=1.0)
        first = fusion.update({"level_code": 2, "level": "警告", "disease_count": 2})
        second = fusion.update({"level_code": 2, "level": "警告", "disease_count": 3})
        third = fusion.update({"level_code": 2, "level": "警告", "disease_count": 4})

        self.assertEqual(first["stable_level_code"], 0)
        self.assertEqual(second["pending_count"], 2)
        self.assertEqual(third["stable_level_code"], 2)
        self.assertTrue(third["level_changed"])
        self.assertEqual(third["averages"]["disease_count"], 3.0)

    def test_phase3_runtime_contract_exposes_evidence_without_treating_llm_as_truth(self):
        app_source = (ROOT / "system_a" / "core" / "app_fastapi.py").read_text(encoding="utf-8")
        system_b_source = (ROOT / "system_b" / "core" / "app.py").read_text(encoding="utf-8")
        workflow = (ROOT / ".github" / "workflows" / "public-quality.yml").read_text(encoding="utf-8")
        self.assertIn("assess_probability_vector", app_source)
        self.assertIn("ood_suspected", app_source)
        self.assertIn("未经校准", app_source)
        self.assertIn("TemporalFusion", system_b_source)
        self.assertIn('state["temporal_fusion"]', system_b_source)
        self.assertIn('"temporal": temporal_summary', system_b_source)
        self.assertIn("--strict-metadata", workflow)


if __name__ == "__main__":
    unittest.main()
