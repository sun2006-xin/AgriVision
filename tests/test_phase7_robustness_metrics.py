import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SYSTEM_A_CORE = ROOT / "system_a" / "core"
if str(SYSTEM_A_CORE) not in sys.path:
    sys.path.insert(0, str(SYSTEM_A_CORE))


def _record(record_id, truth, prediction, *, confidence, disease_probability,
            lighting="daylight", device="camera-a", source="field"):
    return {
        "id": record_id,
        "true": truth,
        "pred": prediction,
        "confidence": confidence,
        "probabilities": {
            "健康": 1.0 - disease_probability,
            "病害": disease_probability,
        },
        "crop": "番茄",
        "lighting": lighting,
        "device": device,
        "source": source,
        "split": "test",
    }


class Phase7RobustnessMetricTests(unittest.TestCase):
    def test_reports_cross_lighting_and_device_accuracy_gaps(self):
        from evaluation import evaluate_records

        records = [
            _record("day-healthy", "健康", "健康", confidence=0.9, disease_probability=0.1),
            _record("day-disease", "病害", "病害", confidence=0.9, disease_probability=0.9),
            _record(
                "low-healthy", "健康", "病害", confidence=0.6, disease_probability=0.6,
                lighting="low-light", device="phone",
            ),
            _record(
                "low-disease", "病害", "病害", confidence=0.7, disease_probability=0.7,
                lighting="low-light", device="phone",
            ),
        ]

        result = evaluate_records(records, ["健康", "病害"], healthy_label="健康")
        lighting = result["robustness"]["lighting"]
        device = result["robustness"]["device"]

        self.assertTrue(lighting["comparable"])
        self.assertEqual(lighting["worst_slice"], "low-light")
        self.assertAlmostEqual(lighting["accuracy_gap"], 0.5)
        self.assertEqual(device["worst_slice"], "phone")
        self.assertAlmostEqual(device["macro_f1_gap"], 2 / 3)
        self.assertIn("unpaired", lighting["interpretation"])

    def test_field_threshold_curve_reports_operational_fp_fn_tradeoff(self):
        from evaluation import evaluate_records

        records = [
            _record("healthy-clear", "健康", "健康", confidence=0.8, disease_probability=0.2),
            _record("healthy-noisy", "健康", "病害", confidence=0.7, disease_probability=0.7),
            _record("disease-clear", "病害", "病害", confidence=0.8, disease_probability=0.8),
            _record("disease-hidden", "病害", "健康", confidence=0.6, disease_probability=0.4),
        ]

        result = evaluate_records(records, ["健康", "病害"], healthy_label="健康")
        curve = result["field_threshold_curve"]
        at_default = next(row for row in curve["thresholds"] if row["threshold"] == 0.55)

        self.assertTrue(curve["available"])
        self.assertEqual(at_default["false_positive"], 1)
        self.assertEqual(at_default["false_negative"], 1)
        self.assertAlmostEqual(at_default["false_positive_rate"], 0.5)
        self.assertAlmostEqual(at_default["false_negative_rate"], 0.5)
        self.assertEqual(curve["score"], "probability_of_non_healthy")

    def test_threshold_curve_is_unavailable_without_probability_vectors(self):
        from evaluation import evaluate_records

        result = evaluate_records(
            [
                {
                    "true": "healthy",
                    "pred": "healthy",
                    "confidence": 0.9,
                    "source": "field",
                }
            ],
            ["healthy", "disease"],
            healthy_label="healthy",
        )
        curve = result["field_threshold_curve"]
        self.assertFalse(curve["available"])
        self.assertIn("probabilities", curve["reason"])

    def test_threshold_curve_rejects_invalid_thresholds(self):
        from evaluation import field_threshold_curve

        record = _record("sample", "健康", "健康", confidence=0.9, disease_probability=0.1)
        with self.assertRaises(ValueError):
            field_threshold_curve([record], healthy_label="健康", thresholds=[-0.1, 0.5])
        with self.assertRaises(ValueError):
            field_threshold_curve([record], healthy_label="健康", thresholds=[0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
