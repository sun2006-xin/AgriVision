import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system_a" / "core"))


class Stage3EvaluationTests(unittest.TestCase):
    def test_classification_metrics_include_confusion_matrix_and_per_class_scores(self):
        from evaluation import classification_metrics

        result = classification_metrics(
            y_true=["healthy", "disease", "disease", "healthy"],
            y_pred=["healthy", "disease", "healthy", "healthy"],
            labels=["healthy", "disease"],
        )

        self.assertEqual(result["support"], 4)
        self.assertEqual(result["correct"], 3)
        self.assertEqual(result["confusion_matrix"], [[2, 0], [1, 1]])
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["per_class"]["disease"]["recall"], 0.5)

    def test_expected_calibration_error_is_bounded(self):
        from evaluation import expected_calibration_error

        value = expected_calibration_error(
            confidences=[0.9, 0.6, 0.4, 0.8],
            correct=[True, True, False, False],
            bins=2,
        )
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_low_confidence_is_explicitly_marked_uncertain(self):
        from evaluation import assess_confidence

        self.assertEqual(assess_confidence(0.49), {"uncertain": True, "band": "uncertain"})
        self.assertEqual(assess_confidence(0.8), {"uncertain": False, "band": "high"})

    def test_prediction_manifest_evaluation_adds_calibration_error(self):
        from evaluate_predictions import evaluate_records

        result = evaluate_records(
            [
                {"true": "healthy", "pred": "healthy", "confidence": 0.9},
                {"true": "disease", "pred": "healthy", "confidence": 0.8},
            ],
            ["healthy", "disease"],
            bins=2,
        )
        self.assertEqual(result["support"], 2)
        self.assertIn("ece", result)


if __name__ == "__main__":
    unittest.main()
