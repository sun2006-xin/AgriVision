"""Evaluate a JSON prediction manifest without loading any model."""

import argparse
import json
from pathlib import Path

from evaluation import classification_metrics, expected_calibration_error


def evaluate_records(records, labels, bins=10):
    """Evaluate records with true, predicted and confidence fields."""
    y_true = [record["true"] for record in records]
    y_pred = [record["pred"] for record in records]
    confidences = [record["confidence"] for record in records]
    result = classification_metrics(y_true, y_pred, labels)
    result["ece"] = expected_calibration_error(
        confidences, [truth == prediction for truth, prediction in zip(y_true, y_pred)], bins
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate AgriVision prediction manifest")
    parser.add_argument("input", type=Path, help="JSON file containing labels and records")
    parser.add_argument("--bins", type=int, default=10, help="Equal-width calibration bins")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    labels = payload.get("labels") or sorted(
        {record["true"] for record in payload["records"]}
        | {record["pred"] for record in payload["records"]}
    )
    print(json.dumps(evaluate_records(payload["records"], labels, args.bins), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
