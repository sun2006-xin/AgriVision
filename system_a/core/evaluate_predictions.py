"""Evaluate a reproducible AgriVision prediction manifest without a model."""

import argparse
import json
from pathlib import Path

from evaluation import evaluate_manifest, evaluate_records


def main():
    parser = argparse.ArgumentParser(description="Evaluate an AgriVision prediction manifest")
    parser.add_argument("input", type=Path, help="JSON file containing labels, records and optional metadata")
    parser.add_argument("--bins", type=int, default=10, help="Equal-width calibration bins")
    parser.add_argument(
        "--strict-metadata",
        action="store_true",
        help="Require crop, lighting, device, source, split and unique record IDs",
    )
    parser.add_argument("--healthy-label", default=None, help="Label treated as field-screening negative")
    parser.add_argument("--positive-label", default=None, help="Optional one-vs-rest field-screening label")
    parser.add_argument("--field-source", default="field", help="Source value used for field FP/FN metrics")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    try:
        if args.strict_metadata:
            result = evaluate_manifest(
                payload,
                bins=args.bins,
                healthy_label=args.healthy_label,
                positive_label=args.positive_label,
                field_source=args.field_source,
            )
        else:
            labels = payload.get("labels") or sorted(
                {record["true"] for record in payload["records"]}
                | {record["pred"] for record in payload["records"]}
            )
            result = evaluate_records(
                payload["records"], labels, bins=args.bins,
                healthy_label=args.healthy_label,
                positive_label=args.positive_label,
                field_source=args.field_source,
            )
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
