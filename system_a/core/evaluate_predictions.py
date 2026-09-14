"""Evaluate a reproducible AgriVision prediction manifest without a model."""

import argparse
import json
from pathlib import Path

from dataset_audit import audit_manifest
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
    parser.add_argument(
        "--strict-provenance",
        action="store_true",
        help="Also require dataset/model identity, image hashes, capture groups and reviewed labels",
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="With --strict-provenance, verify image files against image_sha256",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Dataset root used with --check-files",
    )
    parser.add_argument("--healthy-label", default=None, help="Label treated as field-screening negative")
    parser.add_argument("--positive-label", default=None, help="Optional one-vs-rest field-screening label")
    parser.add_argument("--field-source", default="field", help="Source value used for field FP/FN metrics")
    args = parser.parse_args()

    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        if args.check_files and not args.strict_provenance:
            raise ValueError("--check-files requires --strict-provenance")
        if args.strict_provenance:
            audit = audit_manifest(
                payload,
                image_root=args.image_root,
                verify_files=args.check_files,
            )
            if not audit["valid"]:
                raise ValueError("image file audit failed: " + json.dumps(audit["file_checks"], ensure_ascii=False))
        if args.strict_metadata or args.strict_provenance:
            result = evaluate_manifest(
                payload,
                bins=args.bins,
                healthy_label=args.healthy_label,
                positive_label=args.positive_label,
                field_source=args.field_source,
                require_provenance=args.strict_provenance,
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
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
