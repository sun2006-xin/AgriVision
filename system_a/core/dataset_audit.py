"""Audit the reproducibility boundary of an AgriVision evaluation manifest."""

from collections import Counter
import hashlib
from pathlib import Path

from evaluation import SLICE_DIMENSIONS, validate_manifest


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coverage(records):
    coverage = {}
    for dimension in SLICE_DIMENSIONS:
        values = Counter(
            record["true"] if dimension == "disease" else record[dimension]
            for record in records
        )
        coverage[dimension] = dict(sorted(values.items()))
    return coverage


def audit_manifest(payload, image_root=None, verify_files=False):
    """Validate provenance and optionally verify every referenced image file.

    Structural violations raise ``ValueError``.  File-system failures are
    returned as a non-valid report so CI and local data collection can show all
    missing or mismatched samples in one pass.
    """
    if verify_files and image_root is None:
        raise ValueError("image_root is required when verify_files is enabled")

    manifest = validate_manifest(
        payload,
        require_metadata=True,
        require_provenance=True,
    )
    records = manifest["records"]
    missing = []
    hash_mismatch = []
    not_regular = []
    outside_root = []
    checked = 0

    if verify_files:
        root = Path(image_root).resolve()
        for record in records:
            candidate = (root / record["image"]).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                outside_root.append(record["id"])
                continue
            if not candidate.exists():
                missing.append(record["id"])
                continue
            if not candidate.is_file():
                not_regular.append(record["id"])
                continue
            checked += 1
            if _sha256(candidate) != record["image_sha256"]:
                hash_mismatch.append(record["id"])

    annotation_counts = Counter(record["annotation_status"] for record in records)
    file_checks = {
        "enabled": verify_files,
        "checked": checked,
        "missing": missing,
        "hash_mismatch": hash_mismatch,
        "not_regular": not_regular,
        "outside_root": outside_root,
    }
    valid = not any((missing, hash_mismatch, not_regular, outside_root))
    return {
        "valid": valid,
        "contract": {
            "strict_provenance": True,
            "required_annotation_statuses": ["verified", "adjudicated"],
        },
        "record_count": len(records),
        "coverage": _coverage(records),
        "annotation": {
            "evaluable": len(records),
            "by_status": dict(sorted(annotation_counts.items())),
        },
        "file_checks": file_checks,
    }
