"""Audit an AgriVision evaluation manifest and its local image files."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_A_CORE = ROOT / "system_a" / "core"
if str(SYSTEM_A_CORE) not in sys.path:
    sys.path.insert(0, str(SYSTEM_A_CORE))

from dataset_audit import audit_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit an AgriVision evaluation manifest")
    parser.add_argument("input", type=Path, help="JSON evaluation manifest")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Local dataset root used to resolve each safe relative image path",
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Verify that image files exist and match image_sha256",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args(argv)

    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        report = audit_manifest(
            payload,
            image_root=args.image_root,
            verify_files=args.check_files,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        report = {"valid": False, "error": str(exc)}
        exit_code = 2
    else:
        exit_code = 0 if report["valid"] else 1

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
