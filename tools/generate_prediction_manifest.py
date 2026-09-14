"""Generate a strict prediction manifest with the current System A classifier."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_A_CORE = ROOT / "system_a" / "core"
if str(SYSTEM_A_CORE) not in sys.path:
    sys.path.insert(0, str(SYSTEM_A_CORE))

from classifier_inference import CLASS_NAMES, build_prediction_manifest


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the current System A ONNX classifier over a labeled manifest"
    )
    parser.add_argument("input", type=Path, help="Strict provenance manifest without prediction fields")
    parser.add_argument("--image-root", type=Path, required=True, help="Dataset root")
    parser.add_argument("--model", type=Path, required=True, help="ONNX classifier weights")
    parser.add_argument("--output", type=Path, required=True, help="Prediction manifest output path")
    parser.add_argument(
        "--provider",
        choices=("cpu", "cuda"),
        default="cpu",
        help="ONNX Runtime execution provider (default: cpu)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Bound ONNX intra-op threads (1-8, default: 1)",
    )
    args = parser.parse_args(argv)

    try:
        if not 1 <= args.threads <= 8:
            raise ValueError("--threads must be between 1 and 8")
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        model_hash = _sha256(args.model)

        import onnxruntime as ort

        available = ort.get_available_providers()
        if args.provider == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise ValueError("CUDAExecutionProvider is not available in this environment")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = args.threads
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(
            str(args.model),
            sess_options=session_options,
            providers=providers,
        )
        result = build_prediction_manifest(
            payload,
            image_root=args.image_root,
            session=session,
            expected_model_sha256=model_hash,
            expected_class_names=CLASS_NAMES,
        )
        rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        args.output.write_text(rendered, encoding="utf-8")
    except (OSError, TypeError, ValueError, json.JSONDecodeError, ImportError) as exc:
        parser.error(str(exc))

    print(f"PREDICTION_MANIFEST_OK records={len(result['records'])} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
