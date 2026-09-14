"""Shared System A classifier preprocessing and ONNX inference contract."""

import hashlib
import io
from functools import lru_cache

import numpy as np

from dataset_audit import audit_manifest, resolve_image_path
from evaluation import assess_probability_vector, validate_manifest


CLASS_NAMES = [
    "苹果黑星病",
    "玉米灰斑病",
    "玉米叶枯病",
    "玉米锈病",
    "健康",
    "马铃薯早疫病",
    "马铃薯晚疫病",
    "南瓜白粉病",
    "番茄细菌性斑点病",
    "番茄早疫病",
    "番茄晚疫病",
    "番茄花叶病毒病",
]

MAX_IMAGE_BYTES = 10 * 1024 * 1024


@lru_cache(maxsize=1)
def _classifier_transform():
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def preprocess_image(image_bytes: bytes):
    """Convert image bytes to the exact ResNet-18 ONNX input tensor."""
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _classifier_transform()(image).unsqueeze(0).numpy()


def _score_vector(raw_output, class_names):
    raw = raw_output[0] if isinstance(raw_output, (list, tuple)) else raw_output
    scores = np.asarray(raw, dtype=np.float64)
    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]
    if scores.ndim != 1 or scores.shape[0] != len(class_names):
        raise ValueError("classifier output class count does not match manifest labels")
    if not np.isfinite(scores).all():
        raise ValueError("classifier output contains non-finite scores")
    return scores


def predict_classifier(session, image_bytes, *, class_names=None, preprocess=preprocess_image,
                       input_name=None, output_name=None, threshold=0.55,
                       margin_threshold=0.15, entropy_threshold=0.75,
                       ood_max_probability=0.40):
    """Run one classifier inference and return reproducible prediction fields."""
    class_names = list(class_names or CLASS_NAMES)
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("class_names must be a non-empty list of unique values")
    if input_name is None:
        inputs = session.get_inputs()
        if not inputs:
            raise ValueError("classifier session has no inputs")
        input_name = inputs[0].name
    if output_name is None:
        outputs = session.get_outputs()
        if not outputs:
            raise ValueError("classifier session has no outputs")
        output_name = outputs[0].name

    tensor = preprocess(image_bytes)
    raw_output = session.run([output_name], {input_name: tensor})
    scores = _score_vector(raw_output, class_names)
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum()
    top_index = int(np.argmax(probabilities))
    probability_map = {
        label: float(probability)
        for label, probability in zip(class_names, probabilities)
    }
    confidence_info = assess_probability_vector(
        probability_map,
        threshold=threshold,
        margin_threshold=margin_threshold,
        entropy_threshold=entropy_threshold,
        ood_max_probability=ood_max_probability,
    )
    return {
        "class": class_names[top_index],
        "pred": class_names[top_index],
        "confidence": round(float(probabilities[top_index]), 4),
        "probabilities": probability_map,
        "confidence_info": confidence_info,
    }


class ClassifierRunner:
    """Small injectable wrapper used by the offline evaluation runner."""

    def __init__(self, session, *, class_names=None, preprocess=preprocess_image,
                 input_name=None, output_name=None, threshold=0.55,
                 margin_threshold=0.15, entropy_threshold=0.75,
                 ood_max_probability=0.40):
        self.session = session
        self.class_names = list(class_names or CLASS_NAMES)
        self.preprocess = preprocess
        self.input_name = input_name
        self.output_name = output_name
        self.thresholds = {
            "threshold": threshold,
            "margin_threshold": margin_threshold,
            "entropy_threshold": entropy_threshold,
            "ood_max_probability": ood_max_probability,
        }

    def predict(self, image_bytes):
        return predict_classifier(
            self.session,
            image_bytes,
            class_names=self.class_names,
            preprocess=self.preprocess,
            input_name=self.input_name,
            output_name=self.output_name,
            **self.thresholds,
        )


def build_prediction_manifest(payload, image_root, session, *, expected_model_sha256=None,
                              expected_class_names=None, preprocess=preprocess_image):
    """Run the current classifier over a provenance-complete labeled manifest."""
    manifest = validate_manifest(
        payload,
        require_metadata=True,
        require_provenance=True,
        require_predictions=False,
    )
    if expected_model_sha256 is not None:
        actual_manifest_hash = manifest["model"]["weights_sha256"]
        if actual_manifest_hash != expected_model_sha256:
            raise ValueError("manifest model hash does not match the selected model file")
    if expected_class_names is not None and list(manifest["labels"]) != list(expected_class_names):
        raise ValueError("manifest labels do not match the selected classifier class order")
    if any(
        field in record
        for record in manifest["records"]
        for field in ("pred", "confidence", "probabilities")
    ):
        raise ValueError("input manifest must not already contain prediction fields")

    audit = audit_manifest(payload, image_root=image_root, verify_files=True)
    if not audit["valid"]:
        raise ValueError("image file audit failed: " + str(audit["file_checks"]))

    runner = ClassifierRunner(
        session,
        class_names=manifest["labels"],
        preprocess=preprocess,
    )
    records = []
    for record in manifest["records"]:
        image_path = resolve_image_path(image_root, record["image"])
        if image_path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError("image exceeds the 10 MB classifier input limit")
        image_bytes = image_path.read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != record["image_sha256"]:
            raise ValueError("image changed after the provenance audit")
        prediction = runner.predict(image_bytes)
        enriched = dict(record)
        enriched.update({
            "pred": prediction["pred"],
            "confidence": prediction["confidence"],
            "probabilities": prediction["probabilities"],
        })
        records.append(enriched)

    result = dict(manifest)
    result["records"] = records
    return validate_manifest(
        result,
        require_metadata=True,
        require_provenance=True,
        require_predictions=True,
    )
