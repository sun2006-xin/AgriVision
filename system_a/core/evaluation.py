"""Dependency-free evaluation, calibration and uncertainty helpers.

The module deliberately does not load a model.  It evaluates an explicit
prediction manifest so that published numbers can be reproduced from labels,
predictions and capture metadata rather than from an opaque runtime state.
"""

from collections import defaultdict
import math
import re


MANIFEST_SCHEMA_VERSION = 1
REQUIRED_METADATA_FIELDS = ("crop", "lighting", "device", "source", "split")
SLICE_DIMENSIONS = ("crop", "disease", "lighting", "device", "source", "split")
_RECORD_FIELDS = {
    "id", "true", "pred", "confidence", "probabilities",
    "crop", "lighting", "device", "source", "split", "group_id",
    "field_true", "field_pred", "model", "notes", "image", "image_sha256",
    "annotation_status", "annotation_source",
}
_MODEL_FIELDS = {"name", "version", "weights_sha256", "code_revision"}
_REQUIRED_MODEL_FIELDS = ("name", "version", "weights_sha256")
_REQUIRED_DATASET_FIELDS = ("name", "version", "label_policy")
_ALLOWED_SPLITS = {"train", "validation", "test"}
_ALLOWED_ANNOTATION_STATUSES = {"verified", "adjudicated"}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_FIELD_THRESHOLDS = (0.25, 0.40, 0.55, 0.70, 0.85)
ROBUSTNESS_DIMENSIONS = ("lighting", "device")


def _safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _validate_labels(labels):
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("labels must be a non-empty list of unique values")
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("labels must contain non-empty strings")
    return list(labels)


def classification_metrics(y_true, y_pred, labels):
    """Calculate accuracy, confusion matrix and one-vs-rest class metrics."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    labels = _validate_labels(labels)

    label_set = set(labels)
    if any(value not in label_set for value in [*y_true, *y_pred]):
        raise ValueError("predictions contain a label outside labels")

    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for truth, prediction in zip(y_true, y_pred):
        matrix[index[truth]][index[prediction]] += 1

    per_class = {}
    weighted = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for label in labels:
        position = index[label]
        true_positive = matrix[position][position]
        false_positive = sum(matrix[row][position] for row in range(len(labels))) - true_positive
        false_negative = sum(matrix[position]) - true_positive
        support = sum(matrix[position])
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        weighted["precision"] += precision * support
        weighted["recall"] += recall * support
        weighted["f1"] += f1 * support

    correct = sum(matrix[position][position] for position in range(len(labels)))
    total = len(y_true)
    return {
        "support": total,
        "correct": correct,
        "accuracy": _safe_ratio(correct, total),
        "macro_precision": _safe_ratio(sum(item["precision"] for item in per_class.values()), len(labels)),
        "macro_recall": _safe_ratio(sum(item["recall"] for item in per_class.values()), len(labels)),
        "macro_f1": _safe_ratio(sum(item["f1"] for item in per_class.values()), len(labels)),
        "weighted_precision": _safe_ratio(weighted["precision"], total),
        "weighted_recall": _safe_ratio(weighted["recall"], total),
        "weighted_f1": _safe_ratio(weighted["f1"], total),
        "labels": list(labels),
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def _validated_confidences(confidences, correct):
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    values = []
    for confidence in confidences:
        value = _finite(confidence, "confidence")
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be a finite value between 0 and 1")
        values.append(value)
    return values, [bool(value) for value in correct]


def reliability_bins(confidences, correct, bins=10):
    """Return equal-width reliability bins, including empty bins."""
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    confidences, correct = _validated_confidences(confidences, correct)
    buckets = [[] for _ in range(bins)]
    for confidence, is_correct in zip(confidences, correct):
        bucket = min(int(confidence * bins), bins - 1)
        buckets[bucket].append((confidence, is_correct))

    result = []
    for position, bucket in enumerate(buckets):
        lower = position / bins
        upper = (position + 1) / bins
        count = len(bucket)
        mean_confidence = _safe_ratio(sum(item[0] for item in bucket), count)
        accuracy = _safe_ratio(sum(int(item[1]) for item in bucket), count)
        result.append({
            "bin": position,
            "lower": lower,
            "upper": upper,
            "count": count,
            "mean_confidence": mean_confidence,
            "accuracy": accuracy,
            "gap": abs(mean_confidence - accuracy) if count else 0.0,
        })
    return result


def expected_calibration_error(confidences, correct, bins=10):
    """Calculate ECE using equal-width confidence bins."""
    confidences, correct = _validated_confidences(confidences, correct)
    if bins <= 0:
        raise ValueError("bins must be positive")
    if not confidences:
        return 0.0
    return sum(
        bucket["count"] / len(confidences) * bucket["gap"]
        for bucket in reliability_bins(confidences, correct, bins)
        if bucket["count"]
    )


def _top1_brier_score(confidences, correct):
    return _safe_ratio(
        sum((confidence - int(is_correct)) ** 2 for confidence, is_correct in zip(confidences, correct)),
        len(confidences),
    )


def _multiclass_brier_score(records, labels):
    total = 0.0
    for record in records:
        probabilities = record["probabilities"]
        for label in labels:
            expected = 1.0 if record["true"] == label else 0.0
            total += (float(probabilities[label]) - expected) ** 2
    return _safe_ratio(total, len(records))


def calibration_metrics(records, labels, bins=10):
    """Return ECE, Brier score and data for a reliability diagram.

    Full probability vectors produce a multiclass Brier score.  Legacy
    manifests with only top-1 confidence use an explicitly named top-1 score.
    Neither score claims that the model has been calibrated; it measures how
    well the submitted confidence values agree with the supplied labels.
    """
    confidences = [record["confidence"] for record in records]
    correct = [record["true"] == record["pred"] for record in records]
    ece = expected_calibration_error(confidences, correct, bins)
    has_full_probabilities = bool(records) and all(
        isinstance(record.get("probabilities"), dict)
        and set(record["probabilities"]) == set(labels)
        for record in records
    )
    if has_full_probabilities:
        brier_score = _multiclass_brier_score(records, labels)
        brier_mode = "multiclass"
    else:
        brier_score = _top1_brier_score(
            [_finite(record["confidence"], "confidence") for record in records], correct
        )
        brier_mode = "top1"
    return {
        "ece": ece,
        "brier_score": brier_score,
        "brier_mode": brier_mode,
        "reliability_bins": reliability_bins(confidences, correct, bins),
        "calibrated": False,
        "interpretation": "校准评估，不代表模型概率已经校准",
    }


def _validate_text(value, name, max_length):
    if not isinstance(value, str) or not value.strip() or len(value) > max_length or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string of at most {max_length} characters")


def _validate_sha256(value, name):
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _validate_image_path(value):
    _validate_text(value, "image", 512)
    if (
        value.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:", value)
        or "://" in value
        or "\\" in value
        or ":" in value
    ):
        raise ValueError("image must be a portable relative path")
    parts = value.split("/")
    if not parts or any(not part or part in {".", ".."} for part in parts):
        raise ValueError("image must not contain empty, dot or parent path components")


def _validate_dataset_metadata(dataset, require_provenance=False):
    if dataset is None:
        if require_provenance:
            raise ValueError("manifest dataset metadata is required for strict provenance")
        return
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be a JSON object")
    if require_provenance:
        missing = sorted(field for field in _REQUIRED_DATASET_FIELDS if field not in dataset)
        if missing:
            raise ValueError("dataset is missing required field: " + missing[0])
    for field in _REQUIRED_DATASET_FIELDS:
        if field in dataset:
            _validate_text(dataset[field], f"dataset.{field}", 512)


def _validate_model_metadata(model, require_provenance=False):
    if model is None:
        if require_provenance:
            raise ValueError("manifest model metadata is required for strict provenance")
        return
    if not isinstance(model, dict):
        raise ValueError("model must be a JSON object")
    unknown = sorted(set(model) - _MODEL_FIELDS)
    if unknown:
        raise ValueError("unknown model field: " + unknown[0])
    if require_provenance:
        missing = sorted(field for field in _REQUIRED_MODEL_FIELDS if field not in model)
        if missing:
            raise ValueError("model is missing required field: " + missing[0])
    for field in ("name", "version", "code_revision"):
        if field in model:
            _validate_text(model[field], f"model.{field}", 256)
    if "weights_sha256" in model:
        _validate_sha256(model["weights_sha256"], "model.weights_sha256")


def _validate_provenance_fields(record, require_provenance=False):
    has_image = "image" in record
    has_digest = "image_sha256" in record
    if has_image != has_digest:
        raise ValueError("image and image_sha256 must be provided together")
    if has_image:
        _validate_image_path(record["image"])
        _validate_sha256(record["image_sha256"], "image_sha256")

    if "annotation_status" in record:
        if record["annotation_status"] not in _ALLOWED_ANNOTATION_STATUSES:
            raise ValueError("annotation_status must be verified or adjudicated")
    if "annotation_source" in record:
        _validate_text(record["annotation_source"], "annotation_source", 256)

    if require_provenance:
        required = {"image", "image_sha256", "group_id", "annotation_status", "annotation_source"}
        missing = sorted(field for field in required if field not in record)
        if missing:
            raise ValueError("record is missing provenance field: " + missing[0])
        if record["annotation_status"] not in _ALLOWED_ANNOTATION_STATUSES:
            raise ValueError("annotation_status must be verified or adjudicated")
        if record["split"] not in _ALLOWED_SPLITS:
            raise ValueError("split must be one of train, validation or test")


def _validate_record(record, labels=None, require_metadata=False, require_id=False,
                     require_provenance=False, require_predictions=True):
    if not isinstance(record, dict):
        raise ValueError("each record must be a JSON object")
    unknown = sorted(set(record) - _RECORD_FIELDS)
    if unknown:
        raise ValueError("unknown record field: " + unknown[0])
    required = {"true"}
    if require_predictions:
        required.update({"pred", "confidence"})
    if require_id:
        required.add("id")
    if require_metadata:
        required.update(REQUIRED_METADATA_FIELDS)
    missing = sorted(field for field in required if field not in record)
    if missing:
        raise ValueError("record is missing required field: " + missing[0])
    for field in ("true", "pred"):
        if field in record and (not isinstance(record[field], str) or not record[field].strip()):
            raise ValueError(f"{field} must be a non-empty string")
    if "confidence" in record:
        confidence = _finite(record["confidence"], "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
    if "id" in record and (
        not isinstance(record["id"], str) or not record["id"].strip()
        or len(record["id"]) > 128 or "/" in record["id"] or "\\" in record["id"]
    ):
        raise ValueError("id must be a safe non-empty identifier")
    for field in REQUIRED_METADATA_FIELDS:
        if field in record and (
            not isinstance(record[field], str) or not record[field].strip() or len(record[field]) > 128
        ):
            raise ValueError(f"{field} must be a non-empty string")
    if "group_id" in record and (
        not isinstance(record["group_id"], str) or not record["group_id"].strip()
        or len(record["group_id"]) > 128
    ):
        raise ValueError("group_id must be a non-empty string")
    if "field_true" in record or "field_pred" in record:
        if not isinstance(record.get("field_true"), bool) or not isinstance(record.get("field_pred"), bool):
            raise ValueError("field_true and field_pred must be provided together as booleans")
    if "probabilities" in record:
        probabilities = record["probabilities"]
        if not isinstance(probabilities, dict) or not probabilities:
            raise ValueError("probabilities must be a non-empty object")
        if labels is not None and not set(probabilities).issubset(set(labels)):
            raise ValueError("probabilities contain a label outside labels")
        probability_sum = 0.0
        for label, value in probabilities.items():
            if not isinstance(label, str):
                raise ValueError("probability labels must be strings")
            value = _finite(value, "probability")
            if not 0.0 <= value <= 1.0:
                raise ValueError("probabilities must be between 0 and 1")
            probability_sum += value
        if abs(probability_sum - 1.0) > 0.001:
            raise ValueError("probabilities must sum to 1")
    _validate_provenance_fields(record, require_provenance=require_provenance)


def validate_manifest(payload, require_metadata=True, require_provenance=False,
                      require_predictions=True):
    """Validate and return a JSON-safe evaluation manifest copy.

    Strict manifests require crop, lighting, device, source and split for
    every sample.  ``group_id`` prevents frames from one capture sequence
    leaking across train/validation/test splits.
    """
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {MANIFEST_SCHEMA_VERSION}")
    labels = payload.get("labels")
    if not isinstance(labels, list):
        raise ValueError("manifest labels must be a list")
    labels = _validate_labels(labels)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest records must be a non-empty list")
    if "dataset" in payload and payload["dataset"] is None:
        raise ValueError("dataset must be a JSON object")
    if "model" in payload and payload["model"] is None:
        raise ValueError("model must be a JSON object")
    _validate_dataset_metadata(payload.get("dataset"), require_provenance=require_provenance)
    _validate_model_metadata(payload.get("model"), require_provenance=require_provenance)

    allowed_manifest_fields = {"schema_version", "dataset", "model", "labels", "records"}
    unknown = sorted(set(payload) - allowed_manifest_fields)
    if unknown:
        raise ValueError("unknown manifest field: " + unknown[0])

    seen_ids = set()
    seen_image_hashes = set()
    groups = defaultdict(set)
    for record in records:
        _validate_record(
            record,
            labels,
            require_metadata=require_metadata,
            require_id=True,
            require_provenance=require_provenance,
            require_predictions=require_predictions,
        )
        if record["true"] not in labels or (
            "pred" in record and record["pred"] not in labels
        ):
            raise ValueError("record labels must be declared in manifest labels")
        if record["id"] in seen_ids:
            raise ValueError("record ids must be unique")
        seen_ids.add(record["id"])
        if "probabilities" in record and set(record["probabilities"]) != set(labels):
            raise ValueError("strict manifest probabilities must include every declared label")
        if require_provenance:
            image_hash = record["image_sha256"]
            if image_hash in seen_image_hashes:
                raise ValueError("image_sha256 must be unique across evaluation records")
            seen_image_hashes.add(image_hash)
        if record.get("group_id"):
            groups[record["group_id"]].add(record["split"])
    if any(len(splits) > 1 for splits in groups.values()):
        raise ValueError("group_id must not span multiple dataset splits")

    import json
    normalized = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": payload.get("dataset", {}),
        "labels": labels,
        "records": records,
    }
    if "model" in payload:
        normalized["model"] = payload["model"]
    return json.loads(json.dumps(normalized, ensure_ascii=False))


def _resolve_healthy_label(labels, healthy_label):
    if healthy_label:
        return healthy_label
    for candidate in ("健康", "healthy", "normal"):
        if candidate in labels:
            return candidate
    return None


def _binary_metrics(pairs):
    true_positive = sum(truth and prediction for truth, prediction in pairs)
    false_positive = sum(not truth and prediction for truth, prediction in pairs)
    false_negative = sum(truth and not prediction for truth, prediction in pairs)
    true_negative = sum(not truth and not prediction for truth, prediction in pairs)
    positive_support = true_positive + false_negative
    negative_support = true_negative + false_positive
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "positive_support": positive_support,
        "negative_support": negative_support,
        "false_positive_rate": _safe_ratio(false_positive, negative_support),
        "false_negative_rate": _safe_ratio(false_negative, positive_support),
        "precision": _safe_ratio(true_positive, true_positive + false_positive),
        "recall": _safe_ratio(true_positive, positive_support),
        "accuracy": _safe_ratio(true_positive + true_negative, len(pairs)),
    }


def _validated_thresholds(thresholds):
    values = list(DEFAULT_FIELD_THRESHOLDS if thresholds is None else thresholds)
    if not values:
        raise ValueError("thresholds must contain at least one value")
    normalized = []
    for threshold in values:
        if isinstance(threshold, bool):
            raise ValueError("thresholds must be numeric values between 0 and 1")
        threshold = _finite(threshold, "threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("thresholds must be between 0 and 1")
        normalized.append(threshold)
    if len(set(normalized)) != len(normalized):
        raise ValueError("thresholds must be unique")
    return sorted(normalized)


def field_error_rates(records, healthy_label=None, positive_label=None):
    """Calculate field-screening false-positive and false-negative rates.

    Records may provide explicit ``field_true``/``field_pred`` booleans.  If
    they do not, a binary screen is derived from a declared healthy label (or
    a one-vs-rest positive label).  No metric is emitted as available without
    an explicit binary definition.
    """
    records = list(records)
    result = {
        "available": False,
        "sample_count": len(records),
        "reason": "需要 field_true/field_pred 或 healthy_label/positive_label",
    }
    if not records:
        result["reason"] = "没有 source=field 的记录"
        return result
    labels = sorted({record.get("true") for record in records if isinstance(record.get("true"), str)})
    healthy_label = _resolve_healthy_label(labels, healthy_label)
    if positive_label is None and healthy_label is None and not all(
        "field_true" in record and "field_pred" in record for record in records
    ):
        return result

    pairs = []
    for record in records:
        if "field_true" in record and "field_pred" in record:
            truth = record["field_true"]
            prediction = record["field_pred"]
        elif positive_label is not None:
            truth = record["true"] == positive_label
            prediction = record["pred"] == positive_label
        else:
            truth = record["true"] != healthy_label
            prediction = record["pred"] != healthy_label
        pairs.append((bool(truth), bool(prediction)))

    metrics = _binary_metrics(pairs)
    result.update({
        "available": True,
        "positive_label": positive_label or f"非{healthy_label}",
        "healthy_label": healthy_label,
        "sample_count": len(pairs),
        **metrics,
    })
    result.pop("reason", None)
    return result


def field_threshold_curve(records, healthy_label=None, positive_label=None, thresholds=None):
    """Measure field FP/FN trade-offs at probability thresholds.

    The curve is available only when every record has a complete probability
    vector.  This keeps argmax labels from being misrepresented as threshold
    evidence.
    """
    records = list(records)
    result = {
        "available": False,
        "sample_count": len(records),
        "reason": "需要每条记录的完整 probabilities 向量",
    }
    if not records:
        result["reason"] = "没有 source=field 的记录"
        return result
    if not all(isinstance(record.get("probabilities"), dict) for record in records):
        return result

    labels = sorted({record.get("true") for record in records if isinstance(record.get("true"), str)})
    healthy_label = _resolve_healthy_label(labels, healthy_label)
    if positive_label is not None:
        if positive_label not in labels:
            result["reason"] = "positive_label 不在记录标签中"
            return result
        score_name = positive_label
    else:
        if healthy_label is None:
            result["reason"] = "需要 healthy_label 或 positive_label"
            return result
        score_name = f"非{healthy_label}"

    scored = []
    for record in records:
        probabilities = record["probabilities"]
        if positive_label is not None:
            if positive_label not in probabilities:
                result["reason"] = "probabilities 缺少 positive_label"
                return result
            score = probabilities[positive_label]
            truth = (
                bool(record["field_true"])
                if "field_true" in record
                else record["true"] == positive_label
            )
        else:
            if healthy_label not in probabilities:
                result["reason"] = "probabilities 缺少 healthy_label"
                return result
            score = 1.0 - probabilities[healthy_label]
            truth = (
                bool(record["field_true"])
                if "field_true" in record
                else record["true"] != healthy_label
            )
        score = _finite(score, "positive probability")
        if not 0.0 <= score <= 1.0:
            raise ValueError("positive probability must be between 0 and 1")
        scored.append((truth, score))

    curve = []
    for threshold in _validated_thresholds(thresholds):
        metrics = _binary_metrics([
            (truth, score >= threshold) for truth, score in scored
        ])
        metrics["threshold"] = threshold
        curve.append(metrics)
    return {
        "available": True,
        "sample_count": len(scored),
        "healthy_label": healthy_label,
        "positive_label": positive_label or f"非{healthy_label}",
        "score": "probability_of_positive" if positive_label else "probability_of_non_healthy",
        "thresholds": curve,
        "interpretation": "阈值敏感性分析，不代表已完成业务阈值验收",
    }


def _group_summary(records, labels, bins):
    y_true = [record["true"] for record in records]
    y_pred = [record["pred"] for record in records]
    metrics = classification_metrics(y_true, y_pred, labels)
    calibration = calibration_metrics(records, labels, bins)
    metrics["ece"] = calibration["ece"]
    metrics["brier_score"] = calibration["brier_score"]
    metrics["brier_mode"] = calibration["brier_mode"]
    return metrics


def slice_robustness(records, labels, bins=10, dimensions=ROBUSTNESS_DIMENSIONS):
    """Summarize unpaired metric gaps across lighting and device slices."""
    records = list(records)
    labels = _validate_labels(labels)
    result = {}
    for dimension in dimensions:
        grouped = defaultdict(list)
        for record in records:
            grouped[str(record.get(dimension, "unknown"))].append(record)
        summaries = {
            value: _group_summary(group, labels, bins)
            for value, group in sorted(grouped.items())
        }
        entries = list(summaries.items())
        accuracy_values = [summary["accuracy"] for _, summary in entries]
        f1_values = [summary["macro_f1"] for _, summary in entries]
        best_accuracy = max(entries, key=lambda item: (item[1]["accuracy"], item[0])) if entries else None
        worst_accuracy = min(entries, key=lambda item: (item[1]["accuracy"], item[0])) if entries else None
        result[dimension] = {
            "comparable": len(entries) >= 2,
            "slice_count": len(entries),
            "support": {value: summary["support"] for value, summary in entries},
            "accuracy_by_slice": {value: summary["accuracy"] for value, summary in entries},
            "macro_f1_by_slice": {value: summary["macro_f1"] for value, summary in entries},
            "accuracy_gap": max(accuracy_values) - min(accuracy_values) if entries else 0.0,
            "macro_f1_gap": max(f1_values) - min(f1_values) if entries else 0.0,
            "best_slice": best_accuracy[0] if best_accuracy else None,
            "worst_slice": worst_accuracy[0] if worst_accuracy else None,
            "interpretation": "unpaired slice comparison; not a paired lighting or device transfer experiment",
        }
        if len(entries) < 2:
            result[dimension]["reason"] = "至少需要两个不同切片才能比较差异"
    return result


def evaluate_records(records, labels, bins=10, healthy_label=None, positive_label=None,
                     field_source="field", field_thresholds=None):
    """Evaluate records and return overall, slice and field-screening metrics.

    This function retains compatibility with the original three-column
    ``true/pred/confidence`` records.  Use :func:`evaluate_manifest` for the
    strict, metadata-complete public evaluation contract.
    """
    records = list(records)
    labels = _validate_labels(labels)
    for record in records:
        _validate_record(record, labels, require_metadata=False, require_id=False)
        if record["true"] not in labels or record["pred"] not in labels:
            raise ValueError("record labels must be declared in labels")

    result = _group_summary(records, labels, bins)
    result["calibration"] = calibration_metrics(records, labels, bins)

    field_records = [record for record in records if record.get("source") == field_source]
    result["field_error_rates"] = field_error_rates(
        field_records, healthy_label=healthy_label, positive_label=positive_label
    )
    result["field_threshold_curve"] = field_threshold_curve(
        field_records,
        healthy_label=healthy_label,
        positive_label=positive_label,
        thresholds=field_thresholds,
    )

    slices = {}
    for dimension in SLICE_DIMENSIONS:
        grouped = defaultdict(list)
        for record in records:
            value = record["true"] if dimension == "disease" else record.get(dimension, "unknown")
            grouped[str(value)].append(record)
        slices[dimension] = {
            value: _group_summary(group, labels, bins)
            for value, group in sorted(grouped.items())
        }
    result["slices"] = slices
    result["robustness"] = slice_robustness(records, labels, bins=bins)
    return result


def evaluate_manifest(payload, bins=10, healthy_label=None, positive_label=None,
                      field_source="field", require_provenance=False, field_thresholds=None):
    """Strictly validate and evaluate a metadata-complete manifest."""
    manifest = validate_manifest(
        payload,
        require_metadata=True,
        require_provenance=require_provenance,
    )
    result = evaluate_records(
        manifest["records"], manifest["labels"], bins=bins,
        healthy_label=healthy_label, positive_label=positive_label,
        field_source=field_source,
        field_thresholds=field_thresholds,
    )
    result["manifest"] = {
        "schema_version": manifest["schema_version"],
        "dataset": manifest.get("dataset", {}),
        "record_count": len(manifest["records"]),
        "dimensions": list(REQUIRED_METADATA_FIELDS),
        "provenance": {
            "required": require_provenance,
            "file_identity": require_provenance,
            "annotation_statuses": sorted(_ALLOWED_ANNOTATION_STATUSES),
        },
    }
    if "model" in manifest:
        result["manifest"]["model"] = manifest["model"]
    return result


def assess_confidence(score, threshold=0.55):
    """Return the original simple confidence marker for compatibility."""
    score = _finite(score, "score")
    threshold = _finite(threshold, "threshold")
    if not 0.0 <= score <= 1.0 or not 0.0 <= threshold <= 1.0:
        raise ValueError("score and threshold must be between 0 and 1")
    if score < threshold:
        return {"uncertain": True, "band": "uncertain"}
    if score >= 0.75:
        return {"uncertain": False, "band": "high"}
    return {"uncertain": False, "band": "medium"}


def assess_probability_vector(probabilities, threshold=0.55, margin_threshold=0.15,
                              entropy_threshold=0.75, ood_max_probability=0.40):
    """Derive transparent uncertainty/OOD *screening* evidence.

    This is an abstention heuristic over model scores, not a learned OOD
    detector.  It is intentionally labelled ``ood_suspected`` so callers do
    not mistake a softmax score for a calibrated probability.
    """
    if isinstance(probabilities, dict):
        labels = list(probabilities)
        values = [probabilities[label] for label in labels]
    else:
        labels = []
        values = list(probabilities) if probabilities is not None else []
    if not values:
        raise ValueError("probabilities must be non-empty")
    values = [_finite(value, "probability") for value in values]
    if any(value < 0.0 for value in values):
        raise ValueError("probabilities must be non-negative")
    total = sum(values)
    if abs(total - 1.0) > 0.001:
        raise ValueError("probabilities must sum to 1")
    if labels and len(set(labels)) != len(labels):
        raise ValueError("probability labels must be unique")
    if len(values) > 1:
        entropy = -sum(value * math.log(value) for value in values if value > 0)
        entropy /= math.log(len(values))
    else:
        entropy = 0.0
    ordered = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
    top_index, top_probability = ordered[0]
    second_probability = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = top_probability - second_probability
    reasons = []
    if top_probability < threshold:
        reasons.append("low_confidence")
    if margin < margin_threshold and len(values) > 1:
        reasons.append("low_margin")
    if entropy >= entropy_threshold and len(values) > 1:
        reasons.append("high_entropy")
    ood_suspected = (
        len(values) > 1
        and top_probability < ood_max_probability
        and entropy >= entropy_threshold
    )
    if ood_suspected:
        reasons.append("ood_heuristic")
    uncertain = bool(reasons)
    undetermined = (
        len(values) > 1
        and top_probability < threshold
        and entropy >= entropy_threshold
    )
    if ood_suspected:
        band = "ood"
    elif uncertain:
        band = "uncertain"
    elif top_probability >= 0.75 and margin >= 0.40:
        band = "high"
    else:
        band = "medium"
    return {
        "uncertain": uncertain,
        "undetermined": undetermined,
        "abstain": uncertain,
        "band": band,
        "ood_suspected": ood_suspected,
        "decision_status": (
            "ood_suspected" if ood_suspected
            else "undetermined" if undetermined
            else "uncertain" if uncertain
            else "known"
        ),
        "uncertainty_reason": ",".join(reasons) or "none",
        "top_probability": top_probability,
        "second_probability": second_probability,
        "margin": margin,
        "normalized_entropy": entropy,
        "top_label": labels[top_index] if labels else None,
    }
