"""Dependency-free evaluation and uncertainty helpers for model predictions."""

import math


def _safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def classification_metrics(y_true, y_pred, labels):
    """Calculate accuracy, confusion matrix and one-vs-rest class metrics."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("labels must be a non-empty list of unique values")

    label_set = set(labels)
    if any(value not in label_set for value in [*y_true, *y_pred]):
        raise ValueError("predictions contain a label outside labels")

    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for truth, prediction in zip(y_true, y_pred):
        matrix[index[truth]][index[prediction]] += 1

    per_class = {}
    for label in labels:
        position = index[label]
        true_positive = matrix[position][position]
        false_positive = sum(matrix[row][position] for row in range(len(labels))) - true_positive
        false_negative = sum(matrix[position]) - true_positive
        support = sum(matrix[position])
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": _safe_ratio(2 * precision * recall, precision + recall),
            "support": support,
        }

    correct = sum(matrix[position][position] for position in range(len(labels)))
    return {
        "support": len(y_true),
        "correct": correct,
        "accuracy": _safe_ratio(correct, len(y_true)),
        "macro_precision": _safe_ratio(sum(item["precision"] for item in per_class.values()), len(labels)),
        "macro_recall": _safe_ratio(sum(item["recall"] for item in per_class.values()), len(labels)),
        "macro_f1": _safe_ratio(sum(item["f1"] for item in per_class.values()), len(labels)),
        "labels": list(labels),
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def expected_calibration_error(confidences, correct, bins=10):
    """Calculate ECE using equal-width confidence bins."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    if bins <= 0:
        raise ValueError("bins must be positive")
    if not confidences:
        return 0.0

    buckets = [[] for _ in range(bins)]
    for confidence, is_correct in zip(confidences, correct):
        if not math.isfinite(float(confidence)) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be a finite value between 0 and 1")
        bucket = min(int(confidence * bins), bins - 1)
        buckets[bucket].append((float(confidence), bool(is_correct)))

    total = len(confidences)
    return sum(
        len(bucket) / total
        * abs(sum(confidence for confidence, _ in bucket) / len(bucket)
              - sum(int(is_correct) for _, is_correct in bucket) / len(bucket))
        for bucket in buckets if bucket
    )


def assess_confidence(score, threshold=0.55):
    """Return an explicit uncertainty marker for user-facing predictions."""
    if not math.isfinite(float(score)) or not 0.0 <= score <= 1.0:
        raise ValueError("score must be a finite value between 0 and 1")
    if score < threshold:
        return {"uncertain": True, "band": "uncertain"}
    if score >= 0.75:
        return {"uncertain": False, "band": "high"}
    return {"uncertain": False, "band": "medium"}
