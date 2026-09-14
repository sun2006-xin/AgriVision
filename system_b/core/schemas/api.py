"""Dependency-free request schemas for System B boundary validation."""

import math


def require_object(payload):
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


def _number(value, name, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    if lower is not None and value < lower or upper is not None and value > upper:
        raise ValueError(f"{name} is out of range")
    return value


def validate_yolo_patch(payload):
    payload = require_object(payload)
    allowed = {"enabled", "conf_threshold", "iou_threshold", "dual_yolo_conf_high", "dual_yolo_conf_low"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError("unknown yolo parameter")
    result = dict(payload)
    if "enabled" in result and not isinstance(result["enabled"], bool):
        raise ValueError("enabled must be boolean")
    for key in ("conf_threshold", "iou_threshold", "dual_yolo_conf_high", "dual_yolo_conf_low"):
        if key in result:
            result[key] = _number(result[key], key, 0.0, 1.0)
    if result.get("dual_yolo_conf_low", 0.0) > result.get("dual_yolo_conf_high", 1.0):
        raise ValueError("dual_yolo_conf_low must not exceed dual_yolo_conf_high")
    return result


def validate_alert_patch(payload):
    payload = require_object(payload)
    allowed = {"webhook_url", "enabled", "cooldown_seconds"}
    if set(payload) - allowed:
        raise ValueError("unknown alert parameter")
    result = dict(payload)
    if "webhook_url" in result and (not isinstance(result["webhook_url"], str) or len(result["webhook_url"]) > 2048):
        raise ValueError("webhook_url must be a string no longer than 2048 characters")
    if "enabled" in result and not isinstance(result["enabled"], bool):
        raise ValueError("enabled must be boolean")
    if "cooldown_seconds" in result:
        value = result["cooldown_seconds"]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 86400:
            raise ValueError("cooldown_seconds must be an integer from 0 to 86400")
    return result


def validate_sync_request(payload, default_limit=50):
    payload = require_object(payload)
    allowed = {"limit"}
    if set(payload) - allowed:
        raise ValueError("unknown sync parameter")
    limit = payload.get("limit", default_limit)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer from 1 to 100")
    return {"limit": limit}
