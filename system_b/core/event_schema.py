"""Versioned, privacy-safe event envelope for edge synchronization."""

from datetime import datetime, timezone
import math
import re
import uuid


SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_EVENT_FIELDS = {"schema_version", "event_id", "created_at", "camera_id", "payload"}
_PAYLOAD_FIELDS = {
    "level",
    "level_code",
    "disease_count",
    "white_count",
    "disease_ratio",
    "white_ratio",
    "green_ratio",
}


def project_event(event):
    """Return only the versioned, non-credential event contract fields."""
    if not isinstance(event, dict):
        raise ValueError("event must be a JSON object")
    projected = {key: event[key] for key in _EVENT_FIELDS if key in event}
    payload = event.get("payload")
    if isinstance(payload, dict):
        projected["payload"] = {
            key: payload[key] for key in _PAYLOAD_FIELDS if key in payload
        }
    return projected


def build_detection_event(camera_id, result):
    if not _SAFE_ID.fullmatch(camera_id):
        raise ValueError("camera_id contains unsupported characters")

    payload = {
        "level": result.get("level", "正常"),
        "level_code": int(result.get("level_code", 0)),
        "disease_count": int(result.get("disease_count", 0)),
        "white_count": int(result.get("white_count", 0)),
        "disease_ratio": float(result.get("disease_ratio", 0.0)),
        "white_ratio": float(result.get("white_ratio", 0.0)),
        "green_ratio": float(result.get("green_ratio", 0.0)),
    }
    if payload["level_code"] not in range(4) or any(
        value < 0 for key, value in payload.items() if key.endswith("count")
    ):
        raise ValueError("detection counts or level are invalid")
    if any(
        not math.isfinite(payload[key]) or not 0.0 <= payload[key] <= 1.0
        for key in ("disease_ratio", "white_ratio", "green_ratio")
    ):
        raise ValueError("detection ratios must be finite values between 0 and 1")

    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "camera_id": camera_id,
        "payload": payload,
    }
