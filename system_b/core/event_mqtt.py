"""Opt-in MQTT event contract with an injectable publisher."""

import json
import re
from time import sleep

try:
    from .event_transport import build_batch_idempotency_key
except ImportError:
    from event_transport import build_batch_idempotency_key
try:
    from .event_schema import project_event
except ImportError:
    from event_schema import project_event


_SAFE_TOPIC = re.compile(r"^[A-Za-z0-9._/-]{1,200}$")


def validate_mqtt_topic(topic):
    """Allow one concrete publish topic; reject wildcard subscriptions."""
    if (
        not isinstance(topic, str)
        or not _SAFE_TOPIC.fullmatch(topic)
        or topic.startswith("/")
        or topic.endswith("/")
        or "//" in topic
    ):
        raise ValueError("MQTT topic must be one concrete publish topic")
    return topic


def build_mqtt_payload(events):
    """Serialize a versioned batch without images, URLs, or credentials."""
    events = [project_event(event) for event in events]
    return json.dumps(
        {
            "schema_version": 1,
            "batch_id": build_batch_idempotency_key(events),
            "events": events,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class MqttEventTransport:
    """Publish one bounded batch; publisher must return True when accepted."""

    def __init__(self, topic, publisher, max_attempts=3, backoff_seconds=0.5):
        self.topic = validate_mqtt_topic(topic)
        if not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds must not be negative")
        self.publisher = publisher
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds

    def sync(self, events, acknowledge):
        events = [project_event(event) for event in events]
        if not events:
            return {"sent": 0, "attempts": 0}

        payload = build_mqtt_payload(events)
        for attempt in range(1, self.max_attempts + 1):
            try:
                accepted = self.publisher(self.topic, payload, 1, False)
                if accepted is True:
                    for event in events:
                        acknowledge(event["event_id"])
                    return {"sent": len(events), "attempts": attempt}
            except (OSError, RuntimeError, ValueError, TypeError):
                pass
            if attempt < self.max_attempts and self.backoff_seconds:
                sleep(self.backoff_seconds * attempt)
        return {
            "sent": 0,
            "attempts": self.max_attempts,
            "failure_type": "retry_exhausted",
        }
