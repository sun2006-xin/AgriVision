"""Bounded, opt-in transport for privacy-safe offline detection events."""

import hashlib
import json
from time import sleep
from urllib.parse import urlparse


def validate_sink_url(url):
    """Allow HTTPS sinks; allow HTTP only for local development endpoints."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("event sink URL is required")
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("event sink URL must use HTTP(S) and include a host")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("event sink URL must not contain credentials, query or fragment")
    if parsed.scheme == "http" and parsed.hostname.lower() not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("public event sinks must use HTTPS")
    try:
        parsed.port
    except ValueError as error:
        raise ValueError("event sink URL has an invalid port") from error
    return url.strip()


def build_batch_idempotency_key(events):
    """Build a stable, non-sensitive key for one ordered event batch."""
    serialized = json.dumps(
        list(events),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class EventTransport:
    def __init__(self, sink_url, sender, max_attempts=3, backoff_seconds=0.5):
        self.sink_url = validate_sink_url(sink_url)
        if not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds must not be negative")
        self.sender = sender
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds

    def sync(self, events, acknowledge):
        """Send one bounded batch and acknowledge events only after 2xx success."""
        events = list(events)
        if not events:
            return {"sent": 0, "attempts": 0}

        for attempt in range(1, self.max_attempts + 1):
            try:
                status_code = self.sender(self.sink_url, events)
                if 200 <= int(status_code) < 300:
                    for event in events:
                        acknowledge(event["event_id"])
                    return {"sent": len(events), "attempts": attempt}
            except (OSError, ValueError, TypeError):
                pass
            if attempt < self.max_attempts and self.backoff_seconds:
                sleep(self.backoff_seconds * attempt)
        return {"sent": 0, "attempts": self.max_attempts}
