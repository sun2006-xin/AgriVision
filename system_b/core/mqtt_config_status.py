"""Credential-free MQTT deployment configuration preflight."""

import os
import re
from urllib.parse import urlparse

try:
    from .mqtt_config import validate_broker_url
except ImportError:  # pragma: no cover - supports the app's direct-module imports
    from mqtt_config import validate_broker_url


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_SAFE_CLIENT_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_SAFE_TOPIC = re.compile(r"^[A-Za-z0-9._/-]{1,200}$")


def _valid_topic(topic):
    return (
        isinstance(topic, str)
        and bool(_SAFE_TOPIC.fullmatch(topic))
        and not topic.startswith("/")
        and not topic.endswith("/")
        and "//" not in topic
    )


def build_mqtt_config_status(broker_url, topic, client_id, username, password, ca_certs):
    """Return only fixed flags and issue codes; never return configuration values."""
    issues = []
    scheme = None
    parsed = urlparse(broker_url.strip()) if isinstance(broker_url, str) else None

    if not broker_url or not isinstance(broker_url, str) or not broker_url.strip():
        issues.append("broker_url_missing")
    else:
        try:
            validate_broker_url(broker_url)
            scheme = parsed.scheme
        except ValueError:
            if parsed and parsed.scheme == "mqtt" and parsed.hostname not in _LOCAL_HOSTS:
                issues.append("remote_tls_required")
            else:
                issues.append("broker_url_invalid")

    if not topic:
        issues.append("topic_missing")
    elif not _valid_topic(topic):
        issues.append("topic_invalid")

    if not client_id:
        issues.append("client_id_missing")
    elif not isinstance(client_id, str) or not _SAFE_CLIENT_ID.fullmatch(client_id):
        issues.append("client_id_invalid")

    has_username = isinstance(username, str) and bool(username)
    has_password = isinstance(password, str) and bool(password)
    credentials_configured = has_username and has_password
    if has_username != has_password:
        issues.append("credentials_incomplete")

    return {
        "configured": not issues,
        "scheme": scheme,
        "tls": scheme == "mqtts",
        "topic_configured": bool(topic) and "topic_invalid" not in issues,
        "client_id_configured": bool(client_id) and "client_id_invalid" not in issues,
        "credentials_configured": credentials_configured,
        "ca_configured": bool(ca_certs) and os.path.isfile(ca_certs),
        "issues": sorted(set(issues)),
    }
