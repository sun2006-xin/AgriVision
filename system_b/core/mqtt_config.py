"""Safe, credential-free MQTT broker URL validation."""

from urllib.parse import urlparse


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def validate_broker_url(url):
    """Require TLS for remote brokers and keep credentials outside the URL."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("MQTT broker URL is required")

    normalized = url.strip()
    parsed = urlparse(normalized)
    if parsed.scheme not in {"mqtt", "mqtts"} or not parsed.hostname:
        raise ValueError("MQTT broker URL must use mqtt:// or mqtts:// and include a host")
    if parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("MQTT broker URL must not contain credentials or a path")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("MQTT broker URL has an invalid port") from error
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("MQTT broker port must be between 1 and 65535")
    if parsed.scheme == "mqtt" and parsed.hostname.lower() not in _LOCAL_HOSTS:
        raise ValueError("remote MQTT brokers must use mqtts://")
    return normalized
