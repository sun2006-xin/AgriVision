"""Optional Paho MQTT runtime; no broker connection occurs at import time."""

import re
from urllib.parse import urlparse

from event_mqtt import MqttEventTransport
from mqtt_config import validate_broker_url


_SAFE_CLIENT_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


def validate_client_id(client_id):
    if not isinstance(client_id, str) or not _SAFE_CLIENT_ID.fullmatch(client_id):
        raise ValueError("MQTT client_id contains unsupported characters")
    return client_id


def create_paho_transport(
    broker_url,
    topic,
    client_id,
    username=None,
    password=None,
    ca_certs=None,
    max_attempts=3,
    backoff_seconds=0.5,
):
    """Create a connected Paho transport and a close callback.

    Credentials are accepted only as separate arguments and are never logged.
    The optional paho-mqtt dependency is imported only when this function runs.
    """
    normalized_url = validate_broker_url(broker_url)
    validate_client_id(client_id)
    if username is not None and (not isinstance(username, str) or not username):
        raise ValueError("MQTT username must be a non-empty string")
    if username is not None and not isinstance(password, str):
        raise ValueError("MQTT password is required when username is set")

    try:
        import paho.mqtt.client as mqtt
    except ImportError as error:
        raise RuntimeError("install requirements-mqtt.txt to enable MQTT runtime") from error

    parsed = urlparse(normalized_url)
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=client_id,
        protocol=mqtt.MQTTv5,
    )
    if username is not None:
        client.username_pw_set(username, password)
    if parsed.scheme == "mqtts":
        client.tls_set(ca_certs=ca_certs)

    client.connect(parsed.hostname, parsed.port or (8883 if parsed.scheme == "mqtts" else 1883), 30)
    client.loop_start()

    def publisher(publish_topic, payload, qos, retain):
        info = client.publish(publish_topic, payload, qos=qos, retain=retain)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            return False
        info.wait_for_publish(timeout=10)
        return bool(info.is_published())

    def close():
        client.disconnect()
        client.loop_stop()

    return MqttEventTransport(topic, publisher, max_attempts, backoff_seconds), close
