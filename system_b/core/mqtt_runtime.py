"""Optional Paho MQTT runtime; no broker connection occurs at import time."""

import re
import threading
from time import sleep
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
    connect_attempts=3,
    connect_backoff_seconds=0.5,
    connack_timeout=5.0,
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
    if not 1 <= connect_attempts <= 5:
        raise ValueError("connect_attempts must be between 1 and 5")
    if connect_backoff_seconds < 0:
        raise ValueError("connect_backoff_seconds must not be negative")
    if isinstance(connack_timeout, bool) or not isinstance(connack_timeout, (int, float)):
        raise ValueError("connack_timeout must be a number")
    if not 0.1 <= connack_timeout <= 30:
        raise ValueError("connack_timeout must be between 0.1 and 30 seconds")

    try:
        import paho.mqtt.client as mqtt
    except ImportError as error:
        raise RuntimeError("install requirements-mqtt.txt to enable MQTT runtime") from error

    parsed = urlparse(normalized_url)
    def build_client():
        new_client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            protocol=mqtt.MQTTv5,
        )
        if username is not None:
            new_client.username_pw_set(username, password)
        if parsed.scheme == "mqtts":
            new_client.tls_set(ca_certs=ca_certs)
        return new_client

    client = build_client()

    port = parsed.port or (8883 if parsed.scheme == "mqtts" else 1883)
    loop_started = False
    connection_event = threading.Event()
    connection_failure = []
    reconnect_on_failure = None

    def on_connect(_client, _userdata, _flags, reason_code, _properties):
        failure = getattr(reason_code, "is_failure", None)
        if failure is None:
            try:
                failure = int(reason_code) != int(mqtt.MQTT_ERR_SUCCESS)
            except (TypeError, ValueError):
                failure = str(reason_code).lower() not in {"0", "success"}
        if failure:
            connection_failure.append(True)
        connection_event.set()

    # Real Paho clients expose ``is_connected``.  Validate CONNACK for every
    # such client, including anonymous local-development connections; a
    # successful TCP connect alone does not prove MQTT session acceptance.
    validate_connack = hasattr(client, "is_connected")
    if validate_connack:
        client.on_connect = on_connect
    for attempt in range(1, connect_attempts + 1):
        try:
            if validate_connack:
                connection_event.clear()
                connection_failure.clear()
            connection_result = client.connect(parsed.hostname, port, 30)
            if connection_result is not None and int(connection_result) != int(mqtt.MQTT_ERR_SUCCESS):
                raise RuntimeError("MQTT broker rejected connection")
            if validate_connack:
                reconnect_on_failure = getattr(client, "reconnect_on_failure", True)
                client.reconnect_on_failure = False
                client.loop_start()
                loop_started = True
                if not connection_event.wait(connack_timeout) or connection_failure:
                    raise RuntimeError("MQTT broker rejected connection")
                client.reconnect_on_failure = reconnect_on_failure
            break
        except Exception as error:
            if validate_connack and reconnect_on_failure is not None:
                client.reconnect_on_failure = reconnect_on_failure
            if attempt == connect_attempts:
                try:
                    client.disconnect()
                except Exception:
                    pass
                try:
                    client.loop_stop()
                except Exception:
                    pass
                try:
                    client.reinitialise()
                except Exception:
                    pass
                raise RuntimeError("MQTT broker connection failed") from error
            try:
                client.disconnect()
            except Exception:
                pass
            try:
                client.loop_stop()
            except Exception:
                pass
            try:
                # Paho keeps its internal wake-up socketpair after loop_stop.
                client.reinitialise()
            except Exception:
                pass
            loop_started = False
            # Use a fresh configured client so a failed connection cannot leak
            # transport state, callbacks, or socket resources into a retry.
            client = build_client()
            if validate_connack:
                client.on_connect = on_connect
            if connect_backoff_seconds:
                sleep(connect_backoff_seconds * attempt)
    if not loop_started:
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
