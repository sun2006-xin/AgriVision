import pathlib
import socketserver
import sys
import threading
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system_b" / "core"))


def read_mqtt_packet(stream):
    header = stream.read(1)
    if not header:
        return None, b""
    multiplier = 1
    remaining = 0
    while True:
        value = stream.read(1)
        if not value:
            return None, b""
        byte = value[0]
        remaining += (byte & 127) * multiplier
        if not byte & 128:
            break
        multiplier *= 128
    return header[0] >> 4, stream.read(remaining)


class LocalMqttHandler(socketserver.StreamRequestHandler):
    published = []

    def handle(self):
        while True:
            packet_type, body = read_mqtt_packet(self.rfile)
            if packet_type is None:
                return
            if packet_type == 1:  # CONNECT -> CONNACK (MQTT v5)
                self.wfile.write(b"\x20\x03\x00\x00\x00")
                self.wfile.flush()
            elif packet_type == 3:  # PUBLISH
                topic_length = int.from_bytes(body[:2], "big")
                offset = 2 + topic_length
                packet_id = int.from_bytes(body[offset:offset + 2], "big")
                properties_length = body[offset + 2]
                payload = body[offset + 3 + properties_length:]
                topic = body[2:2 + topic_length].decode("utf-8")
                self.__class__.published.append((topic, packet_id, payload))
                self.wfile.write(bytes((0x40, 0x04, body[offset], body[offset + 1], 0, 0)))
                self.wfile.flush()
            elif packet_type == 12:  # PINGREQ -> PINGRESP
                self.wfile.write(b"\xd0\x00")
                self.wfile.flush()
            elif packet_type == 14:
                return


class ReconnectingMqttHandler(LocalMqttHandler):
    connections = 0

    def handle(self):
        connection_number = None
        while True:
            packet_type, body = read_mqtt_packet(self.rfile)
            if packet_type is None:
                return
            if packet_type == 1:
                self.__class__.connections += 1
                connection_number = self.__class__.connections
                self.wfile.write(b"\x20\x03\x00\x00\x00")
                self.wfile.flush()
            elif packet_type == 3:
                if connection_number == 1:
                    return
                topic_length = int.from_bytes(body[:2], "big")
                offset = 2 + topic_length
                packet_id = int.from_bytes(body[offset:offset + 2], "big")
                self.__class__.published.append((body[2:2 + topic_length].decode("utf-8"), packet_id, body))
                self.wfile.write(bytes((0x40, 0x04, body[offset], body[offset + 1], 0, 0)))
                self.wfile.flush()
            elif packet_type == 12:
                self.wfile.write(b"\xd0\x00")
                self.wfile.flush()
            elif packet_type == 14:
                return


class Stage10LocalMqttBrokerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import paho.mqtt.client  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("install requirements-mqtt.txt for local MQTT integration")
        LocalMqttHandler.published = []
        cls.server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), LocalMqttHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_paho_runtime_publishes_qos1_to_local_broker(self):
        from event_mqtt import MqttEventTransport
        from mqtt_runtime import create_paho_transport

        port = self.server.server_address[1]
        transport, close = create_paho_transport(
            f"mqtt://127.0.0.1:{port}",
            "agrivision/events",
            "agrivision-local-test",
            max_attempts=1,
            backoff_seconds=0,
        )
        acknowledged = []
        try:
            result = transport.sync(
                [{"event_id": "local-mqtt-1", "schema_version": 1}],
                acknowledged.append,
            )
        finally:
            close()

        self.assertEqual(result, {"sent": 1, "attempts": 1})
        self.assertEqual(acknowledged, ["local-mqtt-1"])
        self.assertEqual(len(LocalMqttHandler.published), 1)
        self.assertEqual(LocalMqttHandler.published[0][0], "agrivision/events")
        self.assertEqual(LocalMqttHandler.published[0][1] > 0, True)


class Stage18MqttReconnectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import paho.mqtt.client  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("install requirements-mqtt.txt for local MQTT integration")
        ReconnectingMqttHandler.connections = 0
        ReconnectingMqttHandler.published = []
        cls.server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), ReconnectingMqttHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_running_client_reconnects_after_broker_drops_first_connection(self):
        from mqtt_runtime import create_paho_transport

        port = self.server.server_address[1]
        transport, close = create_paho_transport(
            f"mqtt://127.0.0.1:{port}",
            "agrivision/events",
            "agrivision-reconnect-test",
            max_attempts=5,
            backoff_seconds=0.2,
            connect_attempts=1,
            connect_backoff_seconds=0,
        )
        acknowledged = []
        try:
            result = transport.sync(
                [{"event_id": "reconnect-1", "schema_version": 1}],
                acknowledged.append,
            )
        finally:
            close()

        self.assertEqual(result["sent"], 1)
        self.assertEqual(acknowledged, ["reconnect-1"])
        self.assertGreaterEqual(ReconnectingMqttHandler.connections, 2)
        self.assertEqual(len(ReconnectingMqttHandler.published), 1)


if __name__ == "__main__":
    unittest.main()
