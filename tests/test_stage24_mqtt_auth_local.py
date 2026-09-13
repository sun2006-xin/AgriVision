from pathlib import Path
import socketserver
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system_b" / "core"))

from mqtt_runtime import create_paho_transport
from tests.test_stage10_local_mqtt_broker import read_mqtt_packet
from tests.test_stage23_mqtt_tls_local import TlsMqttServer, write_test_certificate


class AuthTlsMqttHandler(socketserver.StreamRequestHandler):
    username = b"agrivision-test-user"
    password = b"agrivision-test-pass"
    published = []

    def handle(self):
        while True:
            packet_type, body = read_mqtt_packet(self.rfile)
            if packet_type is None:
                return
            if packet_type == 1:
                if self.username not in body or self.password not in body:
                    self.wfile.write(b"\x20\x03\x00\x86\x00")
                    self.wfile.flush()
                    return
                self.wfile.write(b"\x20\x03\x00\x00\x00")
                self.wfile.flush()
            elif packet_type == 3:
                topic_length = int.from_bytes(body[:2], "big")
                offset = 2 + topic_length
                packet_id = int.from_bytes(body[offset:offset + 2], "big")
                payload = body[offset + 3 + body[offset + 2]:]
                topic = body[2:2 + topic_length].decode("utf-8")
                self.__class__.published.append((topic, packet_id, payload))
                self.wfile.write(bytes((0x40, 0x04, body[offset], body[offset + 1], 0, 0)))
                self.wfile.flush()
            elif packet_type == 12:
                self.wfile.write(b"\xd0\x00")
                self.wfile.flush()
            elif packet_type == 14:
                return


class Stage24MqttAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        certificate, private_key = write_test_certificate(cls.temp_dir.name)
        AuthTlsMqttHandler.published = []
        cls.server = TlsMqttServer(("127.0.0.1", 0), AuthTlsMqttHandler, certificate, private_key)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.certificate = certificate

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.temp_dir.cleanup()

    def test_paho_sends_separate_credentials_over_tls(self):
        port = self.server.server_address[1]
        transport, close = create_paho_transport(
            f"mqtts://127.0.0.1:{port}",
            "agrivision/events",
            "agrivision-auth-test",
            username=AuthTlsMqttHandler.username.decode("ascii"),
            password=AuthTlsMqttHandler.password.decode("ascii"),
            ca_certs=self.certificate,
            max_attempts=1,
            backoff_seconds=0,
        )
        acknowledged = []
        try:
            result = transport.sync(
                [{"event_id": "auth-local-1", "schema_version": 1}],
                acknowledged.append,
            )
        finally:
            close()

        self.assertEqual(result, {"sent": 1, "attempts": 1})
        self.assertEqual(acknowledged, ["auth-local-1"])
        self.assertEqual(len(AuthTlsMqttHandler.published), 1)
        self.assertEqual(AuthTlsMqttHandler.published[0][0], "agrivision/events")

    def test_wrong_credentials_fail_without_publishing(self):
        from mqtt_runtime import create_paho_transport

        port = self.server.server_address[1]
        published_before = len(AuthTlsMqttHandler.published)
        with self.assertRaises(RuntimeError) as context:
            create_paho_transport(
                f"mqtts://127.0.0.1:{port}",
                "agrivision/events",
                "agrivision-auth-reject-test",
                username=AuthTlsMqttHandler.username.decode("ascii"),
                password="bad",
                ca_certs=self.certificate,
                max_attempts=1,
                backoff_seconds=0,
            )

        self.assertEqual(str(context.exception), "MQTT broker connection failed")
        self.assertEqual(len(AuthTlsMqttHandler.published), published_before)


if __name__ == "__main__":
    unittest.main()
