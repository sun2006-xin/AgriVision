import socketserver
import sys
import threading
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system_b" / "core"))

from mqtt_runtime import create_paho_transport
from tests.test_stage10_local_mqtt_broker import read_mqtt_packet


class RejectingMqttHandler(socketserver.StreamRequestHandler):
    def handle(self):
        packet_type, _body = read_mqtt_packet(self.rfile)
        if packet_type == 1:
            # MQTT v5: Not authorized.
            self.wfile.write(b"\x20\x03\x00\x87\x00")
            self.wfile.flush()


class Stage25MqttConnackValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import paho.mqtt.client  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("install requirements-mqtt.txt for local MQTT integration")
        cls.server = socketserver.ThreadingTCPServer(
            ("127.0.0.1", 0), RejectingMqttHandler
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_no_auth_connection_rejects_failure_connack_before_transport_is_returned(self):
        port = self.server.server_address[1]
        with self.assertRaisesRegex(RuntimeError, "^MQTT broker connection failed$"):
            create_paho_transport(
                f"mqtt://127.0.0.1:{port}",
                "agrivision/events",
                "agrivision-connack-reject-test",
                max_attempts=1,
                backoff_seconds=0,
            )


if __name__ == "__main__":
    unittest.main()
