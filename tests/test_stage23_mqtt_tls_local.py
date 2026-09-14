import datetime
import ipaddress
from pathlib import Path
import socketserver
import ssl
import sys
import tempfile
import threading
import unittest

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system_b" / "core"))

from mqtt_runtime import create_paho_transport
from tests.test_stage10_local_mqtt_broker import read_mqtt_packet


class TlsMqttHandler(socketserver.StreamRequestHandler):
    published = []

    def handle(self):
        while True:
            packet_type, body = read_mqtt_packet(self.rfile)
            if packet_type is None:
                return
            if packet_type == 1:
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


class TlsMqttServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, server_address, handler_class, certificate, private_key):
        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.context.load_cert_chain(certificate, private_key)
        super().__init__(server_address, handler_class)

    def get_request(self):
        raw_socket, address = super().get_request()
        try:
            return self.context.wrap_socket(raw_socket, server_side=True), address
        except Exception:
            raw_socket.close()
            raise


def write_test_certificate(directory):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "AgriVision local MQTT")])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(minutes=10))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    certificate_path = Path(directory) / "ca.crt"
    private_key_path = Path(directory) / "server.key"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    private_key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    return str(certificate_path), str(private_key_path)


class Stage23MqttTlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        certificate, private_key = write_test_certificate(cls.temp_dir.name)
        TlsMqttHandler.published = []
        cls.server = TlsMqttServer(("127.0.0.1", 0), TlsMqttHandler, certificate, private_key)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.certificate = certificate

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.temp_dir.cleanup()

    def test_paho_validates_local_tls_certificate_and_publishes(self):
        port = self.server.server_address[1]
        transport, close = create_paho_transport(
            f"mqtts://127.0.0.1:{port}",
            "agrivision/events",
            "agrivision-tls-test",
            ca_certs=self.certificate,
            max_attempts=1,
            backoff_seconds=0,
        )
        acknowledged = []
        try:
            result = transport.sync(
                [{"event_id": "tls-local-1", "schema_version": 1}],
                acknowledged.append,
            )
        finally:
            close()

        self.assertEqual(result, {"sent": 1, "attempts": 1})
        self.assertEqual(acknowledged, ["tls-local-1"])
        self.assertEqual(len(TlsMqttHandler.published), 1)
        self.assertEqual(TlsMqttHandler.published[0][0], "agrivision/events")


if __name__ == "__main__":
    unittest.main()
