import json
import pathlib
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system_b" / "core"))


class LoopbackHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append((self.path, dict(self.headers), body))
        self.send_response(202)
        self.end_headers()

    def log_message(self, format, *args):
        return


class Stage6HttpLoopbackTests(unittest.TestCase):
    def setUp(self):
        LoopbackHandler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), LoopbackHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_real_loopback_http_delivery_carries_contract_and_acks_queue(self):
        from event_transport import EventTransport, send_events_http
        from offline_cache import OfflineEventCache

        event = {
            "event_id": "event-http-1",
            "schema_version": 1,
            "camera_id": "cam-test",
            "payload": {"level_code": 2, "disease_count": 1},
        }
        port = self.server.server_address[1]
        transport = EventTransport(
            f"http://127.0.0.1:{port}/events",
            send_events_http,
            backoff_seconds=0,
        )

        with tempfile.TemporaryDirectory() as directory:
            cache = OfflineEventCache(directory)
            cache.put(event)
            result = transport.sync(cache.list_pending(), cache.ack)

            self.assertEqual(result, {"sent": 1, "attempts": 1})
            self.assertEqual(cache.list_pending(), [])

        self.assertEqual(len(LoopbackHandler.requests), 1)
        path, headers, body = LoopbackHandler.requests[0]
        self.assertEqual(path, "/events")
        self.assertEqual(body, {"schema_version": 1, "events": [event]})
        self.assertEqual(len(headers["Idempotency-Key"]), 64)
        self.assertNotIn("password", json.dumps(body).lower())


if __name__ == "__main__":
    unittest.main()
