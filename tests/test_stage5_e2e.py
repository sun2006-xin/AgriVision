import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system_b" / "core"))


class InMemoryEventSink:
    """Local-only receiver used to verify the delivery contract without a network."""

    def __init__(self):
        self.seen_keys = set()
        self.accepted_events = []
        self.fail_after_first_accept = True

    def send(self, url, events):
        from event_transport import build_batch_idempotency_key

        key = build_batch_idempotency_key(events)
        if key not in self.seen_keys:
            self.seen_keys.add(key)
            self.accepted_events.extend(events)
            if self.fail_after_first_accept:
                self.fail_after_first_accept = False
                raise OSError("simulated response loss after ingest")
        return 208 if key in self.seen_keys else 202


class Stage5E2ETests(unittest.TestCase):
    def test_response_loss_retries_without_duplicate_and_acks_after_success(self):
        from event_transport import EventTransport
        from offline_cache import OfflineEventCache

        event = {
            "event_id": "event-e2e-1",
            "schema_version": 1,
            "camera_id": "cam-test",
            "payload": {"level_code": 1, "disease_count": 2},
        }
        sink = InMemoryEventSink()
        transport = EventTransport(
            "http://127.0.0.1:9100/events",
            sink.send,
            max_attempts=3,
            backoff_seconds=0,
        )

        with tempfile.TemporaryDirectory() as directory:
            cache = OfflineEventCache(directory)
            cache.put(event)
            result = transport.sync(cache.list_pending(), cache.ack)

            self.assertEqual(result, {"sent": 1, "attempts": 2})
            self.assertEqual(sink.accepted_events, [event])
            self.assertEqual(cache.list_pending(), [])

            duplicate_result = transport.sync([event], lambda event_id: None)
            self.assertEqual(duplicate_result, {"sent": 1, "attempts": 1})
            self.assertEqual(sink.accepted_events, [event])


if __name__ == "__main__":
    unittest.main()
