import unittest

from system_b.core.event_sync_status import EventSyncStatus


class Stage22RetryClassificationTests(unittest.TestCase):
    def test_status_preserves_only_fixed_failure_category(self):
        status = EventSyncStatus()

        status.record("failure", "http", pending=1, failure_type="permanent")

        snapshot = status.snapshot(True, 30, "http", False, 1)
        self.assertEqual(snapshot["last_failure_type"], "permanent")

    def test_client_error_is_not_retried(self):
        from system_b.core.event_transport import EventTransport

        attempts = []

        def sender(url, events):
            attempts.append(1)
            return 401

        transport = EventTransport(
            "http://127.0.0.1:9100/events",
            sender,
            max_attempts=5,
            backoff_seconds=0,
        )
        result = transport.sync([{"event_id": "permanent-1"}], lambda event_id: None)

        self.assertEqual(attempts, [1])
        self.assertEqual(result, {"sent": 0, "attempts": 1, "failure_type": "permanent"})

    def test_server_error_still_retries_with_bounded_attempts(self):
        from system_b.core.event_transport import EventTransport

        attempts = []

        def sender(url, events):
            attempts.append(1)
            return 503

        transport = EventTransport(
            "http://127.0.0.1:9100/events",
            sender,
            max_attempts=3,
            backoff_seconds=0,
        )
        result = transport.sync([{"event_id": "temporary-1"}], lambda event_id: None)

        self.assertEqual(len(attempts), 3)
        self.assertEqual(result, {"sent": 0, "attempts": 3, "failure_type": "retry_exhausted"})


if __name__ == "__main__":
    unittest.main()
