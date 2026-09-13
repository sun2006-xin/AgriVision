import tempfile
import unittest
import multiprocessing
from pathlib import Path


from system_b.core.process_sync_lock import ProcessSyncLock


def _probe_lock_in_child(path, results, continue_event):
    lock = ProcessSyncLock(path)
    results.put(lock.acquire(blocking=False))
    continue_event.wait(2)
    acquired_after_release = lock.acquire(blocking=False)
    results.put(acquired_after_release)
    if acquired_after_release:
        lock.release()


class Stage21ProcessSyncLockTests(unittest.TestCase):
    def test_system_b_uses_shared_runtime_lock_file(self):
        source = (Path(__file__).resolve().parents[1] / "system_b" / "core" / "app.py").read_text(encoding="utf-8")

        self.assertIn("from process_sync_lock import ProcessSyncLock", source)
        self.assertIn("ProcessSyncLock(os.path.join(OFFLINE_EVENTS_DIR, \".sync-lock.db\"))", source)

    def test_lock_is_exclusive_across_processes(self):
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "sync-lock.db")
            holder = ProcessSyncLock(path)
            self.assertTrue(holder.acquire(blocking=False))
            results = context.Queue()
            continue_event = context.Event()
            child = context.Process(target=_probe_lock_in_child, args=(path, results, continue_event))
            try:
                child.start()
                self.assertFalse(results.get(timeout=5))
                holder.release()
                continue_event.set()
                self.assertTrue(results.get(timeout=5))
                child.join(timeout=5)
                self.assertEqual(child.exitcode, 0)
            finally:
                continue_event.set()
                if child.pid is not None:
                    if child.is_alive():
                        child.terminate()
                    child.join(timeout=5)
                holder.release()

    def test_second_lock_holder_is_rejected_until_first_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sync-lock.db"
            first = ProcessSyncLock(path)
            second = ProcessSyncLock(path)

            self.assertTrue(first.acquire(blocking=False))
            self.assertFalse(second.acquire(blocking=False))
            first.release()
            self.assertTrue(second.acquire(blocking=False))
            second.release()

    def test_lock_context_releases_after_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sync-lock.db"
            first = ProcessSyncLock(path)
            second = ProcessSyncLock(path)

            with self.assertRaises(RuntimeError):
                with first:
                    self.assertFalse(second.acquire(blocking=False))
                    raise RuntimeError("test")

            self.assertTrue(second.acquire(blocking=False))
            second.release()


if __name__ == "__main__":
    unittest.main()
