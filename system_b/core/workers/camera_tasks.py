"""Per-camera bounded task queues and stoppable worker lifecycles."""

import logging
import queue as queue_module
import threading
import time
from dataclasses import dataclass, field

try:
    from ..observability import log_event
except ImportError:
    from observability import log_event


@dataclass
class _TaskSlot:
    queue: queue_module.Queue = field(default_factory=lambda: queue_module.Queue(maxsize=1))
    lock: threading.RLock = field(default_factory=threading.RLock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    periodic_stop: threading.Event = field(default_factory=threading.Event)
    worker: threading.Thread = None
    periodic_worker: threading.Thread = None
    active: bool = False
    accepted: int = 0
    completed: int = 0
    failed: int = 0
    last_error_code: str = ""
    last_started_at: str = ""
    last_finished_at: str = ""


class CameraTaskManager:
    """Run one bounded worker queue per camera and task kind."""

    DEFAULT_TASKS = ("detection", "sd_sync", "save_to_sd")

    def __init__(self, camera_ids, metrics=None, logger=None):
        self._logger = logger or logging.getLogger(__name__)
        self._metrics = metrics
        self._lock = threading.RLock()
        self._slots = {
            camera_id: {task: _TaskSlot() for task in self.DEFAULT_TASKS}
            for camera_id in camera_ids
        }

    def _slot(self, camera_id, task):
        with self._lock:
            if camera_id not in self._slots:
                raise KeyError("unknown camera")
            if task not in self._slots[camera_id]:
                raise KeyError("unknown task")
            return self._slots[camera_id][task]

    def _ensure_worker(self, camera_id, task, slot):
        with slot.lock:
            if slot.worker is None or not slot.worker.is_alive():
                slot.stop_event.clear()
                slot.worker = threading.Thread(
                    target=self._worker_loop,
                    args=(camera_id, task, slot),
                    name=f"agrivision-{task}-{camera_id}",
                    daemon=True,
                )
                slot.worker.start()

    def submit(self, camera_id, task, callback):
        if not callable(callback):
            raise TypeError("callback must be callable")
        slot = self._slot(camera_id, task)
        with slot.lock:
            if slot.active or not slot.queue.empty() or slot.stop_event.is_set():
                return False
            try:
                slot.queue.put_nowait(callback)
            except queue_module.Full:
                return False
            slot.accepted += 1
            self._set_queue_metric(task, slot.queue.qsize(), camera_id)
        self._ensure_worker(camera_id, task, slot)
        return True

    def start_periodic(self, camera_id, task, callback, interval_seconds, run_immediately=False):
        if isinstance(interval_seconds, bool) or not isinstance(interval_seconds, (int, float)) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if not callable(callback):
            raise TypeError("callback must be callable")
        slot = self._slot(camera_id, task)
        with slot.lock:
            if slot.periodic_worker is not None and slot.periodic_worker.is_alive():
                return False
            slot.periodic_stop.clear()
            slot.stop_event.clear()
            slot.periodic_worker = threading.Thread(
                target=self._periodic_loop,
                args=(camera_id, task, callback, float(interval_seconds), run_immediately, slot),
                name=f"agrivision-periodic-{task}-{camera_id}",
                daemon=True,
            )
            slot.periodic_worker.start()
        return True

    def _periodic_loop(self, camera_id, task, callback, interval_seconds, run_immediately, slot):
        if run_immediately:
            self.submit(camera_id, task, callback)
        while not slot.periodic_stop.wait(interval_seconds):
            self.submit(camera_id, task, callback)

    def _worker_loop(self, camera_id, task, slot):
        while not slot.stop_event.is_set():
            try:
                callback = slot.queue.get(timeout=0.2)
            except queue_module.Empty:
                continue
            with slot.lock:
                slot.active = True
                slot.last_started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                self._set_queue_metric(task, 0, camera_id)
            try:
                callback()
            except Exception:
                with slot.lock:
                    slot.failed += 1
                    slot.last_error_code = "task_failed"
                log_event(
                    self._logger,
                    logging.ERROR,
                    "camera_task_failed",
                    camera_id=camera_id,
                    task=task,
                )
            else:
                with slot.lock:
                    slot.completed += 1
                    slot.last_error_code = ""
            finally:
                with slot.lock:
                    slot.active = False
                    slot.last_finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                slot.queue.task_done()

    def _set_queue_metric(self, task, depth, camera_id=None):
        if self._metrics is not None:
            self._metrics.set_queue_depth(task, depth, camera_id=camera_id)

    def snapshot(self, camera_id=None):
        with self._lock:
            camera_items = self._slots.items() if camera_id is None else [(camera_id, self._slots[camera_id])]
            result = {}
            for current_camera_id, tasks in camera_items:
                result[current_camera_id] = {}
                for task, slot in tasks.items():
                    with slot.lock:
                        result[current_camera_id][task] = {
                            "queued": slot.queue.qsize(),
                            "running": slot.active,
                            "accepted": slot.accepted,
                            "completed": slot.completed,
                            "failed": slot.failed,
                            "last_error_code": slot.last_error_code,
                            "last_started_at": slot.last_started_at,
                            "last_finished_at": slot.last_finished_at,
                        }
            return {"cameras": result}

    def wait_for_idle(self, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = self.snapshot()["cameras"]
            if all(
                not task["running"] and task["queued"] == 0
                for camera in snapshot.values() for task in camera.values()
            ):
                return True
            time.sleep(0.01)
        return False

    def stop_all(self, timeout=5):
        with self._lock:
            slots = [slot for tasks in self._slots.values() for slot in tasks.values()]
        for slot in slots:
            slot.periodic_stop.set()
            slot.stop_event.set()
        deadline = time.monotonic() + timeout
        for slot in slots:
            for worker in (slot.periodic_worker, slot.worker):
                if worker is not None and worker is not threading.current_thread():
                    remaining = max(0, deadline - time.monotonic())
                    worker.join(remaining)
