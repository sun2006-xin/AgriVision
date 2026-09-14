"""Small dependency-free Prometheus-style metrics registry for System B."""

import math
import threading


class MetricsRegistry:
    """Keep bounded counters, gauges, and histograms with fixed label sets."""

    HTTP_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    INFERENCE_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(self):
        self._lock = threading.RLock()
        self._counters = {}
        self._gauges = {}
        self._histograms = {}

    @staticmethod
    def _key(labels):
        return tuple(sorted((str(key), str(value)) for key, value in (labels or {}).items()))

    @staticmethod
    def _escape(value):
        return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    @classmethod
    def _render_labels(cls, labels):
        if not labels:
            return ""
        return "{" + ",".join(
            f'{key}="{cls._escape(value)}"' for key, value in labels
        ) + "}"

    def inc(self, name, labels=None, value=1):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError("metric increment must be a finite non-negative number")
        key = self._key(labels)
        with self._lock:
            self._counters[(name, key)] = self._counters.get((name, key), 0) + value

    def set_gauge(self, name, value, labels=None):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("metric gauge must be a finite number")
        with self._lock:
            self._gauges[(name, self._key(labels))] = value

    def observe(self, name, value, labels=None, buckets=None):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError("metric observation must be a finite non-negative number")
        bucket_values = tuple(buckets or self.HTTP_BUCKETS)
        if not bucket_values or any(
            not isinstance(bound, (int, float)) or not math.isfinite(float(bound)) or bound <= 0
            for bound in bucket_values
        ):
            raise ValueError("metric buckets must be finite positive numbers")
        if tuple(sorted(bucket_values)) != bucket_values:
            raise ValueError("metric buckets must be sorted")
        key = (name, self._key(labels))
        with self._lock:
            item = self._histograms.setdefault(
                key, {"buckets": [0] * len(bucket_values), "count": 0, "sum": 0.0, "bounds": bucket_values}
            )
            for index, bound in enumerate(bucket_values):
                if value <= bound:
                    item["buckets"][index] += 1
            item["count"] += 1
            item["sum"] += float(value)

    def observe_http(self, method, route, status_code, duration_seconds):
        status_class = f"{int(status_code) // 100}xx"
        labels = {"method": method, "route": route, "status_class": status_class}
        self.inc("agrivision_http_requests_total", labels)
        self.observe("agrivision_http_request_duration_seconds", duration_seconds, labels, self.HTTP_BUCKETS)

    def observe_inference(self, engine, duration_ms):
        self.observe(
            "agrivision_inference_duration_seconds",
            float(duration_ms) / 1000.0,
            {"engine": engine},
            self.INFERENCE_BUCKETS,
        )

    def set_queue_depth(self, task, depth, camera_id=None):
        labels = {"task": task}
        if camera_id is not None:
            labels["camera_id"] = camera_id
        self.set_gauge("agrivision_camera_queue_depth", max(0, int(depth)), labels)

    def inc_alert(self, outcome):
        if outcome not in {"success", "failure", "skipped"}:
            raise ValueError("unsupported alert outcome")
        self.inc("agrivision_alert_notifications_total", {"outcome": outcome})

    def prometheus_text(self):
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            histograms = {
                key: {
                    "buckets": list(value["buckets"]),
                    "count": value["count"],
                    "sum": value["sum"],
                    "bounds": value["bounds"],
                }
                for key, value in self._histograms.items()
            }

        lines = []
        metric_names = {name for name, _ in counters} | {name for name, _ in gauges} | {name for name, _ in histograms}
        for name in sorted(metric_names):
            lines.append(f"# TYPE {name} {'histogram' if any(key[0] == name for key in histograms) else 'gauge' if any(key[0] == name for key in gauges) else 'counter'}")
            for (metric_name, labels), value in sorted(counters.items()):
                if metric_name == name:
                    lines.append(f"{name}{self._render_labels(labels)} {value}")
            for (metric_name, labels), value in sorted(gauges.items()):
                if metric_name == name:
                    lines.append(f"{name}{self._render_labels(labels)} {value}")
            for (metric_name, labels), value in sorted(histograms.items()):
                if metric_name != name:
                    continue
                for bound, count in zip(value["bounds"], value["buckets"]):
                    bucket_labels = tuple(sorted((*labels, ("le", str(bound)))))
                    lines.append(f"{name}_bucket{self._render_labels(bucket_labels)} {count}")
                bucket_labels = tuple(sorted((*labels, ("le", "+Inf"))))
                lines.append(f"{name}_bucket{self._render_labels(bucket_labels)} {value['count']}")
                lines.append(f"{name}_count{self._render_labels(labels)} {value['count']}")
                lines.append(f"{name}_sum{self._render_labels(labels)} {value['sum']}")
        return "\n".join(lines) + ("\n" if lines else "")
