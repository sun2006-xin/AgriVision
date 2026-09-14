"""Explainable, bounded temporal fusion for per-camera detection levels."""

from collections import deque
import math


LEVEL_NAMES = {0: "正常", 1: "注意", 2: "警告", 3: "严重"}
_NUMERIC_FIELDS = (
    "disease_count", "white_count", "disease_ratio", "white_ratio", "green_ratio",
)


def _number(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


class TemporalFusion:
    """Fuse recent frame levels with decay-weighted voting and hysteresis.

    The current stable level changes only after the candidate wins
    ``min_consecutive`` updates.  A single high-level outlier therefore does
    not override a consistent recent history, while the returned support and
    pending fields make every transition inspectable.
    """

    def __init__(self, window_size=5, min_consecutive=3, decay=0.8):
        if isinstance(window_size, bool) or not isinstance(window_size, int) or not 1 <= window_size <= 100:
            raise ValueError("window_size must be an integer from 1 to 100")
        if isinstance(min_consecutive, bool) or not isinstance(min_consecutive, int) or not 1 <= min_consecutive <= window_size:
            raise ValueError("min_consecutive must be an integer within window_size")
        decay = _number(decay, "decay")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be greater than 0 and no greater than 1")
        self.window_size = window_size
        self.min_consecutive = min_consecutive
        self.decay = decay
        self._frames = deque(maxlen=window_size)
        self._stable_level_code = 0
        self._pending_level_code = None
        self._pending_count = 0

    @staticmethod
    def _normalize_frame(frame):
        if not isinstance(frame, dict):
            raise ValueError("frame must be a JSON object")
        raw_code = frame.get("level_code", 0)
        if isinstance(raw_code, bool) or not isinstance(raw_code, int) or raw_code not in LEVEL_NAMES:
            raise ValueError("level_code must be an integer from 0 to 3")
        normalized = {
            "level_code": raw_code,
            "level": LEVEL_NAMES[raw_code],
        }
        for field in _NUMERIC_FIELDS:
            value = _number(frame.get(field, 0.0), field)
            if field.endswith("ratio") and not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be between 0 and 1")
            if field.endswith("count") and value < 0.0:
                raise ValueError(f"{field} must be non-negative")
            normalized[field] = value
        return normalized

    def _aggregate(self):
        frames = list(self._frames)
        weights = [self.decay ** (len(frames) - 1 - index) for index in range(len(frames))]
        total_weight = sum(weights)
        votes = {level: 0.0 for level in LEVEL_NAMES}
        averages = {field: 0.0 for field in _NUMERIC_FIELDS}
        for frame, weight in zip(frames, weights):
            votes[frame["level_code"]] += weight
            for field in _NUMERIC_FIELDS:
                averages[field] += frame[field] * weight
        for field in averages:
            averages[field] = averages[field] / total_weight if total_weight else 0.0
        candidate = max(votes, key=lambda level: (votes[level], -level))
        support = votes[candidate] / total_weight if total_weight else 0.0
        return frames, votes, averages, candidate, support

    def update(self, frame):
        normalized = self._normalize_frame(frame)
        previous_stable = self._stable_level_code
        self._frames.append(normalized)
        frames, votes, averages, candidate, support = self._aggregate()

        if candidate == self._stable_level_code:
            self._pending_level_code = None
            self._pending_count = 0
        elif candidate == self._pending_level_code:
            self._pending_count += 1
        else:
            self._pending_level_code = candidate
            self._pending_count = 1

        if self._pending_level_code is not None and self._pending_count >= self.min_consecutive:
            self._stable_level_code = self._pending_level_code
            self._pending_level_code = None
            self._pending_count = 0

        return {
            "method": "decayed_weighted_vote",
            "window_size": self.window_size,
            "decay": self.decay,
            "frame_count": len(frames),
            "candidate_level_code": candidate,
            "candidate_level": LEVEL_NAMES[candidate],
            "stable_level_code": self._stable_level_code,
            "stable_level": LEVEL_NAMES[self._stable_level_code],
            "level_changed": self._stable_level_code != previous_stable,
            "pending_level_code": self._pending_level_code,
            "pending_level": (
                LEVEL_NAMES[self._pending_level_code]
                if self._pending_level_code is not None else None
            ),
            "pending_count": self._pending_count,
            "weighted_votes": {str(level): round(value, 6) for level, value in votes.items()},
            "weighted_support": round(support, 6),
            "averages": {field: round(value, 6) for field, value in averages.items()},
            "recent_levels": [item["level_code"] for item in frames],
            "recent_frames": [dict(item) for item in frames],
            "evidence": {
                "rule": "最近帧按时间衰减加权投票",
                "candidate_support": round(support, 6),
                "transition": "stable" if self._stable_level_code != previous_stable else "pending_or_unchanged",
            },
        }

    def reset(self):
        self._frames.clear()
        self._stable_level_code = 0
        self._pending_level_code = None
        self._pending_count = 0
