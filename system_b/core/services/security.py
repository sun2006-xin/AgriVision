"""Authentication and bounded rate limiting for System B HTTP APIs."""

import hmac
import os
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class SecurityDecision:
    status: str
    reason: str = ""
    retry_after: int = 0


class ApiSecurity:
    """Use bearer authentication when configured and fail closed for remote access."""

    MAX_TRACKED_CLIENTS = 4096

    def __init__(self, api_token="", auth_required=False, rate_limit_max=120, rate_limit_window_seconds=60):
        if isinstance(api_token, str) and api_token and not 16 <= len(api_token) <= 256:
            raise ValueError("api_token must contain 16 to 256 characters")
        if isinstance(rate_limit_max, bool) or not isinstance(rate_limit_max, int) or rate_limit_max < 1:
            raise ValueError("rate_limit_max must be a positive integer")
        if not isinstance(rate_limit_window_seconds, (int, float)) or rate_limit_window_seconds <= 0:
            raise ValueError("rate_limit_window_seconds must be positive")
        self.api_token = api_token or ""
        self.auth_required = bool(auth_required)
        self.rate_limit_max = rate_limit_max
        self.rate_limit_window_seconds = float(rate_limit_window_seconds)
        self._lock = threading.Lock()
        self._windows = {}

    @classmethod
    def from_env(cls):
        token = os.environ.get("AGRIVISION_API_TOKEN", "").strip()
        required = os.environ.get("AGRIVISION_API_AUTH_REQUIRED", "false").lower() in {"1", "true", "yes"}
        try:
            max_requests = int(os.environ.get("AGRIVISION_API_RATE_LIMIT_MAX", "120"))
        except ValueError:
            max_requests = 120
        try:
            window = float(os.environ.get("AGRIVISION_API_RATE_LIMIT_WINDOW", "60"))
        except ValueError:
            window = 60.0
        return cls(token, required, max_requests, window)

    @staticmethod
    def _is_loopback(remote_addr):
        return remote_addr in {None, "", "127.0.0.1", "::1", "localhost"}

    @staticmethod
    def _extract_token(headers):
        authorization = headers.get("Authorization", "") if headers else ""
        prefix = "Bearer "
        if authorization.startswith(prefix):
            return authorization[len(prefix):].strip()
        return (headers.get("X-API-Key", "") if headers else "").strip()

    def _allow_rate(self, remote_addr):
        now = time.monotonic()
        key = remote_addr or "unknown"
        with self._lock:
            if key not in self._windows and len(self._windows) >= self.MAX_TRACKED_CLIENTS:
                expired = [
                    client for client, (started, _) in self._windows.items()
                    if now - started >= self.rate_limit_window_seconds
                ]
                for client in expired:
                    self._windows.pop(client, None)
                if len(self._windows) >= self.MAX_TRACKED_CLIENTS:
                    return False, 1
            started, count = self._windows.get(key, (now, 0))
            if now - started >= self.rate_limit_window_seconds:
                started, count = now, 0
            if count >= self.rate_limit_max:
                retry_after = max(1, int(self.rate_limit_window_seconds - (now - started)))
                self._windows[key] = (started, count)
                return False, retry_after
            self._windows[key] = (started, count + 1)
            return True, 0

    def authorize(self, path, remote_addr, headers):
        if not (path.startswith("/api/") or path == "/metrics"):
            return SecurityDecision("allow")

        if self.api_token:
            allowed, retry_after = self._allow_rate(remote_addr)
            if not allowed:
                return SecurityDecision("rate_limited", "rate_limit_exceeded", retry_after)
            provided = self._extract_token(headers)
            if not provided or not hmac.compare_digest(provided, self.api_token):
                return SecurityDecision("deny", "invalid_api_token")
        elif self.auth_required or not self._is_loopback(remote_addr):
            return SecurityDecision("deny", "api_auth_not_configured")

        if not self.api_token:
            allowed, retry_after = self._allow_rate(remote_addr)
            if not allowed:
                return SecurityDecision("rate_limited", "rate_limit_exceeded", retry_after)
        return SecurityDecision("allow")

    def safe_status(self):
        return {
            "auth_configured": bool(self.api_token),
            "auth_required_for_remote": True,
            "rate_limit_max": self.rate_limit_max,
            "rate_limit_window_seconds": self.rate_limit_window_seconds,
        }
