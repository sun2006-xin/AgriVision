"""Secure, bounded Webhook alert notification service."""

import hashlib
import hmac
import json
import os
import threading
import time
from datetime import datetime
from urllib.parse import urlparse

import requests


DEFAULT_WEBHOOK_HOSTS = {"oapi.dingtalk.com", "localhost", "127.0.0.1", "::1"}


def _allowed_webhook_hosts():
    configured = os.environ.get("AGRIVISION_WEBHOOK_ALLOWED_HOSTS", "")
    hosts = {item.strip().lower() for item in configured.split(",") if item.strip()}
    return hosts or set(DEFAULT_WEBHOOK_HOSTS)


def validate_webhook_url(url, allowed_hosts=None):
    """Allow only configured HTTPS endpoints, or HTTP loopback for development."""
    if not isinstance(url, str) or not url.strip() or len(url.strip()) > 2048:
        raise ValueError("webhook URL is required and must be no longer than 2048 characters")
    normalized = url.strip()
    parsed = urlparse(normalized)
    hostname = (parsed.hostname or "").lower()
    allowed_hosts = {str(host).lower() for host in (allowed_hosts or _allowed_webhook_hosts())}
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ValueError("webhook URL must use HTTP(S) and include a host")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("webhook URL must not contain credentials or fragments")
    if hostname not in allowed_hosts:
        raise ValueError("webhook host is not allowlisted")
    if parsed.scheme == "http" and hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("remote webhook must use HTTPS")
    try:
        parsed.port
    except ValueError as error:
        raise ValueError("webhook URL has an invalid port") from error
    return normalized


class AlertNotifier:
    """Send alert summaries without exposing Webhook credentials in responses or logs."""

    def __init__(self, webhook_url=None, webhook_secret=None, allowed_hosts=None):
        self._lock = threading.RLock()
        self.allowed_hosts = set(allowed_hosts or _allowed_webhook_hosts())
        configured_url = webhook_url if webhook_url is not None else os.environ.get("AGRIVISION_ALERT_WEBHOOK_URL", "")
        if configured_url:
            self.webhook_url = validate_webhook_url(configured_url, self.allowed_hosts)
        else:
            self.webhook_url = ""
        self.webhook_secret = (webhook_secret if webhook_secret is not None else os.environ.get("AGRIVISION_ALERT_WEBHOOK_SECRET", "")).strip()
        if len(self.webhook_secret) > 256:
            raise ValueError("webhook secret is too long")
        self.enabled = os.environ.get("AGRIVISION_ALERT_ENABLED", "false").lower() in {"1", "true", "yes"}
        self.cooldown_seconds = 300
        self._last_alert_time = {}
        self._pending_alert_levels = set()
        self.alert_history = []
        self.MAX_HISTORY = 50

    def configure(self, webhook_url=None, enabled=None, cooldown=None):
        """Update non-secret runtime settings; the signing secret remains environment-only."""
        with self._lock:
            if webhook_url is not None:
                self.webhook_url = validate_webhook_url(webhook_url, self.allowed_hosts) if webhook_url else ""
            if enabled is not None:
                if not isinstance(enabled, bool):
                    raise ValueError("enabled must be boolean")
                self.enabled = enabled
            if cooldown is not None:
                if isinstance(cooldown, bool) or not isinstance(cooldown, int) or not 0 <= cooldown <= 86400:
                    raise ValueError("cooldown must be an integer from 0 to 86400")
                self.cooldown_seconds = cooldown
            return self.get_config()

    def get_config(self):
        with self._lock:
            parsed = urlparse(self.webhook_url) if self.webhook_url else None
            return {
                # Keep the old key as an empty compatibility field; never return the raw URL.
                "webhook_url": "",
                "webhook_url_masked": f"{parsed.scheme}://{parsed.hostname}" if parsed else "",
                "webhook_host": parsed.hostname if parsed else "",
                "webhook_configured": bool(self.webhook_url),
                "webhook_secret_configured": bool(self.webhook_secret),
                "enabled": self.enabled,
                "cooldown_seconds": self.cooldown_seconds,
            }

    def notify(self, level, level_code, disease_count=0, pest_count=0,
               disease_ratio=0.0, pest_ratio=0.0, confidence="", image_path=None):
        with self._lock:
            enabled = self.enabled
            webhook_url = self.webhook_url
            cooldown = self.cooldown_seconds
        if not enabled:
            return {"sent": False, "reason": "通知未启用"}
        if not webhook_url:
            return {"sent": False, "reason": "未配置 Webhook URL"}
        if level_code < 2:
            return {"sent": False, "reason": f"等级 {level} 未达到告警阈值(警告+)"}

        now = time.time()
        with self._lock:
            if level in self._pending_alert_levels:
                return {"sent": False, "reason": "同等级告警正在发送"}
            last_time = self._last_alert_time.get(level, 0)
            if now - last_time < cooldown:
                remaining = int(cooldown - (now - last_time))
                return {"sent": False, "reason": f"冷却中，剩余 {remaining} 秒"}
            self._pending_alert_levels.add(level)

        success = False
        try:
            message = self._build_message(level, level_code, disease_count, pest_count,
                                          disease_ratio, pest_ratio, confidence)
            success = self._send_dingtalk(message)
        except (TypeError, ValueError):
            success = False
        finally:
            with self._lock:
                self._pending_alert_levels.discard(level)
                if success:
                    self._last_alert_time[level] = now
                self.alert_history.insert(0, {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "level": level,
                    "level_code": level_code,
                    "disease_count": disease_count,
                    "pest_count": pest_count,
                    "confidence": confidence,
                    "sent": success,
                })
                self.alert_history = self.alert_history[:self.MAX_HISTORY]
        return {"sent": success, "reason": "发送成功" if success else "发送失败"}

    def _build_message(self, level, level_code, disease_count, pest_count,
                       disease_ratio, pest_ratio, confidence):
        emoji = {2: "⚠️", 3: "🚨"}.get(level_code, "⚠️")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""## {emoji} 病虫害告警 - {level}

**时间**: {now_str}

**检测详情**:
- 病斑数量: {disease_count} 个 (占比 {disease_ratio*100:.1f}%)
- 虫害数量: {pest_count} 个 (占比 {pest_ratio*100:.1f}%)
- 融合置信度: {confidence}

> 来自大棚病虫害实时监控系统"""

    def _send_dingtalk(self, text):
        payload = {
            "msgtype": "markdown",
            "markdown": {"title": "病虫害告警", "text": text},
        }
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.webhook_secret:
            signature = hmac.new(self.webhook_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
            headers["X-AgriVision-Signature"] = f"sha256={signature}"
        try:
            # 签名针对实际发送的字节计算；使用 data 避免 requests 重新序列化
            # JSON 后造成接收端验签不一致。
            response = requests.post(self.webhook_url, data=body, headers=headers, timeout=10)
            if response.status_code != 200:
                return False
            result = response.json()
            return result.get("errcode", -1) == 0
        except (OSError, ValueError, TypeError, requests.RequestException):
            return False

    def get_history(self):
        with self._lock:
            return list(self.alert_history)


_notifier = None


def get_notifier():
    global _notifier
    if _notifier is None:
        _notifier = AlertNotifier()
    return _notifier
