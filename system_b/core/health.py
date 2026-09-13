"""Small, dependency-free health payload builders for System B."""


def build_service_health(service, checks):
    """Return a stable readiness summary without exposing configuration secrets."""
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "service": service,
        "status": "ready" if not failed_checks else "degraded",
        "checks": dict(checks),
        "failed_checks": failed_checks,
    }


def summarize_camera(camera_id, state):
    """Expose only operational state, never URLs or camera configuration."""
    return {
        "id": camera_id,
        "has_frame": state.get("latest_frame") is not None,
        "last_error": state.get("last_error") or None,
    }
