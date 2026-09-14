"""Offline event queue and transport routes for System B."""

from flask import Blueprint, jsonify, request


def create_event_blueprint(
    *,
    offline_event_cache,
    project_event,
    event_transport,
    offline_event_sync_lock,
    validate_sync_request,
    get_mqtt_transport,
    build_mqtt_config_status,
    get_mqtt_settings,
    event_sync_status,
    get_scheduler,
    events_sync_interval,
):
    blueprint = Blueprint("events", __name__)

    def _validate_sync_body():
        payload = request.get_json(silent=True)
        try:
            return validate_sync_request(payload)
        except ValueError:
            message = (
                "limit must be an integer from 1 to 100"
                if isinstance(payload, dict) and "limit" in payload
                else "request body must be a JSON object"
            )
            raise ValueError(message)

    @blueprint.get("/api/offline_events")
    def api_offline_events():
        events = offline_event_cache.list_pending()
        safe_events = [project_event(event) for event in events if isinstance(event, dict)]
        return jsonify({"count": len(safe_events), "events": safe_events})

    @blueprint.post("/api/offline_events/ack")
    def api_offline_events_ack():
        params = request.get_json(silent=True) or {}
        try:
            offline_event_cache.ack(params.get("event_id"))
        except ValueError as error:
            return jsonify({"success": False, "error": str(error)}), 400
        return jsonify({"success": True})

    @blueprint.post("/api/offline_events/sync")
    def api_offline_events_sync():
        if event_transport is None:
            return jsonify({"success": False, "error": "event sink is not configured"}), 503
        if not offline_event_sync_lock.acquire(blocking=False):
            return jsonify({"success": False, "error": "event sync is already running"}), 409
        try:
            try:
                params = _validate_sync_body()
            except ValueError as error:
                return jsonify({"success": False, "error": str(error)}), 400
            events = offline_event_cache.list_pending()[:params["limit"]]
            result = event_transport.sync(events, offline_event_cache.ack)
            pending = len(offline_event_cache.list_pending())
            return jsonify({"success": result["sent"] == len(events), "pending": pending, **result})
        finally:
            offline_event_sync_lock.release()

    @blueprint.post("/api/offline_events/sync_mqtt")
    def api_offline_events_sync_mqtt():
        if not offline_event_sync_lock.acquire(blocking=False):
            return jsonify({"success": False, "error": "event sync is already running"}), 409
        try:
            try:
                params = _validate_sync_body()
            except ValueError as error:
                return jsonify({"success": False, "error": str(error)}), 400
            try:
                transport = get_mqtt_transport()
            except Exception:
                settings = get_mqtt_settings()
                mqtt_config = build_mqtt_config_status(
                    settings["broker_url"], settings["topic"], settings["client_id"],
                    settings["username"], settings["password"], settings["ca_certs"],
                )
                if mqtt_config["configured"]:
                    event_sync_status.record_mqtt_runtime("connection_failed", "runtime")
                    error_code = "mqtt_connection_failed"
                else:
                    event_sync_status.record_mqtt_runtime("not_configured")
                    error_code = "mqtt_not_configured"
                return jsonify({
                    "success": False,
                    "error": "mqtt sync unavailable",
                    "error_code": error_code,
                }), 503

            events = offline_event_cache.list_pending()[:params["limit"]]
            if not events:
                event_sync_status.record("empty", "mqtt", 0)
                event_sync_status.record_mqtt_runtime("connected")
                return jsonify({"success": True, "pending": 0, "sent": 0, "acked": 0, "attempts": 0})
            result = transport.sync(events, offline_event_cache.ack)
            pending = len(offline_event_cache.list_pending())
            outcome = "success" if result["sent"] == len(events) else "failure"
            event_sync_status.record(
                outcome,
                "mqtt",
                pending,
                sent=result.get("sent", 0),
                acked=result.get("acked", 0),
                failure_type=result.get("failure_type", "") if outcome == "failure" else "",
            )
            event_sync_status.record_mqtt_runtime(
                "connected" if outcome == "success" else "publish_failed",
                result.get("failure_type", "retry_exhausted") if outcome == "failure" else "",
            )
            return jsonify({"success": result["sent"] == len(events), "pending": pending, **result})
        finally:
            offline_event_sync_lock.release()

    @blueprint.get("/api/offline_events/sync_status")
    def api_offline_events_sync_status():
        settings = get_mqtt_settings()
        transport_name = "mqtt" if settings["broker_url"] else "http" if event_transport is not None else "none"
        scheduler = get_scheduler()
        return jsonify(event_sync_status.snapshot(
            enabled=events_sync_interval > 0 and transport_name != "none",
            interval_seconds=events_sync_interval,
            transport=transport_name,
            running=scheduler.running if scheduler is not None else False,
            pending=len(offline_event_cache.list_pending()),
            mqtt_connack_timeout=settings["connack_timeout"] if transport_name == "mqtt" else None,
            mqtt_publish_timeout=settings["publish_timeout"] if transport_name == "mqtt" else None,
            mqtt_config=build_mqtt_config_status(
                settings["broker_url"], settings["topic"], settings["client_id"],
                settings["username"], settings["password"], settings["ca_certs"],
            ),
        ))

    return blueprint


def register_event_routes(app, **dependencies):
    app.register_blueprint(create_event_blueprint(**dependencies))
