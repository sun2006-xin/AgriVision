"""Health and read-only monitoring routes for System B."""

from flask import Blueprint, jsonify, request


def create_monitoring_blueprint(
    cameras,
    get_default_camera_id,
    get_sd_sync_info,
    sd_lock,
    yolo_detector,
    get_yolo_enabled,
    summarize_camera,
    build_service_health,
    comparison_history_size=100,
):
    """Create read-only health, camera and inference-status endpoints.

    Dependencies are passed in explicitly so route behavior can be tested
    without importing the model-loading application module.
    """
    blueprint = Blueprint("monitoring", __name__)

    @blueprint.get("/health/live")
    def health_live():
        return jsonify({"status": "alive", "service": "system-b"})

    @blueprint.get("/health/ready")
    def health_ready():
        camera_summaries = []
        for camera_id, camera in cameras.items():
            with camera["state"]["frame_lock"]:
                camera_summaries.append(summarize_camera(camera_id, camera["state"]))

        camera_ready = any(
            summary["has_frame"]
            for summary in camera_summaries
            if cameras[summary["id"]].get("enabled", True)
        )
        yolo_ready = (not get_yolo_enabled()) or yolo_detector.is_loaded()
        payload = build_service_health("system-b", {"camera": camera_ready, "yolo": yolo_ready})
        payload["cameras"] = camera_summaries
        return jsonify(payload), (200 if payload["status"] == "ready" else 503)

    @blueprint.get("/health")
    def health():
        return health_ready()

    @blueprint.get("/api/cameras")
    def api_cameras():
        result = {}
        for camera_id, camera in cameras.items():
            result[camera_id] = {
                "id": camera["id"],
                "name": camera["name"],
                "enabled": camera.get("enabled", True),
            }
        return jsonify({"cameras": result})

    @blueprint.get("/api/status")
    def api_status():
        camera_id = request.args.get("camera_id", get_default_camera_id())
        camera = cameras.get(camera_id)
        if not camera:
            return jsonify({"error": "摄像头不存在"}), 404
        state = camera["state"]
        with state["frame_lock"]:
            data = state["latest_result"].copy()
        data["error"] = state["last_error"]
        with sd_lock:
            data["sd_sync"] = get_sd_sync_info(camera_id).copy()
        return jsonify(data)

    @blueprint.get("/api/dual_status")
    def api_dual_status():
        camera_id = request.args.get("camera_id", get_default_camera_id())
        camera = cameras.get(camera_id)
        if not camera:
            return jsonify({"error": "摄像头不存在"}), 404
        state = camera["state"]
        with state["frame_lock"]:
            data = state["latest_result"].copy()
            data["dual"] = state["latest_dual_result"]
            data["yolo_enabled"] = get_yolo_enabled()
            data["yolo_loaded"] = yolo_detector.is_loaded()
        data["error"] = state["last_error"]
        with sd_lock:
            data["sd_sync"] = get_sd_sync_info(camera_id).copy()
        return jsonify(data)

    @blueprint.get("/api/yolo_stats")
    def api_yolo_stats():
        return jsonify(yolo_detector.get_stats())

    @blueprint.get("/api/comparison")
    def api_comparison():
        camera_id = request.args.get("camera_id", get_default_camera_id())
        camera = cameras.get(camera_id)
        if not camera:
            return jsonify({"frames": [], "total": 0})
        state = camera["state"]
        n = request.args.get("n", 50, type=int)
        n = max(0, min(n, comparison_history_size))
        with state["frame_lock"]:
            history_copy = list(state["comparison_history"])
        return jsonify({"frames": history_copy[-n:] if n else [], "total": len(history_copy)})

    return blueprint


def register_monitoring_routes(app, **dependencies):
    """Register health and read-only monitoring endpoints on the Flask app."""
    app.register_blueprint(create_monitoring_blueprint(**dependencies))
