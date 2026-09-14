"""System A proxy routes for deep diagnosis."""

import logging

from flask import Blueprint, jsonify, request


def create_diagnosis_blueprint(
    *,
    cameras,
    get_default_camera_id,
    system_a_url,
    system_a_api_token,
    cv2_module,
    requests_module,
    log_event,
    logger=None,
):
    blueprint = Blueprint("diagnosis", __name__)
    logger = logger or logging.getLogger("agrivision.system_b.diagnosis")

    @blueprint.post("/api/deep_diagnose")
    def api_deep_diagnose():
        try:
            params = request.get_json(silent=True) or {}
            camera_id = params.get("camera_id", get_default_camera_id())
            camera = cameras.get(camera_id)
            if not camera:
                return jsonify({"error": "摄像头不存在"}), 404
            with camera["state"]["frame_lock"]:
                frame = camera["state"]["latest_frame"]
                original_image = camera["state"]["latest_original_image"]
            if frame is None:
                return jsonify({"error": "暂无画面，请稍后再试"}), 400
            if original_image is not None:
                encoded, buffer = cv2_module.imencode(".jpg", original_image)
                if not encoded:
                    return jsonify({"success": False, "error": "图像编码失败"}), 500
                image_bytes = buffer.tobytes()
            else:
                image_bytes = frame
            log_event(logger, logging.INFO, "deep_diagnose_request", camera_id=camera_id)
            response = requests_module.post(
                f"{system_a_url}/report",
                files={"file": ("frame.jpg", image_bytes, "image/jpeg")},
                headers={"Authorization": f"Bearer {system_a_api_token}"} if system_a_api_token else {},
                timeout=180,
            )
            if response.status_code == 200:
                data = response.json()
                log_event(logger, logging.INFO, "deep_diagnose_completed", camera_id=camera_id, status=200)
                return jsonify({"success": True, "report": data})
            log_event(logger, logging.WARNING, "deep_diagnose_upstream_failed", camera_id=camera_id, status=response.status_code)
            return jsonify({"success": False, "error": f"System A 返回 {response.status_code}"}), 502
        except requests_module.exceptions.Timeout:
            return jsonify({"success": False, "error": "System A 响应超时（LLM推理可能需要2分钟）"}), 504
        except requests_module.exceptions.ConnectionError:
            return jsonify({"success": False, "error": "无法连接 System A，请确认 System A 已启动"}), 503
        except Exception:
            log_event(logger, logging.ERROR, "deep_diagnose_failed", stage="runtime")
            return jsonify({"success": False, "error": "深度诊断失败"}), 500

    @blueprint.get("/api/system_a/status")
    def api_system_a_status():
        try:
            response = requests_module.get(f"{system_a_url}/health", timeout=5)
            return jsonify({"online": response.status_code == 200})
        except Exception:
            return jsonify({"online": False})

    return blueprint


def register_diagnosis_routes(app, **dependencies):
    app.register_blueprint(create_diagnosis_blueprint(**dependencies))
