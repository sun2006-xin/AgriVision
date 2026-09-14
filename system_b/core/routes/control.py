"""Control-plane routes for detector, alert and parameter configuration."""

import logging

from flask import Blueprint, Response, jsonify, request


def create_control_blueprint(
    *,
    cameras,
    get_default_camera_id,
    config_manager,
    config_lock,
    yolo_detector,
    get_yolo_enabled,
    set_yolo_enabled,
    dual_verifier,
    alert_notifier,
    validate_yolo_patch,
    validate_alert_patch,
    issue_session,
    session_cookie_name,
    session_ttl_seconds,
    generate_mask_image,
    log_event,
    logger=None,
):
    blueprint = Blueprint("control", __name__)
    logger = logger or logging.getLogger("agrivision.system_b.control")

    @blueprint.post("/api/auth/session")
    def api_auth_session():
        """Exchange a bearer token for a short-lived cookie used by media tags."""
        session_id = issue_session(request.remote_addr, request.headers)
        if session_id is None:
            return jsonify({"authenticated": False}), 401
        response = jsonify({"authenticated": True, "expires_in": session_ttl_seconds})
        response.set_cookie(
            session_cookie_name,
            session_id,
            max_age=session_ttl_seconds,
            httponly=True,
            samesite="Strict",
            secure=bool(request.is_secure),
        )
        return response

    @blueprint.post("/api/yolo/config")
    def api_yolo_config():
        try:
            params = validate_yolo_patch(request.get_json(silent=True))
            if "enabled" in params:
                set_yolo_enabled(params["enabled"])
            if "conf_threshold" in params or "iou_threshold" in params:
                yolo_detector.update_config(
                    conf_threshold=params.get("conf_threshold"),
                    iou_threshold=params.get("iou_threshold"),
                )
            if "dual_yolo_conf_high" in params:
                dual_verifier.yolo_conf_high = float(params["dual_yolo_conf_high"])
            if "dual_yolo_conf_low" in params:
                dual_verifier.yolo_conf_low = float(params["dual_yolo_conf_low"])
            return jsonify({"success": True, "message": "YOLO 参数已更新", "enabled": get_yolo_enabled()})
        except ValueError as error:
            return jsonify({"success": False, "message": str(error)}), 400
        except Exception:
            return jsonify({"success": False, "message": "YOLO 参数更新失败"}), 500

    @blueprint.route("/api/alert/config", methods=["GET", "POST"])
    def api_alert_config():
        if request.method == "GET":
            return jsonify(alert_notifier.get_config())
        try:
            params = validate_alert_patch(request.get_json(silent=True))
            result = alert_notifier.configure(
                webhook_url=params.get("webhook_url"),
                enabled=params.get("enabled"),
                cooldown=params.get("cooldown_seconds"),
            )
            return jsonify({"success": True, "config": result})
        except ValueError as error:
            return jsonify({"success": False, "message": str(error)}), 400
        except Exception:
            return jsonify({"success": False, "message": "告警配置更新失败"}), 500

    @blueprint.get("/api/alert/history")
    def api_alert_history():
        return jsonify({"history": alert_notifier.get_history()})

    @blueprint.post("/api/alert/test")
    def api_alert_test():
        try:
            result = alert_notifier.notify(
                level="测试",
                level_code=2,
                disease_count=3,
                pest_count=1,
                disease_ratio=0.15,
                pest_ratio=0.05,
                confidence="high",
            )
            return jsonify({"success": result.get("sent", False), "reason": result.get("reason", "")})
        except Exception:
            return jsonify({"success": False, "message": "测试告警失败"}), 500

    @blueprint.route("/api/params", methods=["GET", "POST"])
    def api_params():
        if request.method == "GET":
            with config_lock:
                return jsonify(config_manager.get_current_config())
        try:
            params = request.get_json(silent=True)
            with config_lock:
                is_valid, errors = config_manager.validate_params(params)
                if not is_valid:
                    return jsonify({"success": False, "errors": errors}), 400
                config_manager.update_params(params)
            return jsonify({"success": True, "message": "参数已保存"})
        except ValueError as error:
            return jsonify({"success": False, "errors": [str(error)]}), 400
        except Exception:
            return jsonify({"success": False, "message": "参数保存失败"}), 500

    @blueprint.get("/api/params/info")
    def api_params_info():
        return jsonify(config_manager.get_param_info())

    @blueprint.post("/api/params/reset")
    def api_params_reset():
        try:
            with config_lock:
                config_manager.reset_config()
                params = config_manager.get_current_config()
            return jsonify({"success": True, "params": params})
        except ValueError as error:
            return jsonify({"success": False, "errors": [str(error)]}), 400
        except Exception:
            return jsonify({"success": False, "message": "参数重置失败"}), 500

    @blueprint.post("/api/preview_mask")
    def api_preview_mask():
        try:
            camera_id = request.args.get("camera_id", get_default_camera_id())
            camera = cameras.get(camera_id)
            if not camera:
                return jsonify({"error": "摄像头不存在"}), 404
            data = request.get_json(silent=True)
            if not isinstance(data, dict):
                return jsonify({"error": "请求体必须是 JSON 对象"}), 400
            mask_type = data.get("type", "leaf")
            if mask_type not in {"leaf", "disease", "pest", "annotated"}:
                return jsonify({"error": "不支持的预览类型"}), 400
            params = data.get("params", {})
            with config_lock:
                is_valid, errors = config_manager.validate_params(params)
            if not is_valid:
                return jsonify({"error": "参数校验失败", "errors": errors}), 400
            with camera["state"]["frame_lock"]:
                image = camera["state"]["latest_original_image"]
            if image is None:
                return jsonify({"error": "暂无图像"}), 400
            mask_data = generate_mask_image(image, params, mask_type)
            return Response(mask_data, mimetype="image/jpeg")
        except Exception:
            log_event(logger, logging.ERROR, "mask_preview_failed")
            return jsonify({"error": "预览生成失败"}), 500

    return blueprint


def register_control_routes(app, **dependencies):
    app.register_blueprint(create_control_blueprint(**dependencies))
