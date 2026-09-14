"""Camera storage and per-camera task trigger routes for System B."""

import os

from flask import Blueprint, jsonify, request, send_from_directory


def create_storage_blueprint(
    data_set_dir,
    cameras,
    get_default_camera_id,
    task_manager,
    save_to_sd,
    sync_sd,
    save_state_lock,
    get_save_state,
    sync_state_lock,
    get_sync_state,
    filter_camera_filenames,
):
    blueprint = Blueprint("storage", __name__)

    @blueprint.get("/api/sd_images")
    def api_sd_images():
        if not os.path.exists(data_set_dir):
            return jsonify([])
        names = [
            name for name in os.listdir(data_set_dir)
            if os.path.isfile(os.path.join(data_set_dir, name))
        ]
        return jsonify(sorted(filter_camera_filenames(names), reverse=True))

    @blueprint.get("/dataset/<path:filename>")
    def serve_dataset_image(filename):
        safe_names = filter_camera_filenames([filename])
        if not safe_names:
            return "图片不存在，请先同步", 404
        return send_from_directory(data_set_dir, safe_names[0])

    @blueprint.get("/api/save_to_sd")
    def api_save_to_sd():
        camera_id = request.args.get("camera_id", get_default_camera_id())
        if camera_id not in cameras:
            return jsonify({"started": False, "message": "摄像头不存在"}), 404
        started = task_manager.submit(
            camera_id,
            "save_to_sd",
            lambda camera_id=camera_id: save_to_sd(camera_id),
        )
        return jsonify({
            "started": started,
            "message": "拍照已启动" if started else "已有拍照任务在执行",
        })

    @blueprint.get("/api/save_status")
    def api_save_status():
        camera_id = request.args.get("camera_id", get_default_camera_id())
        if camera_id not in cameras:
            return jsonify({"error": "摄像头不存在"}), 404
        with save_state_lock:
            return jsonify(dict(get_save_state(camera_id)))

    @blueprint.get("/api/sync_now")
    def api_sync_now():
        camera_id = request.args.get("camera_id", get_default_camera_id())
        if camera_id not in cameras:
            return jsonify({"started": False, "message": "摄像头不存在"}), 404
        started = task_manager.submit(
            camera_id,
            "sd_sync",
            lambda camera_id=camera_id: sync_sd(camera_id),
        )
        return jsonify({
            "started": started,
            "message": "同步已启动" if started else "同步已在进行中",
        })

    @blueprint.get("/api/sync_status")
    def api_sync_status():
        camera_id = request.args.get("camera_id", get_default_camera_id())
        if camera_id not in cameras:
            return jsonify({"error": "摄像头不存在"}), 404
        with sync_state_lock:
            return jsonify(get_sync_state(camera_id).copy())

    return blueprint


def register_storage_routes(app, **dependencies):
    app.register_blueprint(create_storage_blueprint(**dependencies))
