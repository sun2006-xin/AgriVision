"""MJPEG video stream routes."""

import time

from flask import Blueprint, Response


def create_video_blueprint(cameras, get_default_camera_id, create_wait_image):
    blueprint = Blueprint("video", __name__)

    @blueprint.get("/video_feed/<camera_id>")
    def video_feed(camera_id):
        camera = cameras.get(camera_id)
        if not camera:
            return "Camera not found", 404
        state = camera["state"]

        def generate():
            while True:
                with state["frame_lock"]:
                    frame = state["latest_frame"]
                    error = state["last_error"]
                if frame:
                    payload = frame
                else:
                    payload = create_wait_image(error or "等待检测中...")
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
                time.sleep(0.5)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @blueprint.get("/video_feed")
    def video_feed_default():
        return video_feed(get_default_camera_id())

    return blueprint


def register_video_routes(app, **dependencies):
    app.register_blueprint(create_video_blueprint(**dependencies))
