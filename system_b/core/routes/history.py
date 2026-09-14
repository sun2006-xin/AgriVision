"""History query and export routes for System B."""

import os

from flask import Blueprint, jsonify, request, send_file


def create_history_blueprint(history_manager):
    blueprint = Blueprint("history", __name__)

    @blueprint.get("/api/history")
    def api_history():
        try:
            page = int(request.args.get("page", 1))
            date_from = request.args.get("date_from")
            date_to = request.args.get("date_to")
            level = request.args.get("level") or None
            return jsonify(history_manager.query_records(date_from, date_to, level, page))
        except (TypeError, ValueError):
            return jsonify({"error": "history query parameters are invalid"}), 400
        except Exception:
            return jsonify({"error": "history query failed"}), 500

    @blueprint.get("/api/history/trend")
    def api_history_trend():
        try:
            days = int(request.args.get("days", 7))
            return jsonify(history_manager.get_trend_data(days))
        except (TypeError, ValueError):
            return jsonify({"error": "history trend parameters are invalid"}), 400
        except Exception:
            return jsonify({"error": "history trend failed"}), 500

    @blueprint.get("/api/history/statistics")
    def api_history_statistics():
        try:
            return jsonify(history_manager.get_statistics(
                request.args.get("date_from"),
                request.args.get("date_to"),
            ))
        except Exception:
            return jsonify({"error": "history statistics failed"}), 500

    @blueprint.get("/api/history/export")
    def api_history_export():
        try:
            zip_path = history_manager.export_data(
                request.args.get("date_from"),
                request.args.get("date_to"),
            )
            if not zip_path or not os.path.exists(zip_path):
                return jsonify({"error": "导出失败"}), 500
            return send_file(
                zip_path,
                as_attachment=True,
                download_name=os.path.basename(zip_path),
            )
        except Exception:
            return jsonify({"error": "history export failed"}), 500

    @blueprint.get("/history/image/<record_id>")
    def serve_history_image(record_id):
        record = history_manager.get_record_by_id(record_id)
        if record and record.get("image_path") and os.path.isfile(record["image_path"]):
            return send_file(record["image_path"], mimetype="image/jpeg")
        return "图片不存在", 404

    return blueprint


def register_history_routes(app, history_manager):
    app.register_blueprint(create_history_blueprint(history_manager))
