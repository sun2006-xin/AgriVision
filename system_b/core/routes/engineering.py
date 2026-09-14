"""Engineering endpoints kept separate from legacy device/page routes."""

from flask import Blueprint, Response, jsonify


def create_engineering_blueprint(task_manager, metrics):
    blueprint = Blueprint("engineering", __name__)

    @blueprint.get("/metrics")
    def metrics_endpoint():
        return Response(metrics.prometheus_text(), mimetype="text/plain; version=0.0.4")

    @blueprint.get("/api/tasks")
    def task_status_endpoint():
        return jsonify(task_manager.snapshot())

    return blueprint


def register_engineering_routes(app, task_manager, metrics):
    """Register the isolated engineering routes on the application."""
    app.register_blueprint(create_engineering_blueprint(task_manager, metrics))
