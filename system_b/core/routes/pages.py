"""HTML page routes kept separate from the application bootstrap."""

from flask import Blueprint


def create_pages_blueprint(html_page, dashboard_page):
    blueprint = Blueprint("pages", __name__)

    @blueprint.get("/")
    def index():
        return html_page

    @blueprint.get("/dashboard")
    def dashboard():
        return dashboard_page

    return blueprint


def register_page_routes(app, html_page, dashboard_page):
    app.register_blueprint(create_pages_blueprint(html_page, dashboard_page))
