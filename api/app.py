"""
Flask application factory.

Using the factory pattern keeps the app instance out of module scope,
which simplifies testing and allows multiple configurations (dev/prod/test).
"""
import logging
import sys
from pathlib import Path

from flask import Flask, jsonify, send_from_directory

# Ensure project root is on sys.path regardless of invocation method
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import API_PORT, API_VERSION, LOG_DATE_FORMAT, LOG_FORMAT, LOG_LEVEL, LOGS_DIR, ROOT_DIR


def configure_logging(app: Flask) -> None:
    """Sets up structured logging to stdout and a rotating file."""
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # stdout handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # Apply to root logger so all modules inherit it
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(stream_handler)

    # File handler — optional, skipped if directory isn't writable (e.g. Railway)
    try:
        log_file = LOGS_DIR / "api.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        pass  # stdout-only logging is fine for cloud deployments

    app.logger.setLevel(level)


def register_blueprints(app: Flask) -> None:
    from api.routes.health import health_bp
    from api.routes.predict import predict_bp

    prefix = f"/api/{API_VERSION}"
    app.register_blueprint(health_bp, url_prefix=prefix)
    app.register_blueprint(predict_bp, url_prefix=prefix)


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "Bad request", "detail": str(e)}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"error": "Method not allowed"}), 405

    @app.errorhandler(422)
    def unprocessable(e):
        return jsonify({"error": "Unprocessable entity", "detail": str(e)}), 422

    @app.errorhandler(500)
    def internal_error(e):
        app.logger.exception("Unhandled server error: %s", e)
        return jsonify({"error": "Internal server error"}), 500


def register_frontend(app: Flask) -> None:
    """Serves the demo frontend at the root URL."""
    frontend_dir = ROOT_DIR / "frontend"

    @app.route("/")
    def index():
        return send_from_directory(str(frontend_dir), "index.html")

    @app.route("/<path:filename>")
    def static_files(filename):
        return send_from_directory(str(frontend_dir), filename)


def create_app() -> Flask:
    """
    Application factory. Creates and fully configures a Flask app instance.

    Returns:
        A configured Flask application.
    """
    app = Flask(__name__)

    configure_logging(app)
    register_blueprints(app)
    register_error_handlers(app)
    register_frontend(app)

    app.logger.info("Jamaican Airbnb Price Predictor API started (port %d)", API_PORT)
    return app
