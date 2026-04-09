"""
Health and model-info endpoints.

GET /api/v1/health       — liveness probe (used by Docker/load balancers)
GET /api/v1/model-info   — model metadata, training metrics, feature importance
"""
import json
import logging
from pathlib import Path

from flask import Blueprint, jsonify

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import METADATA_PATH, MODEL_PATH

logger = logging.getLogger(__name__)
health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health():
    """Liveness probe. Returns 200 if the service is running."""
    model_ready = MODEL_PATH.exists()
    status = "healthy" if model_ready else "degraded"
    code = 200 if model_ready else 503
    logger.debug("Health check: %s", status)
    return jsonify({"status": status, "model_loaded": model_ready}), code


@health_bp.route("/model-info", methods=["GET"])
def model_info():
    """Returns training metadata, CV results, and test metrics."""
    if not METADATA_PATH.exists():
        logger.warning("model-info requested but metadata.json not found")
        return jsonify({"error": "Model not trained yet. Run: python -m ml.train"}), 404

    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    logger.info("model-info endpoint hit")
    return jsonify(metadata), 200
