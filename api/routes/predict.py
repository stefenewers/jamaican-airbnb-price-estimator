"""
Prediction endpoint.

POST /api/v1/predict
    Request body (JSON):
        {
          "bedrooms": 2,
          "bathrooms": 1,
          "accommodates": 4,
          "has_wifi": 1,
          "has_pool": 0,
          "has_ac": 1,
          "has_kitchen": 1,
          "has_parking": 0,
          "location": "Negril"
        }

    Response:
        {
          "predicted_price_usd": 187.43,
          "currency": "USD",
          "unit": "per night",
          "model": "RandomForest",
          "inputs": { ... }
        }
"""
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd
from flask import Blueprint, jsonify, request

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import MODEL_PATH, METADATA_PATH

logger = logging.getLogger(__name__)
predict_bp = Blueprint("predict", __name__)

# Required fields and their expected types
REQUIRED_FIELDS = {
    "bedrooms": int,
    "bathrooms": int,
    "accommodates": int,
    "has_wifi": int,
    "has_pool": int,
    "has_ac": int,
    "has_kitchen": int,
    "has_parking": int,
    "location": str,
}


@lru_cache(maxsize=1)
def _load_pipeline():
    """Loads and caches the model pipeline. Raises on missing artefact."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python -m ml.train"
        )
    logger.info("Loading pipeline from %s", MODEL_PATH)
    return joblib.load(MODEL_PATH)


def _load_model_name() -> str:
    """Returns the winning model name from training metadata."""
    import json
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            return json.load(f).get("model_name", "unknown")
    return "unknown"


def _validate_request(data: dict) -> Tuple[Optional[dict], Optional[str]]:
    """
    Validates the incoming JSON payload.

    Returns:
        (coerced_data, None) on success
        (None, error_message) on failure
    """
    coerced = {}
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in data:
            return None, f"Missing required field: '{field}'"
        try:
            coerced[field] = expected_type(data[field])
        except (ValueError, TypeError):
            return None, f"Field '{field}' must be of type {expected_type.__name__}"
    return coerced, None


@predict_bp.route("/predict", methods=["POST"])
def predict():
    """
    Single-listing price prediction endpoint.

    Validates input, runs it through the full sklearn Pipeline,
    and returns the predicted nightly rate with request metadata.
    """
    t0 = time.perf_counter()

    # ── Parse body ──────────────────────────────────────────────────────────
    body = request.get_json(silent=True)
    if body is None:
        logger.warning("Predict request with no JSON body")
        return jsonify({"error": "Request body must be valid JSON with Content-Type: application/json"}), 400

    # ── Validate ────────────────────────────────────────────────────────────
    coerced, error = _validate_request(body)
    if error:
        logger.warning("Validation error: %s", error)
        return jsonify({"error": error}), 422

    # ── Load model (cached after first call) ────────────────────────────────
    try:
        pipeline = _load_pipeline()
    except FileNotFoundError as exc:
        logger.error("Model pipeline not found: %s", exc)
        return jsonify({"error": str(exc)}), 503

    # ── Run inference ────────────────────────────────────────────────────────
    try:
        df = pd.DataFrame([coerced])
        raw_prediction = pipeline.predict(df)[0]
        predicted_price = round(float(raw_prediction), 2)
    except ValueError as exc:
        # InputValidator raises ValueError on bad feature values
        logger.warning("Feature validation error during inference: %s", exc)
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:
        logger.exception("Unexpected inference error: %s", exc)
        return jsonify({"error": "Prediction failed. Please check your inputs."}), 500

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(
        "Prediction: $%.2f  location=%s  bedrooms=%d  latency=%.1fms",
        predicted_price,
        coerced["location"],
        coerced["bedrooms"],
        latency_ms,
    )

    return jsonify({
        "predicted_price_usd": predicted_price,
        "currency": "USD",
        "unit": "per night",
        "model": _load_model_name(),
        "latency_ms": latency_ms,
        "inputs": coerced,
    }), 200


@predict_bp.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Batch prediction endpoint. Accepts a list of listing objects.

    POST body: { "listings": [ {...}, {...} ] }
    """
    t0 = time.perf_counter()
    body = request.get_json(silent=True)

    if body is None or "listings" not in body:
        return jsonify({"error": "Body must be JSON with a 'listings' array"}), 400

    listings = body["listings"]
    if not isinstance(listings, list) or len(listings) == 0:
        return jsonify({"error": "'listings' must be a non-empty array"}), 422
    if len(listings) > 100:
        return jsonify({"error": "Maximum 100 listings per batch request"}), 422

    try:
        pipeline = _load_pipeline()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    results = []
    errors = []

    for i, listing in enumerate(listings):
        coerced, error = _validate_request(listing)
        if error:
            errors.append({"index": i, "error": error})
            continue
        try:
            df = pd.DataFrame([coerced])
            price = round(float(pipeline.predict(df)[0]), 2)
            results.append({"index": i, "predicted_price_usd": price, "inputs": coerced})
        except Exception as exc:
            errors.append({"index": i, "error": str(exc)})

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info("Batch prediction: %d ok, %d errors, latency=%.1fms",
                len(results), len(errors), latency_ms)

    return jsonify({
        "results": results,
        "errors": errors,
        "total": len(listings),
        "successful": len(results),
        "latency_ms": latency_ms,
    }), 200
