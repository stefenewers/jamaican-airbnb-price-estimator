"""
Model training script with cross-validation and multi-model comparison.

Runs a grid of candidate models, selects the best by CV R², evaluates
on a held-out test set, and persists the winning pipeline + metadata.

Usage:
    python -m ml.train
"""
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

# Add project root to sys.path when run as __main__
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CV_FOLDS,
    DATA_PATH,
    METADATA_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from ml.pipeline import build_pipeline

logger = logging.getLogger(__name__)


# ── Candidate models ───────────────────────────────────────────────────────────
def get_candidate_models() -> dict:
    """Returns candidate regressors keyed by display name."""
    candidates = {
        "LinearRegression": LinearRegression(),
        "Ridge(alpha=1)": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Ridge(alpha=10)": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBRegressor
        candidates["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        logger.info("XGBoost available — included in model comparison")
    except Exception:
        logger.warning("XGBoost unavailable (missing libomp?) — skipping. Run: brew install libomp")

    return candidates


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes a comprehensive regression metric suite."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # Mean Absolute Percentage Error — interpretable as "% off on average"
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "mape": round(float(mape), 4),
    }


# ── Main training loop ─────────────────────────────────────────────────────────
def train(data_path: Path = DATA_PATH) -> dict:
    """
    Full training run: loads data, compares models via CV,
    evaluates winner on held-out test set, persists artefacts.

    Returns:
        Metadata dict written to models/metadata.json.
    """
    logger.info("Loading dataset from %s", data_path)
    df = pd.read_csv(data_path)
    logger.info("Dataset shape: %s", df.shape)

    # Basic data quality checks
    if df.isnull().any().any():
        null_counts = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
        logger.warning("Null values detected: %s", null_counts)
        df = df.dropna()
        logger.info("Dropped null rows — new shape: %s", df.shape)

    X = df.drop(columns=["price"])
    y = df["price"]

    logger.info("Price statistics: mean=%.2f  std=%.2f  min=%.2f  max=%.2f",
                y.mean(), y.std(), y.min(), y.max())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info("Train/test split: %d / %d rows", len(X_train), len(X_test))

    # ── Cross-validation comparison ────────────────────────────────────────────
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    candidates = get_candidate_models()
    cv_results = {}

    logger.info("Running %d-fold CV across %d candidate models...", CV_FOLDS, len(candidates))

    for name, estimator in candidates.items():
        t0 = time.time()
        pipeline = build_pipeline(estimator)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=kf,
            scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
            return_train_score=False,
            n_jobs=1,
        )
        elapsed = time.time() - t0
        mean_r2 = float(np.mean(scores["test_r2"]))
        std_r2 = float(np.std(scores["test_r2"]))
        mean_mae = float(-np.mean(scores["test_neg_mean_absolute_error"]))
        mean_rmse = float(-np.mean(scores["test_neg_root_mean_squared_error"]))

        cv_results[name] = {
            "cv_r2_mean": round(mean_r2, 4),
            "cv_r2_std": round(std_r2, 4),
            "cv_mae_mean": round(mean_mae, 4),
            "cv_rmse_mean": round(mean_rmse, 4),
            "fit_time_s": round(elapsed, 2),
        }
        logger.info(
            "  %-22s  CV R²=%.4f ± %.4f  MAE=$%.2f  RMSE=$%.2f  [%.1fs]",
            name, mean_r2, std_r2, mean_mae, mean_rmse, elapsed,
        )

    # ── Select best model by CV R² ─────────────────────────────────────────────
    best_name = max(cv_results, key=lambda k: cv_results[k]["cv_r2_mean"])
    logger.info("Best model: %s (CV R²=%.4f)", best_name, cv_results[best_name]["cv_r2_mean"])

    # ── Re-fit best model on full training set ─────────────────────────────────
    best_estimator = candidates[best_name]
    best_pipeline = build_pipeline(best_estimator)
    best_pipeline.fit(X_train, y_train)

    # ── Held-out test evaluation ───────────────────────────────────────────────
    y_pred = best_pipeline.predict(X_test)
    test_metrics = compute_metrics(y_test.to_numpy(), y_pred)
    logger.info(
        "Test set — R²=%.4f  MAE=$%.2f  RMSE=$%.2f  MAPE=%.2f%%",
        test_metrics["r2"], test_metrics["mae"], test_metrics["rmse"], test_metrics["mape"],
    )

    # ── Feature importance (tree models only) ─────────────────────────────────
    feature_importance = None
    inner_model = best_pipeline.named_steps["model"]
    if hasattr(inner_model, "feature_importances_"):
        # Get feature names post-encoding
        encoder = best_pipeline.named_steps["location_encoder"]
        base_cols = [c for c in X_train.columns if c != "location"]
        engineered = ["amenity_score", "capacity_index", "bath_ratio", "guest_density"]
        loc_dummies = encoder.dummy_columns_ if hasattr(encoder, "dummy_columns_") else []
        all_features = base_cols + engineered + loc_dummies
        importances = inner_model.feature_importances_
        if len(importances) == len(all_features):
            feature_importance = dict(
                sorted(
                    zip(all_features, importances.tolist()),
                    key=lambda x: x[1], reverse=True,
                )
            )

    # ── Persist artefacts ──────────────────────────────────────────────────────
    joblib.dump(best_pipeline, MODEL_PATH)
    logger.info("Pipeline saved to %s", MODEL_PATH)

    metadata = {
        "model_name": best_name,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "features": X.columns.tolist(),
        "cv_folds": CV_FOLDS,
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "feature_importance": feature_importance,
        "model_path": str(MODEL_PATH),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", METADATA_PATH)

    return metadata


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Configure logging for CLI use
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(__file__).parent.parent / "logs" / "train.log"),
        ],
    )

    metadata = train()

    print("\n" + "=" * 60)
    print(f"  Winner: {metadata['model_name']}")
    print(f"  Test R²:   {metadata['test_metrics']['r2']:.4f}")
    print(f"  Test MAE:  ${metadata['test_metrics']['mae']:.2f}")
    print(f"  Test RMSE: ${metadata['test_metrics']['rmse']:.2f}")
    print(f"  Test MAPE: {metadata['test_metrics']['mape']:.2f}%")
    print("=" * 60)
