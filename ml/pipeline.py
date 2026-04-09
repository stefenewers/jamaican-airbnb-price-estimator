"""
Builds the full sklearn-compatible preprocessing + model pipeline.

Using a Pipeline ensures that every transformation applied at training time
is identically applied at inference — eliminating train/serve skew.
"""
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import (
    AmenityScoreTransformer,
    InputValidator,
    InteractionFeatureTransformer,
    LocationEncoder,
)

logger = logging.getLogger(__name__)


def build_pipeline(model) -> Pipeline:
    """
    Wraps a sklearn-compatible estimator in the full preprocessing pipeline.

    Preprocessing stages (in order):
      1. InputValidator     — range/type checks, clean 422 errors at API layer
      2. AmenityScore       — composite amenity feature
      3. InteractionFeatures — capacity_index, bath_ratio, guest_density
      4. LocationEncoder    — OHE location, aligned to training categories
      5. StandardScaler     — zero-mean / unit-variance for linear models
      6. model              — passed in as the final estimator

    Args:
        model: Any sklearn-compatible regressor.

    Returns:
        An unfitted Pipeline instance.
    """
    logger.info("Building pipeline with estimator: %s", type(model).__name__)

    steps = [
        ("validator", InputValidator()),
        ("amenity_score", AmenityScoreTransformer()),
        ("interactions", InteractionFeatureTransformer()),
        ("location_encoder", LocationEncoder()),
        ("scaler", StandardScaler()),
        ("model", model),
    ]

    return Pipeline(steps=steps)
