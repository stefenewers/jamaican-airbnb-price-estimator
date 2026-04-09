"""
Custom sklearn-compatible transformers for feature engineering.

All transformers follow the fit/transform protocol so they can be
composed into a Pipeline — eliminating train/serve skew entirely.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class AmenityScoreTransformer(BaseEstimator, TransformerMixin):
    """
    Aggregates binary amenity flags into a single composite score.

    Captures the intuition that each additional amenity incrementally
    raises the listing's perceived value without inflating dimensionality.
    """

    AMENITY_COLS = ["has_wifi", "has_pool", "has_ac", "has_kitchen", "has_parking"]
    # Relative weights — pool/AC command a premium in Jamaica's climate
    WEIGHTS = {"has_wifi": 1.0, "has_pool": 2.0, "has_ac": 1.5, "has_kitchen": 1.0, "has_parking": 0.8}

    def fit(self, X: pd.DataFrame, y=None) -> "AmenityScoreTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        present_cols = [c for c in self.AMENITY_COLS if c in X.columns]
        X["amenity_score"] = sum(
            X[col] * self.WEIGHTS.get(col, 1.0) for col in present_cols
        )
        logger.debug("AmenityScoreTransformer: added amenity_score column")
        return X


class InteractionFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Creates interaction and ratio features that capture non-linear relationships.

    - capacity_index: accommodates / bedrooms  (space efficiency)
    - bath_ratio:     bathrooms / bedrooms     (luxury indicator)
    - guest_density:  accommodates * amenity_score (demand × quality)
    """

    def fit(self, X: pd.DataFrame, y=None) -> "InteractionFeatureTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        beds = X["bedrooms"].clip(lower=1)
        X["capacity_index"] = X["accommodates"] / beds
        X["bath_ratio"] = X["bathrooms"] / beds

        if "amenity_score" in X.columns:
            X["guest_density"] = X["accommodates"] * X["amenity_score"]

        logger.debug("InteractionFeatureTransformer: added capacity_index, bath_ratio, guest_density")
        return X


class LocationEncoder(BaseEstimator, TransformerMixin):
    """
    One-hot encodes the location column, dropping the first category
    (Kingston) to avoid multicollinearity.

    Fit captures the category list so unseen locations at inference
    time are handled gracefully rather than causing a column mismatch.
    """

    def __init__(self, drop_first: bool = True):
        self.drop_first = drop_first
        self.location_categories_: Optional[list] = None
        self.dummy_columns_: Optional[list] = None

    def fit(self, X: pd.DataFrame, y=None) -> "LocationEncoder":
        self.location_categories_ = sorted(X["location"].unique().tolist())
        dummies = pd.get_dummies(
            pd.Series(self.location_categories_, name="location"),
            drop_first=self.drop_first,
        )
        self.dummy_columns_ = dummies.columns.tolist()
        logger.info(
            "LocationEncoder fitted on %d categories: %s",
            len(self.location_categories_),
            self.location_categories_,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        dummies = pd.get_dummies(X["location"], drop_first=self.drop_first)

        # Align to training columns — fills zeros for any unseen locations
        for col in self.dummy_columns_:
            if col not in dummies.columns:
                dummies[col] = 0
        dummies = dummies[self.dummy_columns_]

        X = X.drop(columns=["location"])
        X = pd.concat([X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        return X


class InputValidator(BaseEstimator, TransformerMixin):
    """
    Validates and coerces raw input before any transformation.

    Raises ValueError on out-of-range inputs so the API can return
    a clean 422 rather than a cryptic model error downstream.
    """

    CONSTRAINTS = {
        "bedrooms": (1, 10),
        "bathrooms": (1, 10),
        "accommodates": (1, 16),
    }
    BINARY_COLS = ["has_wifi", "has_pool", "has_ac", "has_kitchen", "has_parking"]
    VALID_LOCATIONS = ["Kingston", "Montego Bay", "Negril", "Ocho Rios", "Portland"]

    def fit(self, X: pd.DataFrame, y=None) -> "InputValidator":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col, (lo, hi) in self.CONSTRAINTS.items():
            if col in X.columns:
                if not ((X[col] >= lo) & (X[col] <= hi)).all():
                    raise ValueError(f"'{col}' must be between {lo} and {hi}")

        for col in self.BINARY_COLS:
            if col in X.columns:
                if not X[col].isin([0, 1]).all():
                    raise ValueError(f"'{col}' must be 0 or 1")

        if "location" in X.columns:
            invalid = set(X["location"].unique()) - set(self.VALID_LOCATIONS)
            if invalid:
                raise ValueError(f"Unknown location(s): {invalid}. Valid: {self.VALID_LOCATIONS}")

        return X
