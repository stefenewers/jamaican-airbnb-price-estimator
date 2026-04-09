"""
Unit tests for custom sklearn transformers in ml/features.py.
"""
import pandas as pd
import pytest

from ml.features import (
    AmenityScoreTransformer,
    InputValidator,
    InteractionFeatureTransformer,
    LocationEncoder,
)


class TestAmenityScoreTransformer:
    def test_adds_amenity_score_column(self, sample_df):
        t = AmenityScoreTransformer()
        out = t.fit_transform(sample_df)
        assert "amenity_score" in out.columns

    def test_score_is_nonnegative(self, sample_df):
        t = AmenityScoreTransformer()
        out = t.fit_transform(sample_df)
        assert (out["amenity_score"] >= 0).all()

    def test_max_score_all_amenities(self):
        df = pd.DataFrame([{
            "bedrooms": 1, "bathrooms": 1, "accommodates": 2, "location": "Negril",
            "has_wifi": 1, "has_pool": 1, "has_ac": 1, "has_kitchen": 1, "has_parking": 1,
        }])
        out = AmenityScoreTransformer().fit_transform(df)
        expected = 1.0 + 2.0 + 1.5 + 1.0 + 0.8
        assert abs(out["amenity_score"].iloc[0] - expected) < 1e-6

    def test_zero_score_no_amenities(self):
        df = pd.DataFrame([{
            "bedrooms": 1, "bathrooms": 1, "accommodates": 2, "location": "Negril",
            "has_wifi": 0, "has_pool": 0, "has_ac": 0, "has_kitchen": 0, "has_parking": 0,
        }])
        out = AmenityScoreTransformer().fit_transform(df)
        assert out["amenity_score"].iloc[0] == 0.0

    def test_does_not_modify_original(self, sample_df):
        original_cols = set(sample_df.columns)
        AmenityScoreTransformer().fit_transform(sample_df)
        assert set(sample_df.columns) == original_cols


class TestInteractionFeatureTransformer:
    def test_adds_capacity_index(self, sample_df):
        df = AmenityScoreTransformer().fit_transform(sample_df)
        out = InteractionFeatureTransformer().fit_transform(df)
        assert "capacity_index" in out.columns

    def test_adds_bath_ratio(self, sample_df):
        df = AmenityScoreTransformer().fit_transform(sample_df)
        out = InteractionFeatureTransformer().fit_transform(df)
        assert "bath_ratio" in out.columns

    def test_capacity_index_value(self, sample_df):
        # bedrooms=2, accommodates=4 → capacity_index = 2.0
        df = AmenityScoreTransformer().fit_transform(sample_df)
        out = InteractionFeatureTransformer().fit_transform(df)
        assert abs(out["capacity_index"].iloc[0] - 2.0) < 1e-6

    def test_no_division_by_zero(self):
        df = pd.DataFrame([{
            "bedrooms": 0, "bathrooms": 1, "accommodates": 2, "location": "Kingston",
            "has_wifi": 1, "has_pool": 0, "has_ac": 0, "has_kitchen": 1, "has_parking": 0,
        }])
        df = AmenityScoreTransformer().fit_transform(df)
        # Should not raise
        out = InteractionFeatureTransformer().fit_transform(df)
        assert out["capacity_index"].iloc[0] >= 0


class TestLocationEncoder:
    def test_removes_location_column(self, sample_df):
        enc = LocationEncoder().fit(sample_df)
        out = enc.transform(sample_df)
        assert "location" not in out.columns

    def test_adds_dummy_columns(self, sample_df):
        enc = LocationEncoder().fit(sample_df)
        out = enc.transform(sample_df)
        # At least some location_ columns should appear
        loc_cols = [c for c in out.columns if c.startswith("location_")]
        assert len(loc_cols) > 0

    def test_consistent_columns_unseen_location(self, sample_df):
        """Unseen location at inference should not cause column mismatch."""
        enc = LocationEncoder().fit(sample_df)
        unseen = pd.DataFrame([{**sample_df.iloc[0].to_dict(), "location": "Kingston"}])
        out = enc.transform(unseen)
        assert set(out.columns) == set(enc.transform(sample_df).columns)

    def test_fit_transform_idempotent(self, sample_df):
        enc = LocationEncoder()
        out1 = enc.fit_transform(sample_df)
        out2 = enc.transform(sample_df)
        assert out1.equals(out2)


class TestInputValidator:
    def test_valid_input_passes(self, sample_df):
        v = InputValidator().fit(sample_df)
        out = v.transform(sample_df)
        assert out.shape == sample_df.shape

    def test_invalid_bedrooms_raises(self):
        df = pd.DataFrame([{
            "bedrooms": 99, "bathrooms": 1, "accommodates": 2, "location": "Negril",
            "has_wifi": 1, "has_pool": 0, "has_ac": 1, "has_kitchen": 1, "has_parking": 0,
        }])
        with pytest.raises(ValueError, match="bedrooms"):
            InputValidator().fit_transform(df)

    def test_invalid_location_raises(self):
        df = pd.DataFrame([{
            "bedrooms": 2, "bathrooms": 1, "accommodates": 4, "location": "Miami",
            "has_wifi": 1, "has_pool": 0, "has_ac": 1, "has_kitchen": 1, "has_parking": 0,
        }])
        with pytest.raises(ValueError, match="location"):
            InputValidator().fit_transform(df)

    def test_invalid_binary_raises(self):
        df = pd.DataFrame([{
            "bedrooms": 2, "bathrooms": 1, "accommodates": 4, "location": "Negril",
            "has_wifi": 5, "has_pool": 0, "has_ac": 1, "has_kitchen": 1, "has_parking": 0,
        }])
        with pytest.raises(ValueError, match="has_wifi"):
            InputValidator().fit_transform(df)
