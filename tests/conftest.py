"""
Pytest configuration and shared fixtures.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_listing() -> dict:
    """A valid, fully-specified listing payload."""
    return {
        "bedrooms": 2,
        "bathrooms": 1,
        "accommodates": 4,
        "has_wifi": 1,
        "has_pool": 0,
        "has_ac": 1,
        "has_kitchen": 1,
        "has_parking": 0,
        "location": "Negril",
    }


@pytest.fixture
def sample_df(sample_listing) -> pd.DataFrame:
    return pd.DataFrame([sample_listing])


@pytest.fixture
def flask_app():
    """Creates a test Flask app with testing config."""
    from api.app import create_app
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(flask_app):
    """A test client for the Flask app."""
    with flask_app.test_client() as c:
        yield c
