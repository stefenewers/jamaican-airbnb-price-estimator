"""
Central configuration for the Jamaican Airbnb Price Predictor.
All paths, constants, and tuneable hyperparameters live here.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

DATA_PATH = DATA_DIR / "jamaican_airbnb_mock_dataset.csv"
MODEL_PATH = MODELS_DIR / "pipeline.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"

# Create dirs if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ── Feature Definitions ────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = ["location"]
BINARY_FEATURES = ["has_wifi", "has_pool", "has_ac", "has_kitchen", "has_parking"]
NUMERIC_FEATURES = ["bedrooms", "bathrooms", "accommodates"]
TARGET = "price"

LOCATIONS = ["Kingston", "Montego Bay", "Negril", "Ocho Rios", "Portland"]

# ── Training ───────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ── API ────────────────────────────────────────────────────────────────────────
import os
API_VERSION = "v1"
API_HOST = "0.0.0.0"
API_PORT = int(os.environ.get("PORT", 5001))
DEBUG = False

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
