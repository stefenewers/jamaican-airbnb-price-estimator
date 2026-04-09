# Jamaica Airbnb Price Predictor

A production-grade ML regression service that predicts nightly rental prices for Airbnb listings in Jamaica. Deployed as a Flask REST API with a multi-stage Docker build, structured logging, full preprocessing pipeline, and 5-fold cross-validated model selection.

---

## Architecture

```
Request → Flask API (POST /api/v1/predict)
            └─ Input validation (InputValidator)
            └─ Feature engineering (sklearn Pipeline)
                 ├─ AmenityScoreTransformer   — weighted composite score
                 ├─ InteractionFeatureTransformer — capacity_index, bath_ratio
                 ├─ LocationEncoder           — aligned OHE, no train/serve skew
                 └─ StandardScaler
            └─ Model inference
            └─ JSON response with latency metadata
```

The full preprocessing pipeline is serialized with the model artefact, ensuring that any transformation applied at training time is identically applied at inference — eliminating train/serve skew.

---

## API Reference

### `POST /api/v1/predict`

```json
{
  "bedrooms":     2,
  "bathrooms":    1,
  "accommodates": 4,
  "has_wifi":     1,
  "has_pool":     0,
  "has_ac":       1,
  "has_kitchen":  1,
  "has_parking":  0,
  "location":     "Negril"
}
```

Response:

```json
{
  "predicted_price_usd": 187.43,
  "currency": "USD",
  "unit": "per night",
  "model": "RandomForest",
  "latency_ms": 4.2,
  "inputs": { ... }
}
```

### `POST /api/v1/predict/batch`

Accepts `{ "listings": [...] }` — up to 100 listings per request.

### `GET /api/v1/health`

Liveness probe. Returns `{ "status": "healthy", "model_loaded": true }`.

### `GET /api/v1/model-info`

Returns full training metadata including CV results, test metrics, and feature importances.

---

## Quickstart

### Local development

```bash
git clone https://github.com/stefenewers/jamaican-airbnb-price-predictor.git
cd jamaican-airbnb-price-predictor

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train the model (runs CV comparison, saves pipeline to models/)
python -m ml.train

# Start the API
python run.py
```

The API is now available at `http://localhost:5000`. Open `frontend/index.html` in a browser to use the demo UI.

### Docker

```bash
# Build and run (model trains at build time)
docker-compose up --build

# Or manually
docker build -t airbnb-price-predictor:latest .
docker run -p 5000:5000 airbnb-price-predictor:latest
```

---

## ML Design

### Feature engineering

| Feature | Description |
|---|---|
| `amenity_score` | Weighted sum of binary amenity flags (pool=2×, A/C=1.5×, others=1×) |
| `capacity_index` | `accommodates / bedrooms` — space efficiency signal |
| `bath_ratio` | `bathrooms / bedrooms` — luxury indicator |
| `guest_density` | `accommodates × amenity_score` — demand × quality interaction |

### Model selection

Five candidate models are evaluated via 5-fold cross-validation on the training set. The winner is re-fit on the full training split and evaluated on a held-out 20% test set.

Candidates: `LinearRegression`, `Ridge(α=1)`, `Ridge(α=10)`, `RandomForest`, `XGBoost`

Selection criterion: mean CV R²

### Evaluation metrics

MAE, RMSE, R², and MAPE are computed on the held-out test set. All metrics and per-fold CV scores are written to `models/metadata.json` and exposed via `/api/v1/model-info`.

---

## Testing

```bash
# Run full test suite with coverage
make test

# Or directly
pytest tests/ -v --tb=short --cov=ml --cov=api
```

Tests cover: transformer unit tests, API endpoint contracts, error handling, batch validation, and plausibility bounds on predictions.

---

## Project Structure

```
├── api/
│   ├── app.py               # Flask app factory
│   └── routes/
│       ├── health.py        # GET /health, /model-info
│       └── predict.py       # POST /predict, /predict/batch
├── ml/
│   ├── features.py          # Custom sklearn transformers
│   ├── pipeline.py          # Pipeline builder
│   └── train.py             # Training + CV + evaluation
├── frontend/
│   └── index.html           # Demo UI
├── tests/
│   ├── conftest.py
│   ├── test_features.py
│   └── test_api.py
├── data/
│   └── jamaican_airbnb_mock_dataset.csv
├── models/                  # Generated — not committed
├── config.py                # Central configuration
├── run.py                   # Entry point
├── Dockerfile               # Multi-stage build
├── docker-compose.yml
├── Makefile
└── requirements.txt         # 9 direct dependencies
```

---

## Built by

**Stefen Ewers** — [stefenewers.com](https://www.stefenewers.com) · [LinkedIn](https://www.linkedin.com/in/stefen-ewers/) · [GitHub](https://github.com/stefenewers)
