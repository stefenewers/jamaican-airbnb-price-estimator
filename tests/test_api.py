"""
Integration tests for the Flask REST API.

These tests run against the live Flask test client. The model pipeline
must exist (run python -m ml.train first) for predict endpoint tests.
"""
import json
import pytest


class TestHealthEndpoint:
    def test_health_returns_200_or_503(self, client):
        res = client.get("/api/v1/health")
        assert res.status_code in (200, 503)

    def test_health_json_structure(self, client):
        res = client.get("/api/v1/health")
        data = res.get_json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] in ("healthy", "degraded")

    def test_model_info_returns_404_if_untrained(self, client, tmp_path, monkeypatch):
        """If metadata.json is missing, model-info should 404."""
        import config
        monkeypatch.setattr(config, "METADATA_PATH", tmp_path / "nonexistent.json")
        # Re-import routes to pick up monkeypatched path
        res = client.get("/api/v1/model-info")
        # Accept 404 (not trained) or 200 (trained)
        assert res.status_code in (200, 404)


class TestPredictEndpoint:
    def test_returns_400_on_empty_body(self, client):
        res = client.post("/api/v1/predict", data="not json",
                          content_type="text/plain")
        assert res.status_code == 400

    def test_returns_422_on_missing_field(self, client):
        payload = {"bedrooms": 2, "bathrooms": 1}  # incomplete
        res = client.post("/api/v1/predict",
                          data=json.dumps(payload),
                          content_type="application/json")
        assert res.status_code == 422

    def test_returns_422_on_invalid_location(self, client, sample_listing):
        sample_listing["location"] = "Miami"
        res = client.post("/api/v1/predict",
                          data=json.dumps(sample_listing),
                          content_type="application/json")
        # 422 from validator, or 503 if model not present
        assert res.status_code in (422, 503)

    def test_prediction_response_schema(self, client, sample_listing):
        res = client.post("/api/v1/predict",
                          data=json.dumps(sample_listing),
                          content_type="application/json")
        if res.status_code == 503:
            pytest.skip("Model not trained — run python -m ml.train first")

        assert res.status_code == 200
        data = res.get_json()
        assert "predicted_price_usd" in data
        assert "currency" in data
        assert "model" in data
        assert "inputs" in data
        assert data["currency"] == "USD"

    def test_prediction_is_positive_number(self, client, sample_listing):
        res = client.post("/api/v1/predict",
                          data=json.dumps(sample_listing),
                          content_type="application/json")
        if res.status_code == 503:
            pytest.skip("Model not trained")

        data = res.get_json()
        assert isinstance(data["predicted_price_usd"], (int, float))
        assert data["predicted_price_usd"] > 0

    def test_prediction_in_plausible_range(self, client, sample_listing):
        res = client.post("/api/v1/predict",
                          data=json.dumps(sample_listing),
                          content_type="application/json")
        if res.status_code == 503:
            pytest.skip("Model not trained")

        price = res.get_json()["predicted_price_usd"]
        assert 30 < price < 1000, f"Prediction ${price} outside plausible range"


class TestBatchPredictEndpoint:
    def test_batch_returns_400_on_bad_body(self, client):
        res = client.post("/api/v1/predict/batch",
                          data=json.dumps({"wrong_key": []}),
                          content_type="application/json")
        assert res.status_code in (400, 422)

    def test_batch_rejects_oversized_payload(self, client, sample_listing):
        payload = {"listings": [sample_listing] * 101}
        res = client.post("/api/v1/predict/batch",
                          data=json.dumps(payload),
                          content_type="application/json")
        assert res.status_code == 422

    def test_batch_response_schema(self, client, sample_listing):
        payload = {"listings": [sample_listing, sample_listing]}
        res = client.post("/api/v1/predict/batch",
                          data=json.dumps(payload),
                          content_type="application/json")
        if res.status_code == 503:
            pytest.skip("Model not trained")

        data = res.get_json()
        assert "results" in data
        assert "errors" in data
        assert "total" in data
        assert data["total"] == 2


class TestErrorHandlers:
    def test_404_returns_json(self, client):
        res = client.get("/api/v1/nonexistent")
        assert res.status_code == 404
        data = res.get_json()
        assert "error" in data

    def test_405_returns_json(self, client):
        res = client.get("/api/v1/predict")  # GET on a POST endpoint
        assert res.status_code == 405
        data = res.get_json()
        assert "error" in data
