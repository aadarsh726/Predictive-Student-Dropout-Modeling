from fastapi.testclient import TestClient
from app.app import app
import pytest

client = TestClient(app)

# Explicitly load artifacts for testing since TestClient (without context manager) might miss startup
from app.app import load_artifacts
load_artifacts()

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert "Student Dropout Prediction" in response.text

def test_get_info():
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert "model_type" in data
    assert len(data["features"]) > 0

def test_predict_endpoint():
    # Fetch features to build a valid payload (even if dummy values)
    info_res = client.get("/info")
    features = info_res.json()["features"]
    
    payload = {}
    for f in features:
        if f["type"] == "categorical":
            # Use valid option
            payload[f["name"]] = f["options"][0]
        else:
            payload[f["name"]] = 10.0 # arbitrary number
            
    response = client.post("/predict", json={"features": payload})
    assert response.status_code == 200
    data = response.json()
    
    assert "prediction" in data
    assert "dropout_probability" in data
    assert "risk_level" in data
    assert "suggestions" in data
    assert 0 <= data["dropout_probability"] <= 1

def test_predict_missing_fields():
    # Should handle gracefully (we implemented default logic) or fail validation if strict
    # Our implementation handles missing fields by defaults/median, so it should succeed.
    response = client.post("/predict", json={"features": {}})
    assert response.status_code == 200
    # Optional: if we wanted strict validation, we'd assert 422
