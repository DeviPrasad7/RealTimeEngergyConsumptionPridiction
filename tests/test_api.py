import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "ok"
    assert json_response["model_loaded"] is True
    assert json_response["history_loaded"] is True

def test_predict(client):
    request_data = {
        "settlement_date": "2025-12-15",
        "settlement_period": 35
    }
    response = client.post("/predict", json=request_data)
    assert response.status_code == 200
    json_response = response.json()
    assert "prediction" in json_response
    assert isinstance(json_response["prediction"], float)

def test_predict_bulk(client):
    request_data = [
        {
            "settlement_date": "2025-12-15",
            "settlement_period": 35
        },
        {
            "settlement_date": "2025-12-17",
            "settlement_period": 10
        }
    ]
    response = client.post("/predict_bulk", json=request_data)
    assert response.status_code == 200
    json_response = response.json()
    assert isinstance(json_response, list)
    assert len(json_response) == 2
    for item in json_response:
        assert "prediction" in item
        assert isinstance(item["prediction"], float)

def test_predict_invalid_period(client):
    request_data = {
        "settlement_date": "2025-12-15",
        "settlement_period": 99
    }
    response = client.post("/predict", json=request_data)
    assert response.status_code == 422

def test_predict_invalid_date(client):
    request_data = {
        "settlement_date": "not-a-date",
        "settlement_period": 35
    }
    response = client.post("/predict", json=request_data)
    assert response.status_code == 422

def test_predict_bulk_empty_list(client):
    response = client.post("/predict_bulk", json=[])
    assert response.status_code == 400

def test_predict_bulk_invalid_item(client):
    request_data = [
        {
            "settlement_date": "2025-12-15",
            "settlement_period": 35
        },
        {
            "settlement_date": "not-a-date",
            "settlement_period": 10
        }
    ]
    response = client.post("/predict_bulk", json=request_data)
    assert response.status_code == 422
