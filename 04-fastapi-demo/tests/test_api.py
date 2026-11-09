
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from main import create_app


class FakePredictor:

    def predict(self, text):

        return {
            "category": "Billing",
            "priority": "High",
            "confidence": 0.93
        }


def fake_model_loader():

    return FakePredictor()


app = create_app(model_loader=fake_model_loader)


@pytest.fixture
def client():

    # Context manager triggers the application lifespan.
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_valid_ticket(client):

    response = client.post(
        "/predict",
        json={"text": "I was charged twice"}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["category"] == "Billing"
    assert data["priority"] == "High"
    assert data["confidence"] == 0.93


def test_predict_empty_text(client):

    response = client.post(
        "/predict",
        json={"text": ""}
    )

    assert response.status_code == 422


def test_predict_whitespace_text(client):

    response = client.post(
        "/predict",
        json={"text": "     "}
    )

    assert response.status_code == 422


def test_predict_missing_text(client):

    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422
