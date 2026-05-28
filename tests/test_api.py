from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_home():

    response = client.get("/")

    assert response.status_code == 200

def test_prediction():

    sample = {
        "features": [1]*19
    }

    response = client.post(
        "/predict",
        json=sample
    )

    assert response.status_code == 200