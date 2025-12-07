from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"


def test_predict_endpoint():
    payload = {
        "age": 25,
        "hours_per_week": 5,
        "num_logins_last_month": 10,
        "assignments_submitted": 3,
        "discussion_posts": 2,
        "num_siblings": 1,
        "continent": "Asia",
        "education_level": "Bachelors",
        "preferred_device": "Mobile",
        "has_pet": 1,
        "is_working_professional": 0,
        "videos_watched_pct": 80.0
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], int)
    if "probability" in data:
        assert 0.0 <= data["probability"] <= 1.0

