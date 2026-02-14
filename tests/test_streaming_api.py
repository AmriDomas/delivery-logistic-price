import requests
import time

def test_prediction_triggers_monitoring():
    payload = {
        "distance_km": 20,
        "package_weight_kg": 3,
        "delivery_rating": 4
    }

    for _ in range(5):
        requests.post("http://localhost:8000/predict", json=payload)
        time.sleep(0.5)

    metrics = requests.get("http://localhost:8000/metrics").text
    assert "prediction_requests_total" in metrics
