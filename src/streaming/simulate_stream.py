import requests
import random
import time

API_URL = "http://localhost:8000/predict"

partners = [
    "xpressbees", "fedex", "dhl", "ekart", "blue dart",
    "delhivery", "shadowfax", "ecom express", "amazon logistics"
]

for i in range(10_000):
    payload = {
        "delivery_partner": random.choice(partners),
        "package_type": random.choice([
            "electronics", "pharmacy", "groceries", "documents"
        ]),
        "vehicle_type": random.choice(["bike", "van", "truck"]),
        "delivery_mode": random.choice(["same day", "express", "standard"]),
        "region": random.choice(["north", "south", "west"]),
        "weather_condition": random.choice(["clear", "rainy", "stormy"]),
        "delivery_status": random.choice(["delivered", "delayed"]),
        "distance_km": random.uniform(5, 250),
        "package_weight_kg": random.uniform(1, 30),
        "delayed": random.choice(["yes", "no"]),
        "delivery_rating": random.randint(1, 5)
    }

    requests.post(API_URL, json=payload)
    time.sleep(0.1)
