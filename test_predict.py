import requests

url = "http://localhost:8000/predict"

payload = {
    "delivery_partner": "dhl",
    "package_type": "documents",
    "vehicle_type": "bike",
    "delivery_mode": "standard",
    "region": "west",
    "weather_condition": "clear",
    "delivery_status": "delivered",
    "distance_km": 10,
    "package_weight_kg": 2,
    "delayed": "no",
    "delivery_rating": 4
}

response = requests.post(url, json=payload)

print("Status code:", response.status_code)
print("Response:")
print(response.json())
