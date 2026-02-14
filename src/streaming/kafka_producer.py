from kafka import KafkaProducer
import json
import time

producer = KafkaProducer(
    bootstrap_servers="redpanda:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

while True:
    event = {
        "distance_km": 120,
        "package_weight_kg": 15,
        "delivery_rating": 4
    }
    producer.send("delivery_events", event)
    time.sleep(2)
