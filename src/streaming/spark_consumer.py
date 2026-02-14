from pyspark.sql import SparkSession
import requests, json

spark = SparkSession.builder.appName("DeliveryStream").getOrCreate()

df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "delivery-stream") \
    .load()

def send_to_api(batch_df, batch_id):
    for row in batch_df.collect():
        payload = json.loads(row.value)
        requests.post("http://fastapi:8000/predict", json=payload)

df.writeStream.foreachBatch(send_to_api).start().awaitTermination()
