from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import joblib
import os
from collections import deque

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.monitoring.drift_check import detect_drift
from src.monitoring.trigger_retrain import trigger_retrain
from src.features.feature_engineering import AdvancedFeatureEngineering


# ======================
# APP
# ======================
app = FastAPI(title="Delivery Cost Prediction API")

# ======================
# CONFIG
# ======================
MODEL_NAME = "delivery_cost_model"
MODEL_STAGE = "Production"

REQUEST_BUFFER = deque(maxlen=500)
BASELINE = None
MODEL = None

client = MlflowClient()


# ======================
# PROMETHEUS METRICS
# ======================
REQ_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

REQ_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency"
)


# ======================
# LOAD MODEL (FIXED)
# ======================
MODEL_NAME = "delivery_cost_model"
MODEL_STAGE = "Production"

def load_model():
    client = MlflowClient()

    versions = client.get_latest_versions(
        MODEL_NAME,
        stages=[MODEL_STAGE]
    )

    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)

    return model


# ======================
# LOAD BASELINE (DRIFT)
# ======================
def load_baseline():
    global BASELINE
    if BASELINE is not None:
        return BASELINE

    versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    if not versions:
        return None

    run_id = versions[0].run_id
    path = client.download_artifacts(
        run_id,
        "baseline/baseline_stats.pkl"
    )

    BASELINE = joblib.load(path)
    return BASELINE


# ======================
# INPUT SCHEMA
# ======================
class PredictionRequest(BaseModel):
    delivery_partner: str
    package_type: str
    vehicle_type: str
    delivery_mode: str
    region: str
    weather_condition: str
    delivery_status: str
    delayed: str
    distance_km: float
    package_weight_kg: float
    delivery_rating: int

# ======================
# ENDPOINTS
# ======================
@app.post("/predict")
def predict(payload: PredictionRequest):
    try:
        model = load_model()

        # DEBUG signature model
        schema = model.metadata.get_input_schema()
        print("MODEL SIGNATURE:", [col.name for col in schema.inputs])

        # raw input
        df_raw = pd.DataFrame([payload.dict()])

        # feature engineering
        df_fe = AdvancedFeatureEngineering().transform(df_raw)

        # align dengan signature
        expected_cols = [col.name for col in schema.inputs]

        for col in expected_cols:
            if col not in df_fe.columns:
                df_fe[col] = 0

        df_fe = df_fe[expected_cols]

        pred = model.predict(df_fe)[0]

        return {"prediction": float(pred)}

    except Exception as e:
        return {
            "error": "prediction_failed",
            "message": str(e)
        }

@app.get("/debug_schema")
def debug_schema():
    model = load_model()
    schema = model.metadata.get_input_schema()
    return [col.name for col in schema.inputs]


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model_info")
def model_info():
    versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    if not versions:
        return {"error": "No production model found"}

    v = versions[0]
    run = client.get_run(v.run_id)

    return {
        "model_name": MODEL_NAME,
        "stage": MODEL_STAGE,
        "version": v.version,
        "run_id": v.run_id,
        "metrics": run.data.metrics,
        "params": run.data.params
    }


@app.get("/drift")
def drift_check(threshold: float = 0.15):
    baseline = load_baseline()

    if not baseline:
        return {
            "status": "baseline_not_available",
            "message": "Drift detection disabled"
        }

    result = detect_drift(
        baseline=baseline,
        request_buffer=list(REQUEST_BUFFER),
        threshold=threshold
    )

    if result.get("drift_detected"):
        trigger_retrain()

    return result
