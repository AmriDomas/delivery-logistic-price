import streamlit as st
import requests
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import joblib
import os

# =========================
# CONFIG
# =========================
API_URL = "http://fastapi:8000"
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_NAME = "delivery_cost_model"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# =========================
# HELPERS
# =========================
def get_latest_run_id():
    versions = client.get_latest_versions(
        MODEL_NAME,
        stages=["Production"]
    )
    if not versions:
        return None

    run_id = versions[0].run_id
    st.write("DEBUG run_id:", run_id)
    return run_id


# =========================
# UI SETUP
# =========================
st.set_page_config(layout="wide")
st.title("🚚 Delivery Cost MLOps Dashboard")

tabs = st.tabs([
    "🔮 Predict",
    "📊 Model Metrics",
    "🌳 Feature Importance",
    "🧠 SHAP Explanation"
])

# =========================
# INPUTS
# =========================
with st.sidebar:
    delivery_partner = st.selectbox(
        "Delivery Partner",
        [
            "xpressbees", "fedex", "dhl", "ekart", "blue dart",
            "delhivery", "shadowfax", "ecom express", "amazon logistics"
        ]
    )

    package_type = st.selectbox(
        "Package Type",
        [
            "fragile items", "pharmacy", "documents", "automobile parts",
            "electronics", "clothing", "furniture", "cosmetics", "groceries"
        ]
    )

    vehicle_type = st.selectbox(
        "Vehicle Type",
        ["ev bike", "van", "scooter", "bike", "truck", "ev van"]
    )

    delivery_mode = st.selectbox(
        "Delivery Mode",
        ["two day", "same day", "express", "standard"]
    )

    region = st.selectbox(
        "Region",
        ["west", "central", "south", "north", "east"]
    )

    weather_condition = st.selectbox(
        "Weather Condition",
        ["foggy", "stormy", "rainy", "cold", "hot", "clear"]
    )

    delivery_status = st.selectbox(
        "Delivery Status",
        ["delivered", "delayed", "failed"]
    )

    delayed = st.selectbox("Delayed", ["yes", "no"])

    distance = st.slider("Distance (km)", 1.0, 300.0, 50.0)
    weight = st.slider("Package Weight (kg)", 1.0, 50.0, 5.0)
    rating = st.slider("Delivery Rating", 1, 5, 4)

payload = {
    "delivery_partner": delivery_partner,
    "package_type": package_type,
    "vehicle_type": vehicle_type,
    "delivery_mode": delivery_mode,
    "region": region,
    "weather_condition": weather_condition,
    "delivery_status": delivery_status,
    "distance_km": distance,
    "package_weight_kg": weight,
    "delayed": delayed,
    "delivery_rating": rating
}

# =========================
# TAB 1 — PREDICT
# =========================
with tabs[0]:
    st.subheader("Prediction")

    if st.button("Predict"):
        resp = requests.post(f"{API_URL}/predict", json=payload)

        if resp.status_code != 200:
            st.error(f"API Error ({resp.status_code})")
            st.code(resp.text)
        else:
            try:
                res = resp.json()
                if "prediction" in res:
                    st.metric(
                        "Estimated Delivery Cost",
                        f"{res['prediction']:.2f}"
                    )
                else:
                    st.error("Prediction key missing in response")
                    st.json(res)
            except Exception as e:
                st.error("Invalid JSON response from API")
                st.code(resp.text)

# =========================
# TAB 2 — MODEL METRICS
# =========================
with tabs[1]:
    st.subheader("Model Information")
    info = requests.get(f"{API_URL}/model_info").json()
    st.json(info)

# =========================
# TAB 3 — FEATURE IMPORTANCE
# =========================
with tabs[2]:
    st.subheader("🌳 Feature Importance")

    run_id = get_latest_run_id()
    if not run_id:
        st.warning("No production model found")
    else:
        try:
            fi_path = client.download_artifacts(
                run_id,
                "explainability/feature_importance.csv"
            )
            fi = pd.read_csv(fi_path)

            st.bar_chart(
                fi.set_index("feature")["importance"]
            )
        except Exception as e:
            st.error(f"Feature importance not available: {e}")

# =========================
# TAB 4 — SHAP
# =========================
with tabs[3]:
    st.subheader("🧠 Global SHAP Explanation")

    run_id = get_latest_run_id()
    if not run_id:
        st.warning("No production model found")
    else:
        try:
            shap_path = client.download_artifacts(
                run_id,
                "explainability/shap_summary.pkl"
            )

            shap_data = joblib.load(shap_path)

            st.write("Base value:", shap_data["base_value"])

            shap_df = pd.Series(
                shap_data["mean_abs_shap"]
            ).sort_values(ascending=False)

            st.bar_chart(shap_df)

        except Exception as e:
            st.error(f"SHAP data not available: {e}")
