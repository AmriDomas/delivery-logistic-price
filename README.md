# Delivery Cost Prediction : End to End MLOps Project

## Overview

This project is a production style machine learning system that predicts delivery costs using real world logistics features.
It demonstrates the full lifecycle of an ML product from data processing to deployment, monitoring, and UI.
The goal is not just model accuracy, but a deployable, versioned, and observable ML system.

## Key Features

 - End to end ML pipeline (training → serving → monitoring)
 - Automated feature engineering
 - Model versioning with MLflow
 - REST API with FastAPI
 - Interactive prediction UI with Streamlit
 - Monitoring with Prometheus & Grafana
 - Fully containerized with Docker

## Tech Stack

| Layer      | Tools                           |
| ---------- | ------------------------------- |
| Modeling   | XGBoost, LightGBM, RandomForest |
| Tracking   | MLflow                          |
| API        | FastAPI                         |
| UI         | Streamlit                       |
| Monitoring | Prometheus, Grafana             |
| Storage    | MinIO (S3-compatible)           |
| Database   | PostgreSQL                      |
| Deployment | Docker, Docker Compose          |

## System Architecture

```bash
Training Pipeline
      ↓
   MLflow Registry
      ↓
     FastAPI
      ↓
    Streamlit UI
      ↓
Prometheus → Grafana
```
## Model Performance

Best Model: XGBoost
| Metric | Value   |
| ------ | ------- |
| R²     | ~0.9999 |
| RMSE   | ~3.27   |
| MAE    | ~2.56   |

## Example Prediction
```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "delivery_partner": "dhl",
  "package_type": "electronics",
  "vehicle_type": "bike",
  "delivery_mode": "express",
  "region": "west",
  "weather_condition": "clear",
  "delivery_status": "delivered",
  "delayed": "no",
  "distance_km": 50,
  "package_weight_kg": 5,
  "delivery_rating": 4
}'
```
Response:

```json
{
  "prediction": 123.45
}
```

## How to Run

1. Start the infrastructure
   ```bash
   docker compose -f infra/docker-compose.yml up -d
   ```
2. Set MLflow environment variables
   Required so training can log models and artifacts to MinIO.
   ```bash
   export MLFLOW_TRACKING_URI=http://localhost:5000
   export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
   export AWS_ACCESS_KEY_ID=minioadmin
   export AWS_SECRET_ACCESS_KEY=minioadmin
   export AWS_DEFAULT_REGION=us-east-1
   ```
3. Train the model
   ```bash
   python -m src.training.train_model
   ```
4. Launch API and UI
   ```bash
   docker compose -f infra/docker-compose.yml up -d --build
   ```
## Service Endpoints

   | Service      | URL                                            |
   | ------------ | ---------------------------------------------- |
   | Streamlit UI | [http://localhost:8501](http://localhost:8501) |
   | FastAPI      | [http://localhost:8000](http://localhost:8000) |
   | MLflow       | [http://localhost:5000](http://localhost:5000) |
   | Grafana      | [http://localhost:3000](http://localhost:3000) |
   | Prometheus   | [http://localhost:9090](http://localhost:9090) |

## Skills Demonstrated

   - Machine learning for tabular data
   - Feature engineering pipelines
   - MLflow model registry
   - REST API deployment
   - Monitoring and observability
   - Docker based MLOps architecture
## Author

Amri Domas Data Science • Machine Learning • AI Engineering • MLOps

If you’re reviewing this repo as part of a hiring process, feel free to reach out for further explanation or live walkthrough.
