from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess
import requests

def check_drift():
    r = requests.get("http://fastapi:8000/drift").json()
    if not r.get("drift_detected"):
        raise ValueError("No drift detected")

def retrain():
    subprocess.run(
        ["python", "src/training/train_model.py"],
        check=True
    )

def promote():
    subprocess.run(
        ["python", "scripts/promote_model.py"],
        check=True
    )

with DAG(
    dag_id="delivery_cost_auto_retrain",
    start_date=datetime(2026, 1, 1),
    schedule_interval="0 1 * * 1",
    catchup=False,
) as dag:

    drift_check = PythonOperator(
        task_id="check_drift",
        python_callable=check_drift
    )

    retrain_model = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain
    )

    promote_model = PythonOperator(
        task_id="promote_model",
        python_callable=promote
    )

    drift_check >> retrain_model >> promote_model
