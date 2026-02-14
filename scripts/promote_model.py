import mlflow

MODEL_NAME = "delivery_cost_model"

client = mlflow.tracking.MlflowClient()

versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])

if not versions:
    raise RuntimeError("No model in Staging")

v = versions[0]

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=v.version,
    stage="Production",
    archive_existing_versions=True
)

print(f"Model v{v.version} promoted to Production")
