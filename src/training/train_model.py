import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.features.feature_engineering import AdvancedFeatureEngineering
from mlflow.models.signature import infer_signature
import shap
import os
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
# ===============================
# CONFIG
# ===============================
TARGET_COL = "delivery_cost"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV = 5

mlflow.set_experiment("delivery-logistics-cost-prediction")


class AdvancedPredictiveModeling:
    def __init__(self, df, target_col=TARGET_COL):
        self.df = df.copy()
        self.target_col = target_col
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None

    # ---------------------------
    # DATA PREPARATION
    # ---------------------------
    def prepare_modeling_data(self):
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column {self.target_col} not found")

        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        if "cluster" in X.columns:
            X = X.drop(columns=["cluster"])

        self.numerical_features = [
            "distance_km",
            "package_weight_kg",
            "delivery_rating"
        ]

        self.categorical_features = [
            "delivery_partner",
            "package_type",
            "vehicle_type",
            "delivery_mode",
            "region",
            "weather_condition",
            "delivery_status",
            "delayed"
        ]

        self.feature_names = self.numerical_features + self.categorical_features

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    # ---------------------------
    # PREPROCESSOR
    # ---------------------------
    def create_preprocessor(self):
        num_pipe = Pipeline([
            ("scaler", StandardScaler())
        ])

        cat_pipe = Pipeline([
            ("onehot", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ])

        return ColumnTransformer([
            ("num", num_pipe, self.numerical_features),
            ("cat", cat_pipe, self.categorical_features)
        ])

    # ---------------------------
    # MODEL DEFINITIONS
    # ---------------------------
    def initialize_models(self):
        return {
            "RandomForest": {
                "model": RandomForestRegressor(random_state=RANDOM_STATE),
                "params": {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [10, 20, None],
                    "model__min_samples_split": [2, 5],
                    "model__min_samples_leaf": [1, 2],
                }
            },
            "XGBoost": {
                "model": XGBRegressor(
                    random_state=RANDOM_STATE,
                    verbosity=0,
                    objective="reg:squarederror"
                ),
                "params": {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [3, 6, 9],
                    "model__learning_rate": [0.01, 0.1],
                    "model__subsample": [0.8, 1.0],
                }
            },
            "LightGBM": {
                "model": LGBMRegressor(
                    random_state=RANDOM_STATE,
                    verbose=-1
                ),
                "params": {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [5, 10, -1],
                    "model__learning_rate": [0.01, 0.1],
                    "model__num_leaves": [31, 50],
                }
            }
        }

    # ---------------------------
    # TRAINING
    # ---------------------------
    def train_models(self):
        preprocessor = self.create_preprocessor()
        models_cfg = self.initialize_models()

        best_cv_score = float("-inf")

        for name, cfg in models_cfg.items():
            print(f"\n\U0001f680 Training {name}")

            pipeline = Pipeline([
                ("feature_engineering", AdvancedFeatureEngineering()),
                ("preprocessor", preprocessor),
                ("model", cfg["model"])
            ])


            grid = GridSearchCV(
                pipeline,
                cfg["params"],
                cv=CV,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                verbose=1
            )

            grid.fit(self.X_train, self.y_train)

            y_pred = grid.best_estimator_.predict(self.X_test)

            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)

            self.results[name] = {
                "best_params": grid.best_params_,
                "cv_score": grid.best_score_,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "predictions": y_pred
            }

            if grid.best_score_ > best_cv_score:
                best_cv_score = grid.best_score_
                self.best_model = grid.best_estimator_   # <- pipeline lengkap
                self.best_model_name = name

        print(f"\n🏆 Best Model: {self.best_model_name}")

        os.makedirs("artifacts", exist_ok=True)

        numeric_cols = self.X_train.select_dtypes(include=[np.number])

        baseline_stats = {
            "feature_means": numeric_cols.mean().to_dict(),
            "feature_stds": numeric_cols.std().to_dict()
        }

        joblib.dump(
            baseline_stats,
            "artifacts/baseline_stats.pkl"
        )

        if self.best_model is None:
            raise RuntimeError("Best model not found. Training may have failed.")

    def log_shap_summary(self):
        X_sample = self.X_train.sample(200, random_state=42)

        model = self.best_model
        explainer = shap.Explainer(model.predict, X_sample)
        shap_values = explainer(X_sample)

        shap_summary = {
            "base_value": float(shap_values.base_values.mean()),
            "values": shap_values.values.tolist(),
            "feature_names": X_sample.columns.tolist()
        }

        os.makedirs("artifacts/explainability", exist_ok=True)
        joblib.dump(
            shap_summary,
            "artifacts/explainability/shap_summary.pkl"
        )

        mlflow.log_artifact(
            "artifacts/explainability/shap_summary.pkl",
            artifact_path="explainability"
        )


    # ---------------------------
    # LOGGING & SAVE
    # ---------------------------
    def log_and_save(self):
        best_res = self.results[self.best_model_name]

        # ======================
        # PARAMS & METRICS
        # ======================
        mlflow.log_param("best_model", self.best_model_name)
        mlflow.log_params(best_res["best_params"])

        mlflow.log_metric("r2", best_res["r2"])
        mlflow.log_metric("rmse", best_res["rmse"])
        mlflow.log_metric("mae", best_res["mae"])

        # ======================
        # MODEL
        # ======================
        # ambil sample raw input
        X_sample_raw = self.X_train.sample(5, random_state=42)

        # ambil feature engineering dari pipeline
        fe = self.best_model.named_steps["feature_engineering"]

        # transform ke feature space yang benar
        X_sample_fe = fe.transform(X_sample_raw)

        # prediksi tetap pakai pipeline penuh
        y_sample = self.best_model.predict(X_sample_raw)

        # signature berdasarkan fitur setelah engineering
        signature = infer_signature(X_sample_fe, y_sample)

        mlflow.sklearn.log_model(
            sk_model=self.best_model,
            artifact_path="model",
            signature=signature,
            input_example=X_sample_fe
        )

        client = MlflowClient()

        result = mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            "delivery_cost_model"
        )

        client.transition_model_version_stage(
            name="delivery_cost_model",
            version=result.version,
            stage="Production",
            archive_existing_versions=True
        )


        # ======================
        # BASELINE (DRIFT)
        # ======================
        mlflow.log_artifact(
            "artifacts/baseline_stats.pkl",
            artifact_path="baseline"
        )

        # ======================
        # FEATURE IMPORTANCE
        # ======================
        try:
            model = self.best_model.named_steps["model"]
            preprocessor = self.best_model.named_steps["preprocessor"]

            feature_names = preprocessor.get_feature_names_out()

            # ambil importance dari model apapun
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = model.coef_
            else:
                raise ValueError("Model has no feature importance attribute")

            fi = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            }).sort_values("importance", ascending=False)

            os.makedirs("artifacts/explainability", exist_ok=True)
            fi_path = "artifacts/explainability/feature_importance.csv"
            fi.to_csv(fi_path, index=False)

            mlflow.log_artifact(
                fi_path,
                artifact_path="explainability"
            )

            print("Feature importance logged.")

        except Exception as e:
            print(f"[WARN] Feature importance skipped: {e}")

        # ======================
        # GLOBAL SHAP (OFFLINE)
        # ======================
        try:
            import shap

            X_sample = self.X_train.sample(
                min(300, len(self.X_train)),
                random_state=42
            )

            # ambil pipeline parts
            fe = self.best_model.named_steps["feature_engineering"]
            preprocessor = self.best_model.named_steps["preprocessor"]
            model = self.best_model.named_steps["model"]

            # transform ke numeric space
            X_fe = fe.transform(X_sample)
            X_transformed = preprocessor.transform(X_fe)

            feature_names = preprocessor.get_feature_names_out()

            explainer = shap.Explainer(model, X_transformed)
            shap_values = explainer(X_transformed)

            shap_summary = {
                "mean_abs_shap": (
                    pd.DataFrame(
                        np.abs(shap_values.values),
                        columns=feature_names
                    )
                    .mean()
                    .sort_values(ascending=False)
                    .to_dict()
                ),
                "base_value": float(np.mean(shap_values.base_values)),
                "feature_names": feature_names.tolist()
            }

            os.makedirs("artifacts/explainability", exist_ok=True)
            shap_path = "artifacts/explainability/shap_summary.pkl"

            joblib.dump(shap_summary, shap_path)

            mlflow.log_artifact(
                shap_path,
                artifact_path="explainability"
            )

            print("SHAP summary logged.")

        except Exception as e:
            print(f"[WARN] SHAP skipped: {e}")


    # ---------------------------
    # EXECUTION
    # ---------------------------
    def run(self):
        self.prepare_modeling_data()
        self.train_models()
        self.log_and_save()


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    print("📦 Loading data")
    df = pd.read_csv("data/raw/Delivery_Logistics.csv")

    with mlflow.start_run():
        pipeline = AdvancedPredictiveModeling(df)
        pipeline.run()

    print("✅ Training completed & logged to MLflow")
