import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AdvancedFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Stateless, sklearn-compatible feature engineering.
    - NO encoding
    - NO scaling
    - NO fitting
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Stateless transformer
        return self

    def transform(self, X):
        # =========================
        # SAFETY COPY
        # =========================
        df = X.copy()

        # =========================
        # DROP UNUSED / LEAKY COLS
        # =========================
        unused = [
            "delivery_id",
            "delivery_time_hours",
            "expected_time_hours"
        ]
        df = df.drop(columns=unused, errors="ignore")

        # =========================
        # INTERACTION FEATURES
        # =========================
        if {"distance_km", "package_weight_kg"}.issubset(df.columns):
            df["distance_weight_interaction"] = (
                df["distance_km"] * df["package_weight_kg"]
            )

        # =========================
        # POLYNOMIAL FEATURES
        # =========================
        for col in ["distance_km", "package_weight_kg", "delivery_rating"]:
            if col in df.columns:
                df[f"{col}_power_2"] = df[col] ** 2

        # =========================
        # FINAL CHECK
        # =========================
        if df.isnull().any().any():
            raise ValueError("NaN detected after feature engineering")

        return df
