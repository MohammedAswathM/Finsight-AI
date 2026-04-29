"""
Fraud detector inference wrapper.

Loads the best trained fraud model and exposes a single predict_fraud()
function for use by downstream agents. If the all-model bundle exists, the
saved optimal threshold for the best model is used at inference time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATHS = [
    ROOT / "models" / "fraud_detector.joblib",
    ROOT / "models" / "fraud_detector.pkl",
]
ALL_MODELS_PATH = ROOT / "models" / "fraud_models.joblib"
DEFAULT_THRESHOLD = 0.5


def _load_model():
    if ALL_MODELS_PATH.exists():
        bundle = joblib.load(ALL_MODELS_PATH)
        best_name = bundle["best_model_name"]
        model = bundle["models"][best_name]
        threshold = bundle.get("optimal_thresholds", {}).get(best_name, DEFAULT_THRESHOLD)
        return model, float(threshold), best_name

    for path in MODEL_PATHS:
        if path.exists():
            return joblib.load(path), DEFAULT_THRESHOLD, path.stem
    raise FileNotFoundError(
        "No trained fraud model found. "
        "Run `python models/train_fraud.py` after placing creditcard.csv in finsight-ai/data/."
    )


def _build_feature_vector(transaction_features: Dict[str, float]) -> np.ndarray:
    feature_order = [
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount",
    ]

    missing = [f for f in feature_order if f not in transaction_features]
    if missing:
        raise ValueError(
            "Missing features for fraud prediction: "
            f"{missing}. Expected the Kaggle credit card fraud feature set."
        )

    values = [float(transaction_features[name]) for name in feature_order]
    return np.array(values).reshape(1, -1)


def _format_risk_level(probability: float) -> str:
    if probability >= 0.9:
        return "CRITICAL"
    if probability >= 0.75:
        return "HIGH"
    if probability >= 0.4:
        return "MEDIUM"
    return "LOW"


def predict_fraud(transaction_features: Dict[str, float]) -> Dict[str, object]:
    """
    Predict whether a transaction is fraudulent.

    Args:
        transaction_features: dict with the Kaggle credit card fraud features:
            Time, V1..V28, Amount

    Returns:
        dict with fraud_probability, is_fraud, risk_level, confidence, features, and message.
    """
    model, threshold, model_name = _load_model()
    X = _build_feature_vector(transaction_features)
    proba = model.predict_proba(X)[0, 1]
    pred = bool(proba >= threshold)
    risk_level = _format_risk_level(proba)
    confidence = float(proba if pred else 1 - proba)

    return {
        "fraud_probability": round(float(proba), 4),
        "is_fraud": pred,
        "risk_level": risk_level,
        "confidence": round(confidence, 4),
        "model_name": model_name,
        "threshold": round(threshold, 4),
        "features": list(transaction_features.keys()),
        "message": (
            "Fraud model prediction completed. "
            "Supply credit card transaction features in the Kaggle dataset format."
        ),
    }


if __name__ == "__main__":
    sample = {
        "Time": 10000,
        **{f"V{i}": 0.0 for i in range(1, 29)},
        "Amount": 50.0,
    }

    try:
        print(predict_fraud(sample))
    except Exception as exc:
        print("Error:", exc)
        print("Make sure the model artifact exists and the feature keys are correct.")
