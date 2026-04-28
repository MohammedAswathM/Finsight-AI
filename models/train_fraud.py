"""
Train fraud detection models using the Kaggle credit card fraud dataset.

This script trains both XGBoost and LightGBM classifiers, handles class
imbalance with SMOTE, evaluates both models, logs metrics to MLflow, and
saves the best model as a joblib artifact.

Expected dataset path:
    finsight-ai/data/creditcard.csv

The repository does not include the Kaggle dataset by default because it is
large and requires a Kaggle login. Place `creditcard.csv` in the `data/`
folder and rerun this script.
"""

from __future__ import annotations

from pathlib import Path
import multiprocessing
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = MODEL_DIR / "fraud_detector.joblib"


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Fraud dataset not found at {data_path}.\n"
            "Download the Kaggle credit card fraud dataset and place it at this path. "
            "See https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )
    return pd.read_csv(data_path)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
    }


def choose_best_model(results: dict) -> str:
    sorted_models = sorted(results.items(), key=lambda kv: kv[1]["auc_roc"], reverse=True)
    return sorted_models[0][0]


def train_models() -> tuple[dict, pd.DataFrame, pd.Series]:
    df = load_data(DATA_PATH)
    if "Class" not in df.columns:
        raise ValueError("Expected target column 'Class' in the fraud dataset.")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    models = {
        "xgboost": XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=min(8, multiprocessing.cpu_count()),
            random_state=42,
        ),
        "lightgbm": LGBMClassifier(
            n_jobs=min(8, multiprocessing.cpu_count()),
            random_state=42,
        ),
    }

    results: dict[str, dict] = {}
    trained_models: dict[str, object] = {}

    mlflow.set_experiment("fraud_detection")
    mlflow.set_tracking_uri(str(ROOT / "mlruns"))

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_resampled, y_resampled)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        trained_models[name] = model

        with mlflow.start_run(run_name=f"train_fraud_{name}"):
            mlflow.log_params({
                "model_name": name,
                "n_features": X.shape[1],
                "train_size": len(X_resampled),
                "test_size": len(X_test),
            })
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

        print(f"{name} metrics: {metrics}")

    best_name = choose_best_model(results)
    best_model = trained_models[best_name]
    joblib.dump(best_model, BEST_MODEL_PATH)

    print(f"Best model: {best_name}")
    print(f"Saved best model to {BEST_MODEL_PATH}")

    return results, X_test, y_test


def print_summary(results: dict[str, dict]) -> None:
    print("\nModel comparison results:")
    for name, metrics in results.items():
        print(f"\n{name.upper()}: ")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    results, _, _ = train_models()
    print_summary(results)
    print("\nTraining complete. The best model is saved as fraud_detector.joblib.")
