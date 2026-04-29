"""
Train fraud detection models using the Kaggle credit card fraud dataset.

Models trained (in order):
1. Logistic Regression (linear baseline)
2. Random Forest (tree baseline)
3. XGBoost (primary model) — requires: pip install xgboost
4. LightGBM (comparison model) — requires: pip install lightgbm

All models:
- Handle class imbalance with SMOTE oversampling
- Tuned threshold via precision-recall curve for optimal F1
- Report AUC-ROC, AUC-PR, F1 (at optimal threshold), precision, recall
- Include 2×2 confusion matrix
- Feature importance via SHAP (tree models only)
- Log all metrics to MLflow
- Save all final models plus best model (by AUC-PR) as fraud_detector.joblib

Dataset Setup:
    Download the dataset and place it at: data/creditcard.csv
    Command: curl -L -o data/creditcard.csv "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

Expected dataset path:
    finsight-ai/data/creditcard.csv

The repository does not include the Kaggle dataset by default because it is
large (144MB) and requires a Kaggle login. Download it using the command above
and rerun this script.
"""

from __future__ import annotations

from pathlib import Path
import multiprocessing
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    auc,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# Try importing XGBoost and LightGBM with fallback
xgboost_available = False
lightgbm_available = False
xgboost_import_error = None
lightgbm_import_error = None

try:
    from xgboost import XGBClassifier
    xgboost_available = True
except Exception as exc:
    xgboost_import_error = exc
    print(f"⚠️  XGBoost not available: {exc}")

try:
    from lightgbm import LGBMClassifier
    lightgbm_available = True
except Exception as exc:
    lightgbm_import_error = exc
    print(f"⚠️  LightGBM not available: {exc}")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = MODEL_DIR / "fraud_detector.joblib"
ALL_MODELS_PATH = MODEL_DIR / "fraud_models.joblib"
MODEL_ARTIFACT_PATHS = {
    "logistic_regression": MODEL_DIR / "fraud_model_logistic_regression.joblib",
    "random_forest": MODEL_DIR / "fraud_model_random_forest.joblib",
    "xgboost": MODEL_DIR / "fraud_model_xgboost.joblib",
    "lightgbm": MODEL_DIR / "fraud_model_lightgbm.joblib",
}
REPORT_DIR = MODEL_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Fraud dataset not found at {data_path}.\n"
            "Download the Kaggle credit card fraud dataset and place it at this path. "
            "See https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )
    return pd.read_csv(data_path)


def evaluate_model_full(
    model, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """
    Comprehensive model evaluation with threshold tuning.
    
    Returns metrics at optimal threshold (F1 maximization).
    """
    y_pred_default = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # AUC-ROC
    auc_roc = float(roc_auc_score(y_test, y_proba))

    # Precision-Recall curve and AUC-PR
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_proba)
    auc_pr = float(auc(recall_vals, precision_vals))

    # Find optimal threshold (maximize F1)
    f1_scores = 2 * (precision_vals[:-1] * recall_vals[:-1]) / (
        precision_vals[:-1] + recall_vals[:-1] + 1e-10
    )
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Predictions at optimal threshold
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)

    # Metrics at optimal threshold
    f1_optimal = float(f1_score(y_test, y_pred_optimal))
    precision_optimal = float(precision_score(y_test, y_pred_optimal))
    recall_optimal = float(recall_score(y_test, y_pred_optimal))

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_optimal).ravel()

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "f1_optimal": f1_optimal,
        "precision_optimal": precision_optimal,
        "recall_optimal": recall_optimal,
        "optimal_threshold": float(optimal_threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "y_proba": y_proba,  # For SHAP
        "y_pred": y_pred_optimal,
        "y_test": y_test.values,
    }


def plot_feature_importance_shap(
    model, X_test: pd.DataFrame, model_name: str
) -> Path:
    """Generate SHAP feature importance plot for tree-based models."""
    try:
        import shap

        # For tree-based models, use TreeExplainer
        if hasattr(model, "get_booster") or hasattr(model, "booster_"):
            # XGBoost or LightGBM
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, "estimators_"):
            # Random Forest
            explainer = shap.TreeExplainer(model)
        else:
            print(f"SHAP not available for {model_name} (not tree-based)")
            return None

        shap_values = explainer.shap_values(X_test)
        
        # Handle binary classification output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use fraud class

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plot_path = REPORT_DIR / f"shap_importance_{model_name}.png"
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Saved SHAP plot to {plot_path}")
        return plot_path
    except ImportError:
        print(f"  SHAP not available for {model_name} (install shap package)")
        return None


def validate_performance_thresholds(
    name: str,
    metrics: dict,
    min_f1: float = 0.8,
    min_auc_pr: float = 0.8,
    min_precision: float = 0.8,
    min_recall: float = 0.8,
) -> bool:
    """Verify that key metrics meet the target thresholds."""
    passed = True
    notes = []

    if metrics["f1_optimal"] < min_f1:
        notes.append(f"F1 {metrics['f1_optimal']:.3f} < {min_f1}")
        passed = False
    if metrics["auc_pr"] < min_auc_pr:
        notes.append(f"AUC-PR {metrics['auc_pr']:.3f} < {min_auc_pr}")
        passed = False
    if metrics["precision_optimal"] < min_precision:
        notes.append(f"Precision {metrics['precision_optimal']:.3f} < {min_precision}")
        passed = False
    if metrics["recall_optimal"] < min_recall:
        notes.append(f"Recall {metrics['recall_optimal']:.3f} < {min_recall}")
        passed = False

    if passed:
        print(f"  ✅ {name} met target performance thresholds")
    else:
        print(f"  ⚠️  {name} did NOT meet target thresholds:")
        for note in notes:
            print(f"     - {note}")

    return passed


def choose_best_model(results: dict) -> str:
    """Select best model by AUC-PR (more honest for imbalanced data)."""
    sorted_models = sorted(
        results.items(), key=lambda kv: kv[1]["auc_pr"], reverse=True
    )
    return sorted_models[0][0]


def save_final_models(
    trained_models: dict[str, object], results: dict[str, dict], best_name: str
) -> None:
    """Persist every trained final model and a bundle with thresholds/metrics."""
    for name, model in trained_models.items():
        model_path = MODEL_ARTIFACT_PATHS.get(name, MODEL_DIR / f"fraud_model_{name}.joblib")
        joblib.dump(model, model_path)
        print(f"   Saved {name} to {model_path}")

    bundle = {
        "best_model_name": best_name,
        "models": trained_models,
        "metrics": results,
        "optimal_thresholds": {
            name: metrics["optimal_threshold"] for name, metrics in results.items()
        },
    }
    joblib.dump(bundle, ALL_MODELS_PATH)
    print(f"   Saved all-model bundle to {ALL_MODELS_PATH}")


def train_models() -> tuple[dict, pd.DataFrame, pd.Series]:
    missing_models = []
    if not xgboost_available:
        missing_models.append(f"XGBoost ({xgboost_import_error})")
    if not lightgbm_available:
        missing_models.append(f"LightGBM ({lightgbm_import_error})")
    if missing_models:
        raise RuntimeError(
            "Fraud training requires all four final models. Missing: "
            + "; ".join(missing_models)
        )

    df = load_data(DATA_PATH)
    if "Class" not in df.columns:
        raise ValueError("Expected target column 'Class' in the fraud dataset.")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Build model lineup
    models = {}

    # 1. Logistic Regression (linear baseline)
    models["logistic_regression"] = LogisticRegression(
        random_state=42, max_iter=1000, n_jobs=-1
    )

    # 2. Random Forest (tree baseline)
    models["random_forest"] = RandomForestClassifier(
        n_estimators=100,
        n_jobs=min(8, multiprocessing.cpu_count()),
        random_state=42,
    )

    # 3. XGBoost (primary)
    models["xgboost"] = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=min(8, multiprocessing.cpu_count()),
        random_state=42,
        verbosity=0,
    )

    # 4. LightGBM (comparison)
    models["lightgbm"] = LGBMClassifier(
        n_jobs=min(8, multiprocessing.cpu_count()),
        random_state=42,
        verbose=-1,
    )

    results: dict[str, dict] = {}
    trained_models: dict[str, object] = {}

    mlflow.set_experiment("fraud_detection")

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_resampled, y_resampled)
        metrics = evaluate_model_full(model, X_test, y_test)
        
        # Remove numpy arrays for logging
        metrics_log = {
            k: v
            for k, v in metrics.items()
            if k not in ["y_proba", "y_pred", "y_test"]
        }
        results[name] = metrics_log
        trained_models[name] = model

        # Filter metrics for MLflow logging (must be float/int)
        mlflow_metrics = {
            k: v
            for k, v in metrics_log.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }

        with mlflow.start_run(run_name=f"train_fraud_{name}"):
            mlflow.log_params(
                {
                    "model_name": name,
                    "n_features": X.shape[1],
                    "train_size": len(X_resampled),
                    "test_size": len(X_test),
                }
            )
            mlflow.log_metrics(mlflow_metrics)
            mlflow.sklearn.log_model(model, "model")

        # Print results
        print(f"  AUC-ROC: {metrics_log['auc_roc']:.4f}")
        print(f"  AUC-PR:  {metrics_log['auc_pr']:.4f}")
        print(f"  F1 (at threshold {metrics_log['optimal_threshold']:.3f}): {metrics_log['f1_optimal']:.4f}")
        print(f"  Precision: {metrics_log['precision_optimal']:.4f}")
        print(f"  Recall:    {metrics_log['recall_optimal']:.4f}")
        cm = metrics_log["confusion_matrix"]
        print(f"  Confusion Matrix: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")

        # Validate performance thresholds for tree-based models
        if name in ["random_forest", "xgboost", "lightgbm"]:
            validate_performance_thresholds(name, metrics_log)

        # Generate SHAP plot for tree-based models
        if name in ["random_forest", "xgboost", "lightgbm"]:
            plot_feature_importance_shap(model, X_test, name)

    best_name = choose_best_model(results)
    best_model = trained_models[best_name]
    joblib.dump(best_model, BEST_MODEL_PATH)
    save_final_models(trained_models, results, best_name)

    print(f"\n✅ Best model: {best_name} (AUC-PR: {results[best_name]['auc_pr']:.4f})")
    print(f"   Saved to {BEST_MODEL_PATH}")

    return results, X_test, y_test


def print_summary(results: dict[str, dict]) -> None:
    print("\n" + "=" * 80)
    print("MODEL COMPARISON REPORT")
    print("=" * 80)
    for name, metrics in results.items():
        print(f"\n{name.upper()}:")
        print(f"  AUC-ROC:            {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR:             {metrics['auc_pr']:.4f}")
        print(f"  F1 (optimal):       {metrics['f1_optimal']:.4f}")
        print(f"  Precision (optimal): {metrics['precision_optimal']:.4f}")
        print(f"  Recall (optimal):   {metrics['recall_optimal']:.4f}")
        print(f"  Optimal Threshold:  {metrics['optimal_threshold']:.4f}")
        cm = metrics["confusion_matrix"]
        print(f"  Confusion Matrix:   TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    print("=" * 80)


if __name__ == "__main__":
    results, _, _ = train_models()
    print_summary(results)
    print("\n✅ Training complete. All final models and the best model are saved.")
    print(f"   SHAP plots saved to {REPORT_DIR}/")
