"""Train the binary volatility-regime classifier (20-day forward).

Target: 1 iff next-20-day realised volatility is above the ticker's trailing
1-year rolling median volatility. This is a well-posed problem because
volatility clusters (GARCH effect, 40 years of empirical support — Engle 1982,
Bollerslev 1986). Expected accuracy substantially above 50% with proper
methodology.

Run:
    python -m models.train_volatility
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.pop("MLFLOW_TRACKING_URI", None)


def _resolve_mlruns_dir() -> Path:
    allusers = os.environ.get("ALLUSERSPROFILE")
    if allusers and " " not in allusers:
        return Path(allusers) / "finsight-ai" / "mlruns"
    return Path("C:/finsight-ai/mlruns")


_MLRUNS_DIR = _resolve_mlruns_dir().resolve()
_MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
_MLRUNS_URI = "file:///" + str(_MLRUNS_DIR).replace("\\", "/").lstrip("/")
os.environ["MLFLOW_TRACKING_URI"] = _MLRUNS_URI
os.environ["MLFLOW_REGISTRY_URI"] = _MLRUNS_URI

import mlflow  # noqa: E402
import mlflow.lightgbm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402

from models.feature_engineering import (  # noqa: E402
    FEATURE_COLS,
    TARGET_COL,
    build_features_multi,
)

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
    "TSLA", "JPM", "XOM", "JNJ", "WMT", "UNH", "V", "PG", "HD",
]

ARTIFACT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ARTIFACT_DIR / "volatility_predictor.pkl"

TEST_FRACTION = 0.20


def _temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = df.index.unique().sort_values()
    cutoff = dates[int(len(dates) * (1 - TEST_FRACTION))]
    return df[df.index < cutoff], df[df.index >= cutoff]


def _fit_params() -> dict:
    return dict(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=40,
        reg_alpha=0.1,
        reg_lambda=0.1,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def main() -> None:
    mlflow.set_tracking_uri(_MLRUNS_URI)
    mlflow.set_registry_uri(_MLRUNS_URI)
    print(f"[mlflow] tracking URI: {_MLRUNS_URI}")
    mlflow.set_experiment("volatility-regime")

    print(f"Building VOLATILITY features for {len(TICKERS)} tickers over 5y + SPY...")
    df = build_features_multi(TICKERS, period="5y", include_target=True, target_type="volatility")
    df = df.sort_index()
    print(f"Total rows: {len(df):,}   target balance: {df[TARGET_COL].mean():.3f} (fraction HIGH-vol)")

    train_df, test_df = _temporal_split(df)
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL].astype(int)
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL].astype(int)

    print(f"Train rows: {len(X_train):,}  |  Test rows: {len(X_test):,}")
    print(f"Train date range: {train_df.index.min().date()} .. {train_df.index.max().date()}")
    print(f"Test  date range: {test_df.index.min().date()} .. {test_df.index.max().date()}")

    with mlflow.start_run(run_name="lgbm_volatility_regime_20d"):
        params = _fit_params()
        mlflow.log_params(params)
        mlflow.log_param("tickers", ",".join(TICKERS))
        mlflow.log_param("period", "5y")
        mlflow.log_param("horizon_days", 20)
        mlflow.log_param("test_fraction", TEST_FRACTION)
        mlflow.log_param("features", ",".join(FEATURE_COLS))
        mlflow.log_param("classification", "binary_vol_regime")

        tscv = TimeSeriesSplit(n_splits=5)
        cv_accs, cv_f1s, cv_aucs = [], [], []
        X_sorted = X_train.sort_index()
        y_sorted = y_train.loc[X_sorted.index]
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_sorted), start=1):
            m = LGBMClassifier(**params)
            m.fit(X_sorted.iloc[tr_idx], y_sorted.iloc[tr_idx])
            va_pred = m.predict(X_sorted.iloc[va_idx])
            va_proba = m.predict_proba(X_sorted.iloc[va_idx])[:, 1]
            cv_accs.append(accuracy_score(y_sorted.iloc[va_idx], va_pred))
            cv_f1s.append(f1_score(y_sorted.iloc[va_idx], va_pred, average="macro"))
            cv_aucs.append(roc_auc_score(y_sorted.iloc[va_idx], va_proba))
            print(f"  CV fold {fold}: acc={cv_accs[-1]:.3f}  f1={cv_f1s[-1]:.3f}  auc={cv_aucs[-1]:.3f}")

        mlflow.log_metric("cv_accuracy_mean", float(np.mean(cv_accs)))
        mlflow.log_metric("cv_accuracy_std", float(np.std(cv_accs)))
        mlflow.log_metric("cv_f1_macro_mean", float(np.mean(cv_f1s)))
        mlflow.log_metric("cv_roc_auc_mean", float(np.mean(cv_aucs)))

        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        auc = roc_auc_score(y_test, proba)

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1_macro", f1)
        mlflow.log_metric("test_roc_auc", auc)

        print("\n=== CV (on train) ===")
        print(f"acc = {np.mean(cv_accs):.3f} ± {np.std(cv_accs):.3f}")
        print(f"f1  = {np.mean(cv_f1s):.3f}")
        print(f"auc = {np.mean(cv_aucs):.3f}")

        print("\n=== Held-out Test (future dates) ===")
        print(classification_report(y_test, preds, digits=3))
        print("Confusion matrix [rows=true, cols=pred]  (0=LOW, 1=HIGH):")
        print(confusion_matrix(y_test, preds))
        print(f"ROC-AUC: {auc:.3f}")

        importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        print("\nTop features:")
        print(importances.head(10))

        import joblib

        joblib.dump(
            {"model": model, "features": FEATURE_COLS, "classification": "vol_regime"},
            MODEL_PATH,
        )
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.lightgbm.log_model(model, name="model")
        print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
