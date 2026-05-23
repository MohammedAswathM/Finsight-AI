"""Collect local AIML model evidence for the infrastructure report."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


def _mlruns_roots() -> List[Path]:
    return [Path("mlruns")]


def _read_metrics(run_dir: Path) -> Dict[str, float]:
    metrics = {}
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.exists():
        return metrics
    for file in metrics_dir.iterdir():
        try:
            # MLflow metric files: each line is "<timestamp_ms> <value> <step>".
            # We want the value (index 1), not the step (index -1).
            line = file.read_text().strip().splitlines()[-1]
            parts = line.split()
            metrics[file.name] = float(parts[1] if len(parts) >= 2 else parts[0])
        except Exception:
            continue
    return metrics


def collect_mlflow_metrics() -> pd.DataFrame:
    rows = []
    for root in _mlruns_roots():
        if not root.exists():
            continue
        for run_dir in root.glob("*/*"):
            metrics = _read_metrics(run_dir)
            if metrics:
                rows.append(
                    {
                        "experiment": _experiment_name(run_dir.parent),
                        "run_id": run_dir.name,
                        "run_path": str(run_dir),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def _experiment_name(experiment_dir: Path) -> str:
    meta = experiment_dir / "meta.yaml"
    if not meta.exists():
        return experiment_dir.name
    for line in meta.read_text(errors="ignore").splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return experiment_dir.name


def _compact_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return one readable best-run row per MLflow experiment."""
    if metrics.empty:
        return metrics

    priority = [
        "finetuned_f1_macro",
        "test_f1_macro",
        "directional_accuracy",
        "cv_roc_auc_mean",
        "base_f1_macro",
    ]
    rows = []
    for experiment, group in metrics.groupby("experiment", dropna=False):
        score_col = next((col for col in priority if col in group and group[col].notna().any()), None)
        if score_col:
            best = group.loc[group[score_col].astype(float).idxmax()].copy()
        else:
            best = group.iloc[-1].copy()

        row = {
            "experiment": experiment,
            "best_metric": score_col or "n/a",
            "score": best.get(score_col) if score_col else None,
            "run_id": best.get("run_id", ""),
        }
        metric_parts = []
        for col in group.columns:
            if col in {"experiment", "run_id", "run_path"}:
                continue
            value = best.get(col)
            if pd.notna(value):
                metric_parts.append(f"{col}={value:.4f}")
        row["metrics"] = "; ".join(metric_parts)
        rows.append(row)
    return pd.DataFrame(rows)


def wrapper_examples() -> pd.DataFrame:
    rows = []

    try:
        from models.fraud_detector import predict_fraud

        sample = {"Amount": 250.0, "Time": 3600.0, **{f"V{i}": 0.0 for i in range(1, 29)}}
        rows.append({"model": "Fraud Detector", "example_output": str(predict_fraud(sample))})
    except Exception as exc:  # noqa: BLE001
        rows.append({"model": "Fraud Detector", "example_output": f"Unavailable: {exc}"})

    try:
        finbert_dir = Path("models") / "finbert-finetuned"
        if finbert_dir.exists():
            from models.sentiment_model import predict_sentiment

            output = predict_sentiment("Apple beats Q4 earnings estimates")
        else:
            output = {"error": "models/finbert-finetuned not found"}
        rows.append({"model": "FinBERT Sentiment", "example_output": str(output)})
    except Exception as exc:  # noqa: BLE001
        rows.append({"model": "FinBERT Sentiment", "example_output": f"Unavailable: {exc}"})

    try:
        from models.forecaster import predict_trend

        rows.append({"model": "Price Forecaster", "example_output": str(predict_trend("AAPL"))})
    except Exception as exc:  # noqa: BLE001
        rows.append({"model": "Price Forecaster", "example_output": f"Unavailable: {exc}"})

    return pd.DataFrame(rows)


def main() -> None:
    metrics = collect_mlflow_metrics()
    print("\n=== MLflow Best-Run Summary ===")
    compact = _compact_metrics(metrics)
    print(compact.to_string(index=False) if not compact.empty else "No local MLflow metrics found.")
    print("\n=== Wrapper Examples ===")
    print(wrapper_examples().to_string(index=False))


if __name__ == "__main__":
    main()
