"""Collect local AIML model evidence for the infrastructure report."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


def _mlruns_roots() -> List[Path]:
    return [Path("mlruns"), Path("C:/ProgramData/finsight-ai/mlruns"), Path("C:/finsight-ai/mlruns")]


def _read_metrics(run_dir: Path) -> Dict[str, float]:
    metrics = {}
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.exists():
        return metrics
    for file in metrics_dir.iterdir():
        try:
            line = file.read_text().strip().splitlines()[-1]
            metrics[file.name] = float(line.split()[-1])
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
                rows.append({"run_path": str(run_dir), **metrics})
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
    print("\n=== MLflow Metrics ===")
    print(metrics.to_string(index=False) if not metrics.empty else "No local MLflow metrics found.")
    print("\n=== Wrapper Examples ===")
    print(wrapper_examples().to_string(index=False))


if __name__ == "__main__":
    main()
