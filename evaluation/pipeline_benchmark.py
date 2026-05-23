"""Latency benchmark for the three AIML model wrappers."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable


def _time(label: str, fn: Callable[[], object]) -> tuple[str, float, object]:
    started = time.perf_counter()
    try:
        output = fn()
    except Exception as exc:  # noqa: BLE001
        output = {"error": str(exc)}
    return label, (time.perf_counter() - started) * 1000, output


def main() -> None:
    timings = []

    try:
        from models.fraud_detector import predict_fraud

        sample = {"Amount": 250.0, "Time": 3600.0, **{f"V{i}": 0.0 for i in range(1, 29)}}
        timings.append(_time("Fraud model", lambda: predict_fraud(sample)))
    except Exception as exc:  # noqa: BLE001
        timings.append(("Fraud model", 0.0, {"error": str(exc)}))

    finbert_dir = Path("models") / "finbert-finetuned"
    if finbert_dir.exists():
        from models.sentiment_model import predict_sentiment

        timings.append(_time("FinBERT", lambda: predict_sentiment("Apple beats Q4 earnings estimates")))
    else:
        timings.append(("FinBERT", 0.0, {"error": "models/finbert-finetuned not found"}))

    from models.forecaster import predict_trend

    timings.append(_time("Forecaster", lambda: predict_trend("AAPL")))

    print(" | ".join(f"{label}: {elapsed:.0f}ms" for label, elapsed, _ in timings))
    for label, _, output in timings:
        print(f"{label} output: {output}")


if __name__ == "__main__":
    main()
