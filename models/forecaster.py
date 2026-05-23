"""Inference wrapper for the 5-day price-trend forecaster (Member 3).

Binary classifier under the hood (UP vs DOWN). We synthesise a FLAT label at
inference time when the model's confidence is low — gives a 3-class user-facing
output while keeping training well-posed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

_MODEL_PATH = Path(__file__).parent / "forecaster.pkl"
_cache: Dict = {}

# Calibration-independent; empirical — re-tune if the confusion matrix shifts.
FLAT_CONFIDENCE_BAND = 0.55  # if max-class proba < this, report FLAT


def _load():
    if "bundle" in _cache:
        return _cache["bundle"]
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Forecaster not trained yet. Run `python -m models.train_forecaster` "
            f"(expected at {_MODEL_PATH})."
        )
    import joblib

    _cache["bundle"] = joblib.load(_MODEL_PATH)
    return _cache["bundle"]


def predict_trend(ticker: str, days_ahead: int = 20) -> dict:
    """Predict 20-day forward momentum direction for a ticker.

    Returns {ticker, direction, confidence, days_ahead}.
    direction is UP / DOWN / FLAT. FLAT is emitted when model confidence is
    below FLAT_CONFIDENCE_BAND (model is uncertain).
    Falls back to direction='UNAVAILABLE' if the model/data is missing.
    """
    try:
        bundle = _load()
        model = bundle["model"]
        features = bundle["features"]

        from models.feature_engineering import build_features

        df = build_features(ticker, period="1y", include_target=False)
        if df.empty:
            raise ValueError("No recent features available.")

        latest = df[features].iloc[[-1]]
        proba = model.predict_proba(latest)[0]
        # Binary model: classes_ is [0, 1] where 1 = UP
        up_prob = float(proba[list(model.classes_).index(1)])

        if up_prob >= FLAT_CONFIDENCE_BAND:
            direction = "UP"
            confidence = up_prob
        elif up_prob <= 1 - FLAT_CONFIDENCE_BAND:
            direction = "DOWN"
            confidence = 1 - up_prob
        else:
            direction = "FLAT"
            confidence = 1 - abs(up_prob - 0.5) * 2  # 1.0 when exactly 0.5

        return {
            "ticker": ticker.upper(),
            "direction": direction,
            "confidence": round(confidence, 4),
            "up_probability": round(up_prob, 4),
            "days_ahead": days_ahead,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ticker": ticker.upper(),
            "direction": "UNAVAILABLE",
            "confidence": 0.0,
            "days_ahead": days_ahead,
            "error": str(exc),
        }


if __name__ == "__main__":
    for t in ["AAPL", "NVDA", "MSFT", "TSLA"]:
        print(t, predict_trend(t))
