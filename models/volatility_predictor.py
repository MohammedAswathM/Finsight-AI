"""Inference wrapper for the volatility-regime classifier."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

_MODEL_PATH = Path(__file__).parent / "volatility_predictor.pkl"
_cache: Dict = {}


def _load():
    if "bundle" in _cache:
        return _cache["bundle"]
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Volatility model not trained yet. Run `python -m models.train_volatility` "
            f"(expected at {_MODEL_PATH})."
        )
    import joblib

    _cache["bundle"] = joblib.load(_MODEL_PATH)
    return _cache["bundle"]


def predict_volatility(ticker: str, days_ahead: int = 20) -> dict:
    """Predict next-20-day volatility regime for a ticker.

    Returns {ticker, regime, confidence, high_probability, days_ahead}.
    regime is HIGH / LOW / UNAVAILABLE.
    HIGH means next-20d realised volatility is expected to be above the
    ticker's trailing 1-year median volatility.
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
        high_prob = float(proba[list(model.classes_).index(1)])

        regime = "HIGH" if high_prob >= 0.5 else "LOW"
        confidence = high_prob if regime == "HIGH" else 1 - high_prob

        return {
            "ticker": ticker.upper(),
            "regime": regime,
            "confidence": round(confidence, 4),
            "high_probability": round(high_prob, 4),
            "days_ahead": days_ahead,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ticker": ticker.upper(),
            "regime": "UNAVAILABLE",
            "confidence": 0.0,
            "days_ahead": days_ahead,
            "error": str(exc),
        }


if __name__ == "__main__":
    for t in ["AAPL", "NVDA", "TSLA"]:
        print(t, predict_volatility(t))
