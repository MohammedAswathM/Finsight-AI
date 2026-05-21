"""Feature builder for the 20-day price-momentum forecaster (Member 3).

Design:
- BINARY target: 1 = UP (20-day fwd return > 0), 0 = DOWN.
- 20-day horizon (not 5-day) because short-horizon direction is near-random
  (Fama 1970; Lo & MacKinlay 1988) while medium-term momentum is robust
  (Jegadeesh & Titman 1993; Moskowitz et al. 2012). See decisions/lessons.md.
- Drops low-signal "noise zone" (|return| < 2%) so the model isn't trained on
  coin-flip examples.
- Ticker features + SPY market features + lagged features.
- All columns coerced to float64 — LightGBM rejects `object`/`Float64`.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

TICKER_FEATURE_COLS = [
    "returns_1d",
    "returns_5d",
    "returns_20d",
    "vs_ma20_pct",
    "vs_ma50_pct",
    "volume_zscore",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "volatility_20d",
    # lag-1 versions (yesterday's values) — captures momentum continuation
    "returns_1d_lag1",
    "rsi_14_lag1",
    "macd_hist_lag1",
]
MARKET_FEATURE_COLS = [
    "mkt_returns_1d",
    "mkt_returns_5d",
    "mkt_volatility_20d",
    "mkt_rsi_14",
]
TIME_FEATURE_COLS = ["day_of_week", "month"]

FEATURE_COLS = TICKER_FEATURE_COLS + MARKET_FEATURE_COLS + TIME_FEATURE_COLS
TARGET_COL = "target"

# Forecast horizon (trading days).
FORECAST_HORIZON = 20

# Drop noise zone: moves smaller than this over 20 days are mostly noise.
# Tuned so ~15–20% of rows are dropped; keeps the task well-posed without
# over-thinning the training set.
NOISE_THRESHOLD = 0.02

# Rolling window for volatility regime baseline (~1 trading year).
VOL_REGIME_WINDOW = 252


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def _download(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=False,
        timeout=10,
    )
    if df.empty:
        raise ValueError(f"No yfinance data for ticker={ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _build_market_frame(period: str) -> pd.DataFrame:
    """SPY-based market-wide features merged by date into each ticker's frame."""
    spy = _download("SPY", period)
    close = spy["Close"]
    out = pd.DataFrame(index=spy.index)
    out["mkt_returns_1d"] = close.pct_change(1)
    out["mkt_returns_5d"] = close.pct_change(5)
    out["mkt_volatility_20d"] = close.pct_change().rolling(20).std()
    out["mkt_rsi_14"] = _rsi(close, 14)
    return out


def build_features(
    ticker: str,
    period: str = "5y",
    include_target: bool = True,
    market_frame: pd.DataFrame | None = None,
    target_type: str = "direction",
) -> pd.DataFrame:
    """Build feature frame for a single ticker.

    Pass a pre-computed `market_frame` (from `_build_market_frame`) when looping
    over many tickers to avoid re-downloading SPY each time.

    target_type:
      - "direction"  : 1 = next-20d return > 0 (noise zone dropped)
      - "volatility" : 1 = next-20d realised vol above 1-year rolling median
    """
    df = _download(ticker, period)
    close = df["Close"]
    volume = df["Volume"]

    df["returns_1d"] = close.pct_change(1)
    df["returns_5d"] = close.pct_change(5)
    df["returns_20d"] = close.pct_change(20)

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    df["vs_ma20_pct"] = (close - ma20) / ma20
    df["vs_ma50_pct"] = (close - ma50) / ma50

    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std().replace(0, np.nan)
    df["volume_zscore"] = (volume - vol_mean) / vol_std

    df["rsi_14"] = _rsi(close, 14)
    macd, signal, hist = _macd(close)
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = hist

    df["volatility_20d"] = close.pct_change().rolling(20).std()

    # Lagged features — yesterday's values
    df["returns_1d_lag1"] = df["returns_1d"].shift(1)
    df["rsi_14_lag1"] = df["rsi_14"].shift(1)
    df["macd_hist_lag1"] = df["macd_hist"].shift(1)

    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    # Merge market features by date.
    mkt = market_frame if market_frame is not None else _build_market_frame(period)
    df = df.join(mkt, how="left")

    if include_target:
        if target_type == "direction":
            future_ret = close.pct_change(FORECAST_HORIZON).shift(-FORECAST_HORIZON)
            df["_future_ret"] = future_ret
            df = df[future_ret.abs() >= NOISE_THRESHOLD]
            df[TARGET_COL] = (df["_future_ret"] > 0).astype(int)
        elif target_type == "volatility":
            # Realised forward volatility (annualised) over next 20 trading days.
            daily_ret = close.pct_change()
            future_vol = (
                daily_ret.rolling(FORECAST_HORIZON).std().shift(-FORECAST_HORIZON)
                * np.sqrt(252)
            )
            # Label = 1 iff next-20d vol is above the trailing 1y median vol
            # for this same ticker (regime shift, not absolute level).
            rolling_median = future_vol.rolling(VOL_REGIME_WINDOW).median()
            df["_future_vol"] = future_vol
            df[TARGET_COL] = (future_vol > rolling_median).astype(int)
            # Drop rows where rolling median wasn't defined yet.
            df = df[rolling_median.notna() & future_vol.notna()]
        else:
            raise ValueError(f"Unknown target_type={target_type!r}")

    keep = FEATURE_COLS + ([TARGET_COL] if include_target else [])
    out = df[keep].dropna().copy()

    for col in FEATURE_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    if include_target:
        out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce").astype("int64")
    return out.dropna()


def build_features_multi(
    tickers: Iterable[str],
    period: str = "5y",
    include_target: bool = True,
    target_type: str = "direction",
) -> pd.DataFrame:
    """Concat features across many tickers. Shares one SPY download."""
    mkt = _build_market_frame(period)
    frames = []
    for t in tickers:
        try:
            f = build_features(
                t,
                period=period,
                include_target=include_target,
                market_frame=mkt,
                target_type=target_type,
            )
            f["ticker"] = t
            frames.append(f)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] skipping {t}: {exc}")
    if not frames:
        raise RuntimeError("No ticker frames built.")
    return pd.concat(frames)


if __name__ == "__main__":
    out = build_features("AAPL")
    print(out.tail())
    print("Shape:", out.shape)
    print("Target balance:\n", out[TARGET_COL].value_counts(normalize=True))
