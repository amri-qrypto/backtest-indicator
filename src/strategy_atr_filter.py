"""Trend-following strategy conditioned on ATR-driven volatility filter."""
from __future__ import annotations

import pandas as pd

from indicators import calculate_ema


def average_true_range(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute the Average True Range (ATR)."""
    required_columns = {"high", "low", "close"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns for ATR: {sorted(missing)}")

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    atr.name = f"ATR_{period}"
    return atr


def atr_filter_signals(
    df: pd.DataFrame,
    ema_span: int = 50,
    atr_period: int = 14,
    median_window: int = 100,
) -> pd.Series:
    """Return long/cash positions where trend and volatility filters align."""
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column")

    ema = calculate_ema(df["close"].astype(float), ema_span)
    atr = average_true_range(df, atr_period)
    atr_median = atr.rolling(window=median_window, min_periods=1).median()

    long_condition = (df["close"].astype(float) > ema) & (atr > atr_median)
    positions = long_condition.astype(float)
    positions = positions.ffill().fillna(0.0)
    positions.name = "position"

    return positions


__all__ = ["average_true_range", "atr_filter_signals"]
