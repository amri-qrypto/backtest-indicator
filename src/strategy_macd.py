"""Moving Average Convergence Divergence (MACD) strategy helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def macd_components(
    close: pd.Series,
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
) -> pd.DataFrame:
    """Return MACD line, signal line, and histogram values."""
    if close.empty:
        raise ValueError("close series must not be empty")

    fast_ema = close.ewm(span=fast_span, adjust=False).mean()
    slow_ema = close.ewm(span=slow_span, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
    histogram = macd_line - signal_line

    data = pd.DataFrame(
        {
            "macd": macd_line.astype(float),
            "macd_signal": signal_line.astype(float),
            "macd_hist": histogram.astype(float),
        },
        index=close.index,
    )
    return data


def macd_signals(
    df: pd.DataFrame,
    price_col: str = "close",
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
) -> pd.Series:
    """Generate long/cash positions based on MACD and signal line crossovers."""
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain a '{price_col}' column")

    macd_df = macd_components(df[price_col].astype(float), fast_span, slow_span, signal_span)

    comparison = np.where(
        macd_df["macd"] > macd_df["macd_signal"],
        1.0,
        np.where(macd_df["macd"] < macd_df["macd_signal"], 0.0, np.nan),
    )
    positions = pd.Series(comparison, index=macd_df.index, dtype=float)
    positions = positions.ffill().fillna(0.0)
    positions.name = "position"

    return positions


__all__ = ["macd_components", "macd_signals"]
