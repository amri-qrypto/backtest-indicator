"""Reusable EMA-based strategy helpers."""
from __future__ import annotations

import pandas as pd

from indicators import calculate_ema


def ema_trend_signals(
    df: pd.DataFrame,
    span: int,
    price_col: str = "close",
) -> pd.Series:
    """Return long/cash signals where price trades above its EMA."""

    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain a '{price_col}' column")

    close = df[price_col].astype(float)
    ema = calculate_ema(close, span)

    positions = (close > ema).astype(float)
    positions.name = "position"

    return positions


__all__ = ["ema_trend_signals"]

