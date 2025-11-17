"""Reusable EMA-based strategy helpers."""
from __future__ import annotations

import pandas as pd

from .indicators import calculate_ema


def ema_trend_signals(
    df: pd.DataFrame,
    span: int,
    price_col: str = "close",
) -> pd.Series:
    """Return long/cash signals where price trades above its EMA."""

    _, _, positions = ema_trend_context(df, span=span, price_col=price_col)
    positions = positions.astype(float)
    positions.name = "position"

    return positions


def ema_trend_context(
    df: pd.DataFrame,
    span: int,
    price_col: str = "close",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return price, EMA, and boolean trend condition series."""

    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain a '{price_col}' column")

    close = df[price_col].astype(float)
    ema = calculate_ema(close, span)
    long_condition = (close > ema).fillna(False)

    return close, ema, long_condition


__all__ = ["ema_trend_signals", "ema_trend_context"]

