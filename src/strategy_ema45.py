"""EMA(45) trend-following strategy."""
from __future__ import annotations

import pandas as pd

from .strategy_ema import ema_trend_signals


def generate_signals(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """Return long/cash positions based on a 45-period EMA."""

    return ema_trend_signals(df, span=45, price_col=price_col)


__all__ = ["generate_signals"]

