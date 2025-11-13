"""Indicator calculation utilities."""
from __future__ import annotations

import pandas as pd


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate an exponential moving average (EMA) using pandas."""
    if span <= 0:
        raise ValueError("span must be a positive integer")
    return series.ewm(span=span, adjust=False).mean()


__all__ = ["calculate_ema"]
