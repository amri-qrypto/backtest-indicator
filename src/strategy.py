"""Signal generation for the EMA vs. price strategy."""
from __future__ import annotations

import pandas as pd


def ema_vs_price_signals(df: pd.DataFrame, ema_col: str = "EMA") -> pd.Series:
    """Generate long/cash signals based on the relationship between close and EMA."""
    if ema_col not in df.columns:
        raise ValueError(f"EMA column '{ema_col}' not found in DataFrame")

    close = df["close"].astype(float)
    ema = df[ema_col].astype(float)

    positions = (close > ema).astype(int)
    positions = positions.reindex(df.index).fillna(0).astype(int)
    positions.name = "position"

    return positions


__all__ = ["ema_vs_price_signals"]
