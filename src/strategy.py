"""Signal generation for EMA based strategies."""
from __future__ import annotations

import pandas as pd

from strategy_ema import ema_trend_signals


def ema_vs_price_signals(
    df: pd.DataFrame,
    ema_span: int = 112,
    price_col: str = "close",
) -> pd.Series:
    """Generate long/cash signals using a configurable EMA span."""

    return ema_trend_signals(df, span=ema_span, price_col=price_col)


__all__ = ["ema_vs_price_signals"]
