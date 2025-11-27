"""Lightweight strategy helpers for notebooks and quick experiments."""
from __future__ import annotations

import pandas as pd

from .strategy_ema import ema_trend_signals
from .strategy_backtest.strategies.vwap import Strategy as VWAPStrategy


def ema_vs_price_signals(
    df: pd.DataFrame,
    ema_span: int = 112,
    price_col: str = "close",
) -> pd.Series:
    """Generate long/cash signals using a configurable EMA span."""

    return ema_trend_signals(df, span=ema_span, price_col=price_col)


def vwap_mean_reversion_signals(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Convenience wrapper around the VWAP strategy implementation.

    Parameters mirror :class:`src.strategy_backtest.strategies.vwap.Strategy` so
    notebooks can fetch both the raw entry/exit signals and indicator context
    (VWAP, RSI, ATR, stop/take-profit levels) without dealing with the
    backtester registry.
    """

    strategy = VWAPStrategy(**kwargs)
    return strategy.generate_signals(df)


__all__ = ["ema_vs_price_signals", "vwap_mean_reversion_signals"]
