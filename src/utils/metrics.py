"""Shared helpers for computing comparable portfolio metrics across pipelines."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from ..performance.metrics import summarise_bar_returns


STANDARD_METRIC_KEYS = ("total_return", "cagr", "max_drawdown", "hit_rate")


def portfolio_metrics_from_returns(
    returns: Sequence[float] | pd.Series, *, bars_per_year: float | None = None
) -> dict[str, float]:
    """Calculate a minimal set of comparable portfolio metrics from returns.

    The returned dictionary is intentionally small (return, drawdown, hit rate)
    so that indicator backtests and ML signal backtests can persist the same
    schema to JSON artifacts.
    """

    summary = summarise_bar_returns(returns, bars_per_year=bars_per_year)
    return {key: float(summary.get(key, float("nan"))) for key in STANDARD_METRIC_KEYS}


def portfolio_metrics_from_equity(
    equity_curve: pd.Series, *, bars_per_year: float | None = None
) -> dict[str, float]:
    """Convert an equity curve into the standard portfolio metrics payload."""

    if equity_curve.empty:
        raise ValueError("equity_curve must not be empty")

    returns = equity_curve.pct_change().fillna(0.0)
    return portfolio_metrics_from_returns(returns, bars_per_year=bars_per_year)


__all__ = [
    "portfolio_metrics_from_equity",
    "portfolio_metrics_from_returns",
    "STANDARD_METRIC_KEYS",
]
