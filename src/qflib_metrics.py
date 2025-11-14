"""Performance metrics powered by QF-Lib's :class:`TimeseriesAnalysis`."""
from __future__ import annotations

from math import sqrt
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .qflib_adapters import to_qfseries

try:  # pragma: no cover - optional dependency at runtime
    from qf_lib.analysis.timeseries_analysis.timeseries_analysis import TimeseriesAnalysis
    from qf_lib.common.enums.frequency import Frequency
except ImportError:  # pragma: no cover - handled via fallback implementation
    TimeseriesAnalysis = None  # type: ignore[assignment]
    Frequency = None  # type: ignore[assignment]


def _call_first_available(obj: object, method_names: Iterable[str]) -> float:
    """Call the first available method on ``obj`` and return the result as ``float``."""
    for name in method_names:
        if hasattr(obj, name):
            method = getattr(obj, name)
            value = method()
            try:
                return float(value)
            except TypeError:  # pragma: no cover - defensive for unexpected outputs
                return float(value[0])
    raise AttributeError(f"None of the methods {list(method_names)} exist on {type(obj)!r}")


def _fallback_metrics(returns: pd.Series) -> Dict[str, float]:
    """Compute metrics without QF-Lib as a graceful fallback."""
    returns = returns.astype(float).fillna(0.0)

    cumulative = (1.0 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1.0

    periods = len(returns)
    annualisation_factor = 252
    periods_for_cagr = max(periods, 1)
    cagr = (1.0 + total_return) ** (annualisation_factor / periods_for_cagr) - 1.0

    mean_return = returns.mean()
    volatility = returns.std(ddof=0)
    annualised_vol = volatility * sqrt(annualisation_factor)
    sharpe_ratio = (mean_return * annualisation_factor / annualised_vol) if annualised_vol > 0 else np.nan

    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    max_drawdown = float(drawdown.min())

    in_drawdown = drawdown < 0
    group_ids = (in_drawdown != in_drawdown.shift()).cumsum()
    drawdown_lengths = in_drawdown.groupby(group_ids).sum()
    active_lengths = drawdown_lengths[drawdown_lengths > 0]
    avg_drawdown_duration = float(active_lengths.mean()) if not active_lengths.empty else 0.0

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe_ratio": float(sharpe_ratio) if np.isfinite(sharpe_ratio) else np.nan,
        "annualised_vol": float(annualised_vol),
        "max_drawdown": max_drawdown,
        "avg_drawdown_duration": avg_drawdown_duration,
    }


def qflib_metrics_from_returns(returns: pd.Series) -> Dict[str, float]:
    """Return a dictionary of performance statistics calculated via QF-Lib.

    Parameters
    ----------
    returns:
        Daily strategy returns indexed by ``DatetimeIndex``.
    """

    if returns.empty:
        raise ValueError("returns series must not be empty")

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns series must use a DatetimeIndex")

    returns = returns.sort_index().astype(float).fillna(0.0)

    if TimeseriesAnalysis is None or Frequency is None:
        return _fallback_metrics(returns)

    try:
        qf_series = to_qfseries(returns)
        analysis = TimeseriesAnalysis(qf_series, Frequency.DAILY)
    except NotImplementedError:
        # ``QFSeries.to_simple_returns`` is abstract in older qf-lib releases. When it isn't
        # implemented we gracefully fall back to the local numpy/pandas based metrics so that
        # notebooks can still be executed without patching the dependency.
        return _fallback_metrics(returns)

    total_return = _call_first_available(analysis, ("total_return",))
    cagr = _call_first_available(analysis, ("cagr",))
    sharpe_ratio = _call_first_available(
        analysis,
        ("sharpe_ratio", "annualised_sharpe_ratio", "annualised_sharpe"),
    )
    annualised_vol = _call_first_available(
        analysis,
        ("annualised_volatility", "annualised_vol", "annualized_volatility"),
    )
    max_drawdown = _call_first_available(analysis, ("max_drawdown", "drawdown"))
    avg_drawdown_duration = _call_first_available(
        analysis,
        (
            "average_drawdown_length",
            "average_drawdown_duration",
            "avg_drawdown_length",
            "avg_drawdown_duration",
        ),
    )

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe_ratio": sharpe_ratio,
        "annualised_vol": annualised_vol,
        "max_drawdown": max_drawdown,
        "avg_drawdown_duration": avg_drawdown_duration,
    }


__all__ = ["qflib_metrics_from_returns"]
