"""Utility functions to compute return statistics for notebooks and scripts."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

import numpy as np
import pandas as pd

from .time_utils import (
    DAYS_PER_YEAR,
    DEFAULT_BARS_PER_DAY,
    DEFAULT_BARS_PER_YEAR,
    resolve_time_factors,
)


@dataclass(frozen=True)
class ReturnSummary:
    """Container for standardised performance statistics."""

    total_return: float
    cagr: float
    annualised_vol: float
    sharpe_ratio: float
    max_drawdown: float
    average_return: float
    volatility: float
    hit_rate: float
    n_bars: int
    bars_per_year: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "annualised_vol": self.annualised_vol,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "average_return": self.average_return,
            "volatility": self.volatility,
            "hit_rate": self.hit_rate,
            "n_bars": float(self.n_bars),
            "bars_per_year": self.bars_per_year,
        }


def _coerce_returns(returns: Sequence[float] | pd.Series) -> pd.Series:
    if isinstance(returns, pd.Series):
        series = returns.astype(float).copy()
    else:
        series = pd.Series(returns, dtype=float)
    return series.fillna(0.0)


def summarise_bar_returns(
    returns: Sequence[float] | pd.Series,
    *,
    bars_per_year: float | None = None,
) -> dict[str, float]:
    """Calculate CAGR, volatility, Sharpe, and drawdown from bar returns."""

    series = _coerce_returns(returns)
    n_bars = int(len(series))
    if n_bars == 0:
        return ReturnSummary(
            total_return=float("nan"),
            cagr=float("nan"),
            annualised_vol=float("nan"),
            sharpe_ratio=float("nan"),
            max_drawdown=float("nan"),
            average_return=float("nan"),
            volatility=float("nan"),
            hit_rate=float("nan"),
            n_bars=0,
            bars_per_year=float(bars_per_year or DEFAULT_BARS_PER_YEAR),
        ).to_dict()

    index = series.index if isinstance(series.index, pd.DatetimeIndex) else None
    factors = resolve_time_factors(
        index,
        bars_per_year_hint=bars_per_year,
        n_bars=n_bars,
        default_bars_per_year=DEFAULT_BARS_PER_YEAR,
    )
    annualisation = factors.bars_per_year

    cumulative = (1.0 + series).cumprod()
    ending_value = float(cumulative.iloc[-1])
    total_return = ending_value - 1.0

    mean_return = float(series.mean())
    volatility = float(series.std(ddof=0))
    annualised_vol = float(volatility * sqrt(annualisation)) if volatility > 0 else float(0.0)
    annualised_return = float(mean_return * annualisation)
    sharpe_ratio = (
        float(annualised_return / annualised_vol)
        if annualised_vol > 0
        else float("nan")
    )

    if np.isfinite(factors.days_between) and factors.days_between > 0:
        cagr = ending_value ** (DAYS_PER_YEAR / factors.days_between) - 1.0
    else:
        cagr = float("nan")

    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    max_drawdown = float(drawdown.min())

    hit_rate = float((series > 0).mean())

    summary = ReturnSummary(
        total_return=float(total_return),
        cagr=float(cagr),
        annualised_vol=float(annualised_vol),
        sharpe_ratio=float(sharpe_ratio),
        max_drawdown=max_drawdown,
        average_return=mean_return,
        volatility=volatility,
        hit_rate=hit_rate,
        n_bars=n_bars,
        bars_per_year=annualisation,
    )
    return summary.to_dict()


def summarise_fold_performance(
    folds: Sequence[tuple[int, pd.DataFrame]],
    *,
    return_column: str = "future_return",
    bars_per_year: float | None = None,
) -> pd.DataFrame:
    """Summarise walk-forward splits with mean return, vol, Sharpe, and hit rate."""

    records: list[dict[str, object]] = []

    for fold_id, frame in folds:
        if return_column not in frame.columns:
            raise KeyError(f"Kolom '{return_column}' tidak ditemukan pada fold {fold_id}")
        returns = frame[return_column].astype(float).fillna(0.0)
        if returns.empty:
            continue
        stats = summarise_bar_returns(returns, bars_per_year=bars_per_year)
        records.append(
            {
                "fold": fold_id,
                "start_time": returns.index.min(),
                "end_time": returns.index.max(),
                "n_bars": stats["n_bars"],
                "mean_return": stats["average_return"],
                "volatility": stats["volatility"],
                "annualised_vol": stats["annualised_vol"],
                "sharpe_ratio": stats["sharpe_ratio"],
                "hit_rate": stats["hit_rate"],
            }
        )

    if not records:
        columns = [
            "fold",
            "start_time",
            "end_time",
            "n_bars",
            "mean_return",
            "volatility",
            "annualised_vol",
            "sharpe_ratio",
            "hit_rate",
        ]
        return pd.DataFrame(columns=columns).set_index("fold")

    frame = pd.DataFrame(records).set_index("fold")
    return frame


__all__ = [
    "DEFAULT_BARS_PER_DAY",
    "DEFAULT_BARS_PER_YEAR",
    "ReturnSummary",
    "summarise_bar_returns",
    "summarise_fold_performance",
]
