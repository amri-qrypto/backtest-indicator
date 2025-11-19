"""Helpers for inferring bar frequency and elapsed time from indices."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SECONDS_PER_MINUTE = 60.0
MINUTES_PER_HOUR = 60.0
HOURS_PER_DAY = 24.0
DAYS_PER_YEAR = 365.0
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR
SECONDS_PER_DAY = SECONDS_PER_HOUR * HOURS_PER_DAY
SECONDS_PER_YEAR = SECONDS_PER_DAY * DAYS_PER_YEAR
DEFAULT_BARS_PER_DAY = 24.0
DEFAULT_BARS_PER_YEAR = DEFAULT_BARS_PER_DAY * DAYS_PER_YEAR


@dataclass(frozen=True)
class TimeFactors:
    """Convenience container for the inferred timing information."""

    bars_per_year: float
    days_between: float


def _median_seconds_between(index: pd.DatetimeIndex) -> float:
    """Return the median distance between consecutive timestamps in seconds."""

    if len(index) < 2:
        return float("nan")
    diffs = np.diff(index.view("i8")) / 1e9
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return float("nan")
    return float(np.median(positive_diffs))


def resolve_time_factors(
    index: pd.Index | None,
    *,
    bars_per_year_hint: float | None = None,
    n_bars: int,
    default_bars_per_year: float = DEFAULT_BARS_PER_YEAR,
) -> TimeFactors:
    """Infer ``bars_per_year`` and elapsed days from an optional ``DatetimeIndex``."""

    bars_per_year = (
        float(bars_per_year_hint)
        if bars_per_year_hint is not None
        else float(default_bars_per_year)
    )
    days_between = float("nan")

    if isinstance(index, pd.DatetimeIndex) and len(index) >= 2:
        values = index.view("i8")
        elapsed_seconds = float(values[-1] - values[0]) / 1e9
        if elapsed_seconds > 0:
            days_between = elapsed_seconds / SECONDS_PER_DAY
        if bars_per_year_hint is None:
            median_seconds = _median_seconds_between(index)
            if np.isfinite(median_seconds) and median_seconds > 0:
                bars_per_year = SECONDS_PER_YEAR / median_seconds

    if (not np.isfinite(days_between)) or days_between <= 0:
        if bars_per_year > 0 and n_bars > 0:
            years = n_bars / bars_per_year
            days_between = years * DAYS_PER_YEAR
        else:
            days_between = float("nan")

    return TimeFactors(bars_per_year=bars_per_year, days_between=days_between)


__all__ = [
    "TimeFactors",
    "DAYS_PER_YEAR",
    "DEFAULT_BARS_PER_DAY",
    "DEFAULT_BARS_PER_YEAR",
    "resolve_time_factors",
]
