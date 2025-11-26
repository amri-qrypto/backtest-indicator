"""Reusable technical feature engineering blocks."""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

PRICE_COL = "close"
HIGH_COL = "high"
LOW_COL = "low"
VOLUME_COL = "volume"


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _multi_horizon_returns(prices: pd.Series, horizons: Sequence[int]) -> Mapping[str, pd.Series]:
    features = {}
    log_prices = np.log(prices)
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError("Return horizons must be positive integers.")
        name = f"ret_{horizon}h"
        series = log_prices.diff(periods=horizon).rename(name)
        features[name] = series
    return features


def _rolling_momentum(prices: pd.Series, windows: Sequence[int]) -> Mapping[str, pd.Series]:
    features = {}
    for window in windows:
        if window <= 0:
            raise ValueError("Momentum windows must be positive integers.")
        name = f"momentum_{window}h"
        series = prices.pct_change(periods=window).rename(name)
        features[name] = series
    return features


def _rolling_volatility(prices: pd.Series, windows: Sequence[int]) -> Mapping[str, pd.Series]:
    returns = prices.pct_change()
    features = {}
    for window in windows:
        if window <= 1:
            raise ValueError("Volatility windows must be greater than 1.")
        name = f"volatility_{window}h"
        series = returns.rolling(window).std().rename(name)
        features[name] = series
    return features


def _volume_change(volumes: pd.Series, windows: Sequence[int]) -> Mapping[str, pd.Series]:
    features = {}
    for window in windows:
        if window <= 0:
            raise ValueError("Volume windows must be positive integers.")
        name = f"volume_change_{window}h"
        series = volumes.pct_change(periods=window).rename(name)
        features[name] = series
    return features


def _average_directional_index(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> Mapping[str, pd.Series]:
    if window <= 1:
        raise ValueError("ADX window must be greater than 1.")

    high_diff = high.diff()
    low_diff = low.shift(1) - low

    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)

    tr_components = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)

    alpha = 1 / window
    smoothed_tr = true_range.ewm(alpha=alpha, adjust=False).mean()
    smoothed_plus_dm = pd.Series(plus_dm, index=high.index).ewm(
        alpha=alpha, adjust=False
    ).mean()
    smoothed_minus_dm = pd.Series(minus_dm, index=high.index).ewm(
        alpha=alpha, adjust=False
    ).mean()

    plus_di = (smoothed_plus_dm / smoothed_tr) * 100
    minus_di = (smoothed_minus_dm / smoothed_tr) * 100

    denominator = (plus_di + minus_di).replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / denominator) * 100
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return {
        f"plus_di_{window}": plus_di.rename(f"plus_di_{window}"),
        f"minus_di_{window}": minus_di.rename(f"minus_di_{window}"),
        f"adx_{window}": adx.rename(f"adx_{window}"),
    }


def build_technical_features(
    df: pd.DataFrame,
    *,
    return_horizons: Sequence[int] = (1, 4, 12, 24),
    momentum_windows: Sequence[int] = (6, 12, 24, 48),
    volatility_windows: Sequence[int] = (6, 24, 72),
    volume_windows: Sequence[int] = (6, 24),
    adx_window: int = 14,
) -> pd.DataFrame:
    """Generate a consolidated ``DataFrame`` containing engineered technical factors."""

    _require_columns(df, [PRICE_COL, HIGH_COL, LOW_COL, VOLUME_COL])

    prices = df[PRICE_COL]
    high = df[HIGH_COL]
    low = df[LOW_COL]
    volumes = df[VOLUME_COL]

    feature_blocks: list[pd.Series] = []
    for block in (
        _multi_horizon_returns(prices, return_horizons),
        _rolling_momentum(prices, momentum_windows),
        _rolling_volatility(prices, volatility_windows),
        _volume_change(volumes, volume_windows),
        _average_directional_index(high, low, prices, adx_window),
    ):
        feature_blocks.extend(block.values())

    features = pd.concat(feature_blocks, axis=1)
    features.index = df.index
    return features


__all__ = ["build_technical_features"]
