"""Reusable technical feature engineering blocks."""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

PRICE_COL = "close"
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


def build_technical_features(
    df: pd.DataFrame,
    *,
    return_horizons: Sequence[int] = (1, 4, 12, 24),
    momentum_windows: Sequence[int] = (6, 12, 24, 48),
    volatility_windows: Sequence[int] = (6, 24, 72),
    volume_windows: Sequence[int] = (6, 24),
) -> pd.DataFrame:
    """Generate a consolidated ``DataFrame`` containing engineered technical factors."""

    _require_columns(df, [PRICE_COL, VOLUME_COL])

    prices = df[PRICE_COL]
    volumes = df[VOLUME_COL]

    feature_blocks: list[pd.Series] = []
    for block in (
        _multi_horizon_returns(prices, return_horizons),
        _rolling_momentum(prices, momentum_windows),
        _rolling_volatility(prices, volatility_windows),
        _volume_change(volumes, volume_windows),
    ):
        feature_blocks.extend(block.values())

    features = pd.concat(feature_blocks, axis=1)
    features.index = df.index
    return features


__all__ = ["build_technical_features"]
