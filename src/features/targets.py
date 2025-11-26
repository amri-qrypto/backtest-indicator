"""Label engineering helpers for supervised ML experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data_loader import load_ohlcv_csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ETHBTC_1H_PATH = PROJECT_ROOT / "data" / "BINANCE_ETHUSDT.P, 60.csv"


def load_ethbtc_1h(path: str | Path | None = None, *, additional_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Return a cleaned ETH/BTC 1h OHLCV ``DataFrame`` using the shared loader."""

    csv_path = Path(path) if path is not None else DEFAULT_ETHBTC_1H_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ETH/BTC 1h CSV not found at {csv_path!s}. Provide a valid path via the 'path' argument."
        )

    df = load_ohlcv_csv(str(csv_path), additional_columns=additional_columns)
    return df


def _validate_horizon_and_column(df: pd.DataFrame, horizon: int, price_column: str) -> None:
    if price_column not in df.columns:
        raise KeyError(f"'{price_column}' column not present in input DataFrame.")
    if horizon <= 0:
        raise ValueError("'horizon' must be a positive integer representing hours ahead.")


def make_forward_return(
    df: pd.DataFrame,
    *,
    horizon: int = 4,
    price_column: str = "close",
    return_type: str = "simple",
) -> pd.Series:
    """Return the future return over ``horizon`` bars using the requested convention."""

    _validate_horizon_and_column(df, horizon, price_column)

    future_price = df[price_column].shift(-horizon)
    current_price = df[price_column]
    if return_type == "simple":
        forward_return = (future_price / current_price) - 1.0
    elif return_type == "log":
        forward_return = np.log(future_price) - np.log(current_price)
    else:  # pragma: no cover - defensive branch for notebook experiments
        raise ValueError("'return_type' must be either 'simple' or 'log'.")

    return forward_return.rename(f"forward_{return_type}_return_{horizon}h")


def make_forward_return_sign(
    df: pd.DataFrame,
    horizon: int = 4,
    price_column: str = "close",
    *,
    binary: bool = True,
) -> pd.Series:
    """Return the direction of the future return as a clean binary label by default."""

    _validate_horizon_and_column(df, horizon, price_column)

    forward_return = make_forward_return(
        df, horizon=horizon, price_column=price_column, return_type="simple"
    )
    missing_mask = forward_return.isna()
    name = f"target_sign_return_{horizon}h"
    if binary:
        target = (forward_return >= 0.0).astype("Int64").rename(name)
    else:
        target = (
            pd.Series(np.sign(forward_return), index=forward_return.index)
            .astype("Int64")
            .rename(name)
        )

    return target.mask(missing_mask)


__all__ = [
    "load_ethbtc_1h",
    "make_forward_return",
    "make_forward_return_sign",
    "DEFAULT_ETHBTC_1H_PATH",
]
