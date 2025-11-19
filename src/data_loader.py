"""Utilities for loading OHLCV data from CSV files."""
from __future__ import annotations

from typing import Iterable, List, Sequence

import pandas as pd
from pandas.api import types as pd_types


REQUIRED_COLUMNS: List[str] = ["open", "high", "low", "close"]
COLUMN_ALIASES = {
    "volume": ["Volume", "VOL", "vol", "Base Volume", "base_volume"],
}
EMA_CANDIDATE_COLUMNS: Sequence[str] = ["EMA", "ema", "Ema", "ema_", "ema_close"]
DEFAULT_ADDITIONAL_COLUMNS: Sequence[str] = [
    "volume",
    "EMA",
    "Oversold HWO Up",
    "Overbought HWO Down",
    "HWO Up",
    "HWO Down",
    "ATR",
    "MACD",
    "Signal",
]


def _ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Ensure selected columns are numeric, coercing when needed."""
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_ohlcv_csv(path: str, additional_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Load OHLCV data from a CSV file and return a cleaned ``DataFrame``.

    Parameters
    ----------
    path:
        Location of the CSV file with OHLCV data.
    additional_columns:
        Extra columns that should be preserved if present in the dataset. By default, the
        loader keeps a curated list of indicator columns required by the strategy notebooks.
    """

    df = pd.read_csv(path)
    df = _standardize_column_aliases(df)

    if "time" not in df.columns:
        raise ValueError("CSV file must contain a 'time' column with UNIX timestamps.")

    time_col = df["time"]
    df["time"] = _normalize_time_column(time_col)
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV file missing required columns: {missing_columns}")

    additional: List[str] = list(DEFAULT_ADDITIONAL_COLUMNS)
    if additional_columns is not None:
        for column in additional_columns:
            if column not in additional:
                additional.append(column)

    ema_column = next((col for col in EMA_CANDIDATE_COLUMNS if col in df.columns), None)
    if ema_column is not None and ema_column != "EMA":
        df = df.rename(columns={ema_column: "EMA"})

    numeric_columns = list(dict.fromkeys([*REQUIRED_COLUMNS, *additional]))
    df = _ensure_numeric(df, numeric_columns)

    columns_to_keep = [col for col in numeric_columns if col in df.columns]
    cleaned = df[columns_to_keep].dropna(subset=REQUIRED_COLUMNS)

    return cleaned


def _standardize_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known column aliases to their canonical names."""

    rename_map = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
                break

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _normalize_time_column(series: pd.Series) -> pd.Series:
    """Return a timezone-aware datetime column from raw CSV data."""

    if pd_types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="s", utc=True)

    numeric_series = pd.to_numeric(series, errors="coerce")
    if numeric_series.notna().all():
        return pd.to_datetime(numeric_series, unit="s", utc=True)

    return pd.to_datetime(series, utc=True)


__all__ = ["load_ohlcv_csv"]
