"""Utility helpers for loading and sanitising TradingView CSV exports."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


def sanitise_column_name(name: str) -> str:
    """Normalise a column name into snake_case for easier downstream access."""

    replacements = {"+": " plus ", "-": " minus ", "@": " at ", "%": " pct "}
    for old, new in replacements.items():
        name = name.replace(old, new)
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


def sanitise_columns(columns: Iterable[str]) -> Tuple[List[str], Dict[str, str]]:
    """Return sanitised column names and the mapping back to the originals."""

    seen: Dict[str, int] = {}
    sanitised: List[str] = []
    mapping: Dict[str, str] = {}

    for original in columns:
        base = sanitise_column_name(original)
        count = seen.get(base, 0)
        candidate = base if count == 0 else f"{base}_{count}"
        while candidate in seen:
            count += 1
            candidate = f"{base}_{count}"
        seen[candidate] = 1
        sanitised.append(candidate)
        mapping[candidate] = original

    return sanitised, mapping


def load_strategy_csv(
    csv_path: str | Path,
    time_column: str = "time",
    price_columns: Sequence[str] = ("open", "high", "low", "close"),
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load a TradingView CSV export with sanitised column names and DatetimeIndex."""

    path = Path(csv_path)
    if not path.exists():  # pragma: no cover - runtime validation in notebooks
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    sanitised, mapping = sanitise_columns(df.columns)
    df.columns = sanitised

    if time_column not in df.columns:
        raise KeyError(f"Time column '{time_column}' tidak ditemukan dalam file {path.name}")

    timestamps = pd.to_datetime(df[time_column], unit="s", utc=True, errors="coerce")
    if timestamps.isna().any():
        timestamps = pd.to_datetime(df[time_column], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Kolom waktu tidak bisa diparse menjadi datetime")

    datetime_index = pd.DatetimeIndex(timestamps)
    if datetime_index.tz is not None:
        datetime_index = datetime_index.tz_convert(None)

    df.index = datetime_index
    df = df.drop(columns=[time_column])

    for column in price_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df, mapping


def build_long_only_signals(
    long_condition: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Convert a boolean long condition into entry/exit boolean series."""

    condition = long_condition.fillna(False).astype(bool)
    prev = condition.shift(1).fillna(False)

    long_entry = (condition & ~prev).astype(bool)
    long_exit = ((~condition) & prev).astype(bool)
    index = condition.index
    short_entry = pd.Series(False, index=index, dtype=bool)
    short_exit = pd.Series(False, index=index, dtype=bool)
    position = condition.astype(int)

    return long_entry, long_exit, short_entry, short_exit, position


__all__ = [
    "sanitise_column_name",
    "sanitise_columns",
    "load_strategy_csv",
    "build_long_only_signals",
]
