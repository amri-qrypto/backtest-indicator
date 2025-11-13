"""Adapters to convert pandas objects into QF-Lib containers."""
from __future__ import annotations

import pandas as pd
from qf_lib.containers.dataframe.qf_dataframe import QFDataFrame
from qf_lib.containers.series.qf_series import QFSeries


def _ensure_datetime_index(obj: pd.Series | pd.DataFrame) -> None:
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise TypeError("Input must be indexed by a DatetimeIndex")


def to_qfseries(series: pd.Series) -> QFSeries:
    """Convert a pandas Series into a ``QFSeries``."""
    _ensure_datetime_index(series)
    series = series.astype(float)
    return QFSeries(series)


def to_qfdataframe(df: pd.DataFrame) -> QFDataFrame:
    """Convert a pandas DataFrame into a ``QFDataFrame``."""
    _ensure_datetime_index(df)
    df = df.astype(float)
    return QFDataFrame(df)


__all__ = ["to_qfseries", "to_qfdataframe"]
