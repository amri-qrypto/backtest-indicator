from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import load_ohlcv_csv


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_load_csv_parses_iso8601_time_column():
    df = load_ohlcv_csv(str(FIXTURES / "mini_ohlcv_iso.csv"))

    assert len(df) == 3
    assert df.index.tzinfo is not None
    assert df.index[0] == pd.Timestamp("2025-05-05 11:00:00+00:00")
    assert df.index.is_monotonic_increasing
    assert "volume" in df.columns
    assert df["volume"].iloc[0] == 10
