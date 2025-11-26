import pandas as pd

from src.data_loader import load_ohlcv_csv
from src.features import build_technical_features


def test_build_technical_features_emits_adx_block():
    df = load_ohlcv_csv("tests/fixtures/mini_ohlcv.csv")

    features = build_technical_features(df, adx_window=14)

    expected_columns = {"adx_14", "plus_di_14", "minus_di_14"}
    assert expected_columns.issubset(features.columns)

    adx_series = features["adx_14"].dropna()
    assert not adx_series.empty
    assert pd.api.types.is_numeric_dtype(adx_series)


def test_build_technical_features_requires_high_low():
    df = load_ohlcv_csv("tests/fixtures/mini_ohlcv.csv").drop(columns=["high"])

    try:
        build_technical_features(df)
    except KeyError as exc:
        assert "Missing required columns" in str(exc)
    else:  # pragma: no cover - defensive for unexpected behavior
        raise AssertionError("build_technical_features should validate high/low columns")
