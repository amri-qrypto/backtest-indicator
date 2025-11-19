"""Unit tests covering custom performance helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.targets import make_forward_return_sign
from src.performance.metrics import summarise_bar_returns


def test_summarise_bar_returns_matches_excel_formula():
    index = pd.date_range("2024-01-01", periods=10, freq="H")
    returns = pd.Series(
        [0.01, -0.02, 0.005, 0.0, 0.01, -0.005, 0.002, 0.003, -0.004, 0.006],
        index=index,
    )
    bars_per_year = 24 * 365

    stats = summarise_bar_returns(returns, bars_per_year=bars_per_year)

    mean_r = returns.mean()
    std_r = returns.std(ddof=0)
    annualised_vol = std_r * np.sqrt(bars_per_year)
    annualised_ret = mean_r * bars_per_year
    expected_sharpe = annualised_ret / annualised_vol

    equity = (1.0 + returns).cumprod()
    days_between = (index[-1] - index[0]).total_seconds() / (24 * 60 * 60)
    expected_cagr = equity.iloc[-1] ** (365.0 / days_between) - 1.0

    assert np.isclose(stats["annualised_vol"], annualised_vol)
    assert np.isclose(stats["sharpe_ratio"], expected_sharpe)
    assert np.isclose(stats["cagr"], expected_cagr)


def test_make_forward_return_sign_is_binary_by_default():
    index = pd.date_range("2024-01-01", periods=6, freq="H")
    prices = pd.Series([100, 101, 100.5, 100.5, 99.5, 100.0], index=index)
    df = pd.DataFrame({"close": prices})

    binary = make_forward_return_sign(df, horizon=1)
    ternary = make_forward_return_sign(df, horizon=1, binary=False)

    assert set(binary.dropna().unique()) <= {0, 1}
    assert {0, 1}.issubset(set(binary.dropna().unique()))
    assert set(ternary.dropna().unique()) <= {-1, 0, 1}
