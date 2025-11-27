from __future__ import annotations

import pandas as pd

from src.strategy_ema112 import Strategy as EMA112Strategy
from src.strategy_backtest.strategies.vwap import Strategy as VWAPStrategy


def _build_ohlcv(index: pd.DatetimeIndex, prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
        },
        index=index,
    )


def test_ema112_handles_low_volatility_trend():
    index = pd.date_range("2023-01-01", periods=5, freq="H")
    prices = [100, 101, 101.5, 102, 102.5]
    df = _build_ohlcv(index, prices)

    strategy = EMA112Strategy(
        ema_length=2,
        atr_length=2,
        atr_stop_multiplier=1.0,
        atr_take_profit_multiplier=5.0,
    )
    signals = strategy.generate_signals(df)

    assert signals["long_entry"].sum() == 1
    assert signals["long_exit"].sum() == 0
    assert signals["position"].iloc[-1] == 1
    assert signals.loc[signals["position"] == 1, "stop_level"].notna().any()


def test_ema112_triggers_stop_on_gap_down():
    index = pd.date_range("2023-01-01", periods=4, freq="H")
    closes = [10.0, 9.0, 9.4, 8.5]
    df = _build_ohlcv(index, closes)

    strategy = EMA112Strategy(
        ema_length=2,
        atr_length=1,
        atr_stop_multiplier=1.0,
        atr_take_profit_multiplier=None,
    )
    signals = strategy.generate_signals(df)

    assert bool(signals["long_entry"].iloc[2])
    assert bool(signals["long_exit"].iloc[3])
    assert signals["position"].iloc[-1] == 0


def test_vwap_resets_by_session_frequency():
    index = pd.date_range("2023-01-01", periods=4, freq="12H")
    prices = [10.0, 11.0, 9.0, 10.0]
    df = _build_ohlcv(index, prices)
    df["volume"] = 100

    strategy = VWAPStrategy(
        rsi_length=2,
        rsi_overbought=100,
        rsi_oversold=0,
        atr_length=1,
        atr_stop_multiplier=1.0,
        atr_take_profit_multiplier=2.0,
        session_frequency="1D",
    )
    signals = strategy.generate_signals(df)

    # Session restart on the 3rd bar (new day) uses only that bar's price/volume.
    assert signals.loc[index[2], "vwap"] == prices[2]
    assert signals.loc[index[0], "vwap"] == prices[0]


def test_vwap_stop_levels_handle_gap_moves():
    index = pd.date_range("2023-01-01", periods=4, freq="H")
    closes = [10.0, 9.0, 9.4, 8.5]
    df = _build_ohlcv(index, closes)
    df["volume"] = 100

    strategy = VWAPStrategy(
        rsi_length=1,
        rsi_overbought=100,
        rsi_oversold=0,
        atr_length=1,
        atr_stop_multiplier=1.0,
        atr_take_profit_multiplier=None,
        session_frequency="1D",
    )
    signals = strategy.generate_signals(df)

    assert bool(signals["long_entry"].iloc[2])
    assert bool(signals["long_exit"].iloc[3])
    # Stop level is placed between entry and the following bar's low, triggering the protective exit.
    assert signals.loc[index[2], "stop_level"] > df.loc[index[3], "low"]
