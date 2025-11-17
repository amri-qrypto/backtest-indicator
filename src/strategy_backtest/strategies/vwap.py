"""VWAP mean-reversion strategy following the user's TradingView logic."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..base import StrategyBase, StrategyMetadata


def _session_vwap(df: pd.DataFrame, freq: str) -> pd.Series:
    index = df.index
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    if "volume" not in df.columns:
        raise KeyError("DataFrame tidak memiliki kolom volume untuk perhitungan VWAP")
    volume = df["volume"].astype(float)
    typical_price = (high + low + close) / 3.0

    if isinstance(index, pd.DatetimeIndex):
        groups = pd.Series(index.floor(freq), index=index)
    else:
        groups = pd.Series(0, index=index)

    tpv = typical_price * volume
    cum_tpv = tpv.groupby(groups).cumsum()
    cum_vol = volume.groupby(groups).cumsum().replace(0.0, np.nan)
    vwap = cum_tpv / cum_vol
    return vwap.ffill()


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


class Strategy(StrategyBase):
    """VWAP mean-reversion dengan filter RSI dan stop ATR."""

    metadata = StrategyMetadata(
        name="vwap",
        description=(
            "Strategi VWAP mean reversion: beli ketika harga di bawah VWAP dengan RSI rebound bullish, "
            "jual ketika harga di atas VWAP dengan RSI bearish dari area jenuh beli."
        ),
        entry=(
            "Long: harga penutupan di bawah VWAP, RSI sebelumnya berada di bawah level oversold lalu menembus ke atas level 50. "
            "Short: harga penutupan di atas VWAP, RSI sebelumnya berada di atas level overbought lalu menembus ke bawah level 50."
        ),
        exit=(
            "Stop loss berbasis ATR yang dieksekusi pada penembusan high/low harian ditambah trailing stop opsional."
        ),
        parameters={
            "rsi_length": 14,
            "rsi_overbought": 60,
            "rsi_oversold": 40,
            "atr_length": 14,
            "atr_stop_multiplier": 1.5,
            "session_frequency": "1D",
            "trail_activation_multiple": 1.0,
            "trail_atr_multiplier": 1.0,
        },
        context_columns=(
            "vwap",
            "rsi",
            "atr",
            "active_entry_price",
            "stop_level",
            "position",
            "best_price",
            "trail_active",
        ),
    )

    def __init__(
        self,
        rsi_length: int = 14,
        rsi_overbought: int = 60,
        rsi_oversold: int = 40,
        atr_length: int = 14,
        atr_stop_multiplier: float = 1.5,
        session_frequency: str = "1D",
        trail_activation_multiple: float = 1.0,
        trail_atr_multiplier: float = 1.0,
    ) -> None:
        super().__init__(
            rsi_length=rsi_length,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            atr_length=atr_length,
            atr_stop_multiplier=atr_stop_multiplier,
            session_frequency=session_frequency,
            trail_activation_multiple=trail_activation_multiple,
            trail_atr_multiplier=trail_atr_multiplier,
        )
        self.rsi_length = rsi_length
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.atr_length = atr_length
        self.atr_stop_multiplier = atr_stop_multiplier
        self.session_frequency = session_frequency
        self.trail_activation_multiple = trail_activation_multiple
        self.trail_atr_multiplier = trail_atr_multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(data.columns)
        if missing:
            raise KeyError(f"DataFrame tidak memiliki kolom harga/volume: {sorted(missing)}")

        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        vwap = _session_vwap(data, self.session_frequency)
        rsi = _rsi(close, self.rsi_length)
        atr = _atr(data, self.atr_length)

        long_condition = pd.Series(False, index=data.index)
        short_condition = pd.Series(False, index=data.index)

        prev_rsi = float("nan")
        long_ready = False
        short_ready = False
        for ts in data.index:
            rsi_value = float(rsi.loc[ts]) if pd.notna(rsi.loc[ts]) else float("nan")
            close_value = float(close.loc[ts]) if pd.notna(close.loc[ts]) else float("nan")
            vwap_value = float(vwap.loc[ts]) if pd.notna(vwap.loc[ts]) else float("nan")

            rsi_cross_up = (
                np.isfinite(prev_rsi)
                and np.isfinite(rsi_value)
                and prev_rsi <= 50
                and rsi_value > 50
            )
            rsi_cross_down = (
                np.isfinite(prev_rsi)
                and np.isfinite(rsi_value)
                and prev_rsi >= 50
                and rsi_value < 50
            )

            if np.isfinite(rsi_value):
                if rsi_value <= self.rsi_oversold:
                    long_ready = True
                if rsi_value >= self.rsi_overbought:
                    short_ready = True

            price_below_vwap = (
                np.isfinite(close_value)
                and np.isfinite(vwap_value)
                and close_value < vwap_value
            )
            price_above_vwap = (
                np.isfinite(close_value)
                and np.isfinite(vwap_value)
                and close_value > vwap_value
            )

            if long_ready and rsi_cross_up and price_below_vwap:
                long_condition.loc[ts] = True
            if short_ready and rsi_cross_down and price_above_vwap:
                short_condition.loc[ts] = True

            if rsi_cross_up:
                long_ready = False
            if rsi_cross_down:
                short_ready = False

            if np.isfinite(rsi_value):
                prev_rsi = rsi_value

        index = data.index
        long_entry = pd.Series(False, index=index)
        long_exit = pd.Series(False, index=index)
        short_entry = pd.Series(False, index=index)
        short_exit = pd.Series(False, index=index)

        active_entry_price = pd.Series(float("nan"), index=index)
        stop_level = pd.Series(float("nan"), index=index)
        position_context = pd.Series("flat", index=index, dtype="object")
        best_price_context = pd.Series(float("nan"), index=index)
        trail_active_context = pd.Series(False, index=index)

        position: Optional[str] = None
        current_entry = float("nan")
        current_stop = float("nan")
        entry_atr_value = float("nan")
        position_best_price = float("nan")
        trail_active_state = False

        for ts in index:
            price = float(close.loc[ts])
            atr_value = float(atr.loc[ts]) if pd.notna(atr.loc[ts]) else float("nan")
            bar_high = float(high.loc[ts])
            bar_low = float(low.loc[ts])

            if position == "long":
                position_best_price = (
                    max(position_best_price, bar_high)
                    if np.isfinite(position_best_price)
                    else bar_high
                )
                if (
                    np.isfinite(self.trail_activation_multiple)
                    and np.isfinite(entry_atr_value)
                    and entry_atr_value > 0
                    and np.isfinite(atr_value)
                    and atr_value > 0
                ):
                    mfe = position_best_price - current_entry
                    if mfe >= self.trail_activation_multiple * entry_atr_value:
                        trail_active_state = True
                        trail_stop = position_best_price - self.trail_atr_multiplier * atr_value
                        if np.isfinite(trail_stop):
                            current_stop = max(current_stop, trail_stop)
                if bar_low <= current_stop:
                    long_exit.loc[ts] = True
                    position = None
            elif position == "short":
                position_best_price = (
                    min(position_best_price, bar_low)
                    if np.isfinite(position_best_price)
                    else bar_low
                )
                if (
                    np.isfinite(self.trail_activation_multiple)
                    and np.isfinite(entry_atr_value)
                    and entry_atr_value > 0
                    and np.isfinite(atr_value)
                    and atr_value > 0
                ):
                    mfe = current_entry - position_best_price
                    if mfe >= self.trail_activation_multiple * entry_atr_value:
                        trail_active_state = True
                        trail_stop = position_best_price + self.trail_atr_multiplier * atr_value
                        if np.isfinite(trail_stop):
                            current_stop = min(current_stop, trail_stop)
                if bar_high >= current_stop:
                    short_exit.loc[ts] = True
                    position = None

            if position is None:
                current_entry = float("nan")
                current_stop = float("nan")
                position_best_price = float("nan")
                entry_atr_value = float("nan")
                trail_active_state = False

                if bool(long_condition.loc[ts]) and np.isfinite(atr_value) and atr_value > 0:
                    long_entry.loc[ts] = True
                    position = "long"
                    current_entry = price
                    current_stop = current_entry - self.atr_stop_multiplier * atr_value
                    entry_atr_value = atr_value
                    position_best_price = bar_high
                elif bool(short_condition.loc[ts]) and np.isfinite(atr_value) and atr_value > 0:
                    short_entry.loc[ts] = True
                    position = "short"
                    current_entry = price
                    current_stop = current_entry + self.atr_stop_multiplier * atr_value
                    entry_atr_value = atr_value
                    position_best_price = bar_low

            else:
                # handle potential flip if opposite signal appears
                if position == "long" and bool(short_condition.loc[ts]) and np.isfinite(atr_value) and atr_value > 0:
                    long_exit.loc[ts] = True
                    short_entry.loc[ts] = True
                    position = "short"
                    current_entry = price
                    current_stop = current_entry + self.atr_stop_multiplier * atr_value
                    entry_atr_value = atr_value
                    position_best_price = bar_low
                    trail_active_state = False
                elif position == "short" and bool(long_condition.loc[ts]) and np.isfinite(atr_value) and atr_value > 0:
                    short_exit.loc[ts] = True
                    long_entry.loc[ts] = True
                    position = "long"
                    current_entry = price
                    current_stop = current_entry - self.atr_stop_multiplier * atr_value
                    entry_atr_value = atr_value
                    position_best_price = bar_high
                    trail_active_state = False

            active_entry_price.loc[ts] = current_entry
            stop_level.loc[ts] = current_stop
            position_context.loc[ts] = position or "flat"
            best_price_context.loc[ts] = position_best_price
            trail_active_context.loc[ts] = trail_active_state

        context = pd.DataFrame(
            {
                "vwap": vwap,
                "rsi": rsi,
                "atr": atr,
                "active_entry_price": active_entry_price,
                "stop_level": stop_level,
                "position": position_context,
                "best_price": best_price_context,
                "trail_active": trail_active_context,
            },
            index=index,
        )

        signals = pd.DataFrame(
            {
                "long_entry": long_entry,
                "long_exit": long_exit,
                "short_entry": short_entry,
                "short_exit": short_exit,
            },
            index=index,
        )

        return pd.concat([signals, context], axis=1)


__all__ = ["Strategy"]
