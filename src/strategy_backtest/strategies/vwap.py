"""VWAP mean-reversion strategy following the user's TradingView logic."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..base import StrategyBase, StrategyMetadata


def _normalize_frequency(freq: str) -> str:
    """Return a pandas-friendly lowercase frequency string."""

    if not isinstance(freq, str):
        raise TypeError("Session frequency harus berupa string.")

    normalized = freq.strip().lower()
    if not normalized:
        raise ValueError("Session frequency tidak boleh kosong.")

    # Validasi agar pandas dapat mengenali freq tersebut.
    try:
        pd.tseries.frequencies.to_offset(normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Session frequency '{freq}' tidak valid.") from exc

    return normalized


def _session_vwap(df: pd.DataFrame, freq: str) -> pd.Series:
    freq = _normalize_frequency(freq)
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
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean().replace(0.0, 1e-12)
    rs = avg_gain / avg_loss
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
            "Strategi VWAP mean reversion: beli ketika harga di bawah VWAP dengan RSI bullish, "
            "jual ketika harga di atas VWAP dengan RSI bearish."
        ),
        entry=(
            "Long: harga penutupan di bawah VWAP, RSI di atas level oversold, dan RSI menembus ke atas level 50. "
            "Short: harga penutupan di atas VWAP, RSI di bawah level overbought, dan RSI menembus ke bawah level 50."
        ),
        exit=(
            "Stop loss berbasis ATR: posisi long keluar jika harga menyentuh entry - ATR*multiplier, "
            "posisi short keluar jika harga menyentuh entry + ATR*multiplier. Target profit opsional dapat diaktifkan "
            "dengan multiplier ATR yang sama untuk memastikan rasio risk-reward konsisten."
        ),
        parameters={
            "rsi_length": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "atr_length": 14,
            "atr_stop_multiplier": 1.5,
            "atr_take_profit_multiplier": None,
            "session_frequency": "1D",
        },
        context_columns=(
            "vwap",
            "rsi",
            "atr",
            "active_entry_price",
            "stop_level",
            "take_profit_level",
            "position",
        ),
    )

    def __init__(
        self,
        rsi_length: int = 14,
        rsi_overbought: int = 70,
        rsi_oversold: int = 30,
        atr_length: int = 14,
        atr_stop_multiplier: float = 1.5,
        atr_take_profit_multiplier: float | None = None,
        session_frequency: str = "1D",
    ) -> None:
        normalized_frequency = _normalize_frequency(session_frequency)

        super().__init__(
            rsi_length=rsi_length,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            atr_length=atr_length,
            atr_stop_multiplier=atr_stop_multiplier,
            atr_take_profit_multiplier=atr_take_profit_multiplier,
            session_frequency=normalized_frequency,
        )
        self.rsi_length = rsi_length
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.atr_length = atr_length
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_take_profit_multiplier = atr_take_profit_multiplier
        self.session_frequency = normalized_frequency

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

        rsi_cross_up = (rsi > 50) & (rsi.shift(1) <= 50)
        rsi_cross_down = (rsi < 50) & (rsi.shift(1) >= 50)

        long_condition = (
            (close < vwap)
            & (rsi > self.rsi_oversold)
            & rsi_cross_up
        ).fillna(False)
        short_condition = (
            (close > vwap)
            & (rsi < self.rsi_overbought)
            & rsi_cross_down
        ).fillna(False)

        index = data.index
        long_entry = pd.Series(False, index=index)
        long_exit = pd.Series(False, index=index)
        short_entry = pd.Series(False, index=index)
        short_exit = pd.Series(False, index=index)

        active_entry_price = pd.Series(float("nan"), index=index)
        stop_level = pd.Series(float("nan"), index=index)
        take_profit_level = pd.Series(float("nan"), index=index)
        position_context = pd.Series("flat", index=index, dtype="object")

        position: Optional[str] = None
        current_entry = float("nan")
        current_stop = float("nan")
        current_take_profit = float("nan")

        for ts in index:
            price = float(close.loc[ts])
            bar_high = float(high.loc[ts])
            bar_low = float(low.loc[ts])
            atr_value = float(atr.loc[ts]) if pd.notna(atr.loc[ts]) else float("nan")

            active_entry_price.loc[ts] = current_entry
            stop_level.loc[ts] = current_stop
            take_profit_level.loc[ts] = current_take_profit
            position_context.loc[ts] = position or "flat"

            if position == "long":
                if np.isfinite(current_stop) and bar_low <= current_stop:
                    long_exit.loc[ts] = True
                    position = None
                elif (
                    self.atr_take_profit_multiplier is not None
                    and np.isfinite(current_take_profit)
                    and bar_high >= current_take_profit
                ):
                    long_exit.loc[ts] = True
                    position = None
            elif position == "short":
                if np.isfinite(current_stop) and bar_high >= current_stop:
                    short_exit.loc[ts] = True
                    position = None
                elif (
                    self.atr_take_profit_multiplier is not None
                    and np.isfinite(current_take_profit)
                    and bar_low <= current_take_profit
                ):
                    short_exit.loc[ts] = True
                    position = None

            if position is None:
                current_entry = float("nan")
                current_stop = float("nan")
                current_take_profit = float("nan")

                if bool(long_condition.loc[ts]) and np.isfinite(atr_value) and atr_value > 0:
                    long_entry.loc[ts] = True
                    position = "long"
                    current_entry = price
                    current_stop = current_entry - self.atr_stop_multiplier * atr_value
                    if self.atr_take_profit_multiplier is not None:
                        current_take_profit = current_entry + self.atr_take_profit_multiplier * atr_value
                elif bool(short_condition.loc[ts]) and np.isfinite(atr_value) and atr_value > 0:
                    short_entry.loc[ts] = True
                    position = "short"
                    current_entry = price
                    current_stop = current_entry + self.atr_stop_multiplier * atr_value
                    if self.atr_take_profit_multiplier is not None:
                        current_take_profit = current_entry - self.atr_take_profit_multiplier * atr_value

                active_entry_price.loc[ts] = current_entry
                stop_level.loc[ts] = current_stop
                take_profit_level.loc[ts] = current_take_profit
                position_context.loc[ts] = position or "flat"
            else:
                # handle potential flip if opposite signal appears
                if position == "long" and bool(short_condition.loc[ts]) and np.isfinite(atr_value) and atr_value > 0:
                    long_exit.loc[ts] = True
                    short_entry.loc[ts] = True
                    position = "short"
                    current_entry = price
                    current_stop = current_entry + self.atr_stop_multiplier * atr_value
                    current_take_profit = (
                        current_entry - self.atr_take_profit_multiplier * atr_value
                        if self.atr_take_profit_multiplier is not None
                        else float("nan")
                    )
                elif position == "short" and bool(long_condition.loc[ts]) and np.isfinite(atr_value) and atr_value > 0:
                    short_exit.loc[ts] = True
                    long_entry.loc[ts] = True
                    position = "long"
                    current_entry = price
                    current_stop = current_entry - self.atr_stop_multiplier * atr_value
                    current_take_profit = (
                        current_entry + self.atr_take_profit_multiplier * atr_value
                        if self.atr_take_profit_multiplier is not None
                        else float("nan")
                    )

                active_entry_price.loc[ts] = current_entry
                stop_level.loc[ts] = current_stop
                take_profit_level.loc[ts] = current_take_profit
                position_context.loc[ts] = position or "flat"

        context = pd.DataFrame(
            {
                "vwap": vwap,
                "rsi": rsi,
                "atr": atr,
                "active_entry_price": active_entry_price,
                "stop_level": stop_level,
                "take_profit_level": take_profit_level,
                "position": position_context,
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
