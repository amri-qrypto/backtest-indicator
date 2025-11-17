"""EMA112 Adaptive Flip strategy ported from the provided TradingView script."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..base import StrategyBase, StrategyMetadata


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


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


@dataclass
class _PositionState:
    direction: Optional[str] = None
    entry_price: float = float("nan")


class Strategy(StrategyBase):
    """Replicates the EMA112 ATR auto-reentry strategy from TradingView."""

    metadata = StrategyMetadata(
        name="ema112_atr",
        description=(
            "Strategi breakout EMA112 dengan stop ATR dan opsi auto flip ketika stop tersentuh."
        ),
        entry=(
            "Long ketika harga menembus ke atas EMA112. Jika sedang short maka posisi ditutup sebelum entry baru. "
            "Short ketika harga menembus ke bawah EMA112 dengan logika serupa."
        ),
        exit=(
            "Stop loss dinamis berdasarkan ATR. Posisi long keluar jika harga penutupan turun di bawah entry - ATR*multiplier. "
            "Posisi short keluar jika harga penutupan naik di atas entry + ATR*multiplier. Jika auto reentry aktif maka stop akan flip ke arah berlawanan."
        ),
        parameters={
            "ema_length": 112,
            "atr_length": 14,
            "atr_multiplier": 1.2,
            "reentry_enabled": True,
        },
        context_columns=(
            "ema",
            "atr",
            "entry_price",
            "long_stop",
            "short_stop",
            "position",
            "long_signal",
            "short_signal",
        ),
    )

    def __init__(
        self,
        ema_length: int = 112,
        atr_length: int = 14,
        atr_multiplier: float = 1.2,
        reentry_enabled: bool = True,
    ) -> None:
        super().__init__(
            ema_length=ema_length,
            atr_length=atr_length,
            atr_multiplier=atr_multiplier,
            reentry_enabled=reentry_enabled,
        )
        self.ema_length = ema_length
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self.reentry_enabled = reentry_enabled

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"open", "high", "low", "close"}
        missing = required_cols - set(data.columns)
        if missing:
            raise KeyError(f"DataFrame tidak memiliki kolom harga: {sorted(missing)}")

        close = data["close"].astype(float)
        ema = _ema(close, self.ema_length)
        atr = _atr(data, self.atr_length)

        prev_close = close.shift(1)
        prev_ema = ema.shift(1)
        long_signal = ((close > ema) & (prev_close <= prev_ema)).fillna(False)
        short_signal = ((close < ema) & (prev_close >= prev_ema)).fillna(False)

        index = data.index
        long_entry = pd.Series(False, index=index)
        long_exit = pd.Series(False, index=index)
        short_entry = pd.Series(False, index=index)
        short_exit = pd.Series(False, index=index)

        entry_price_series = pd.Series(float("nan"), index=index)
        long_stop_series = pd.Series(float("nan"), index=index)
        short_stop_series = pd.Series(float("nan"), index=index)
        position_series = pd.Series("flat", index=index, dtype="object")

        state = _PositionState()

        for ts in index:
            price = float(close.loc[ts])
            atr_value = float(atr.loc[ts]) if pd.notna(atr.loc[ts]) else float("nan")
            long_sig = bool(long_signal.loc[ts])
            short_sig = bool(short_signal.loc[ts])

            # Entry logic (handles close + entry before stop check just like TradingView script)
            if long_sig and state.direction != "long":
                if state.direction == "short":
                    short_exit.loc[ts] = True
                state.direction = "long"
                state.entry_price = price
                long_entry.loc[ts] = True
            elif short_sig and state.direction != "short":
                if state.direction == "long":
                    long_exit.loc[ts] = True
                state.direction = "short"
                state.entry_price = price
                short_entry.loc[ts] = True

            long_stop = (
                state.entry_price - self.atr_multiplier * atr_value
                if state.direction == "long" and np.isfinite(atr_value)
                else np.nan
            )
            short_stop = (
                state.entry_price + self.atr_multiplier * atr_value
                if state.direction == "short" and np.isfinite(atr_value)
                else np.nan
            )

            long_stop_series.loc[ts] = long_stop if state.direction == "long" else np.nan
            short_stop_series.loc[ts] = short_stop if state.direction == "short" else np.nan
            entry_price_series.loc[ts] = state.entry_price if state.direction else np.nan
            position_series.loc[ts] = state.direction or "flat"

            # Exit logic with optional flip
            if state.direction == "long" and np.isfinite(long_stop) and price < long_stop:
                long_exit.loc[ts] = True
                if self.reentry_enabled and np.isfinite(atr_value):
                    state.direction = "short"
                    state.entry_price = price
                    short_entry.loc[ts] = True
                    short_stop_series.loc[ts] = (
                        state.entry_price + self.atr_multiplier * atr_value
                    )
                    position_series.loc[ts] = state.direction
                    entry_price_series.loc[ts] = state.entry_price
                else:
                    state.direction = None
                    state.entry_price = float("nan")
                    entry_price_series.loc[ts] = np.nan
                    position_series.loc[ts] = "flat"
                    short_stop_series.loc[ts] = np.nan
            elif state.direction == "short" and np.isfinite(short_stop) and price > short_stop:
                short_exit.loc[ts] = True
                if self.reentry_enabled and np.isfinite(atr_value):
                    state.direction = "long"
                    state.entry_price = price
                    long_entry.loc[ts] = True
                    long_stop_series.loc[ts] = (
                        state.entry_price - self.atr_multiplier * atr_value
                    )
                    position_series.loc[ts] = state.direction
                    entry_price_series.loc[ts] = state.entry_price
                else:
                    state.direction = None
                    state.entry_price = float("nan")
                    entry_price_series.loc[ts] = np.nan
                    position_series.loc[ts] = "flat"
                    long_stop_series.loc[ts] = np.nan

        signals = pd.DataFrame(
            {
                "long_entry": long_entry,
                "long_exit": long_exit,
                "short_entry": short_entry,
                "short_exit": short_exit,
                "ema": ema,
                "atr": atr,
                "entry_price": entry_price_series,
                "long_stop": long_stop_series,
                "short_stop": short_stop_series,
                "position": position_series,
                "long_signal": long_signal,
                "short_signal": short_signal,
            },
            index=index,
        )

        return signals


__all__ = ["Strategy"]
