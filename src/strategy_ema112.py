"""EMA(112) trend-following strategy with optional ATR stop/take-profit."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy_backtest.base import StrategyBase, StrategyMetadata
from .strategy_ema import ema_trend_context, ema_trend_signals


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Compute an ATR-style volatility estimate for stop sizing."""

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


def generate_signals(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """Return long/cash positions based on a 112-period EMA."""

    return ema_trend_signals(df, span=112, price_col=price_col)


class Strategy(StrategyBase):
    """Long-only EMA(112) trend following strategy."""

    metadata = StrategyMetadata(
        name="ema112",
        description=(
            "Strategi trend following sederhana yang membeli saat harga berada di atas EMA112 "
            "dan kembali ke kas ketika harga turun di bawah EMA."
        ),
        entry=(
            "Entry long ketika penutupan menembus ke atas EMA112. Stop loss/take profit ATR "
            "opsional dihitung dari harga entry untuk menjaga konsistensi risk-reward."
        ),
        exit=(
            "Keluar dari posisi saat penutupan menembus ke bawah EMA112 atau jika harga menyentuh "
            "level stop loss / take profit ATR yang telah dihitung dari harga entry."
        ),
        parameters={
            "ema_length": 112,
            "atr_length": 14,
            "atr_stop_multiplier": 1.2,
            "atr_take_profit_multiplier": 2.0,
        },
        context_columns=(
            "price",
            "ema",
            "atr",
            "entry_price",
            "stop_level",
            "take_profit_level",
            "position",
        ),
    )

    def __init__(
        self,
        ema_length: int = 112,
        price_column: str = "close",
        atr_length: int = 14,
        atr_stop_multiplier: float | None = 1.2,
        atr_take_profit_multiplier: float | None = 2.0,
    ) -> None:
        super().__init__(
            ema_length=ema_length,
            price_column=price_column,
            atr_length=atr_length,
            atr_stop_multiplier=atr_stop_multiplier,
            atr_take_profit_multiplier=atr_take_profit_multiplier,
        )
        self.ema_length = ema_length
        self.price_column = price_column
        self.atr_length = atr_length
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_take_profit_multiplier = atr_take_profit_multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"open", "high", "low", "close"}
        missing = required_cols - set(data.columns)
        if missing:
            raise KeyError(f"DataFrame tidak memiliki kolom harga: {sorted(missing)}")

        price, ema, long_condition = ema_trend_context(
            data, span=self.ema_length, price_col=self.price_column
        )
        atr = (
            _atr(data, self.atr_length)
            if self.atr_stop_multiplier or self.atr_take_profit_multiplier
            else pd.Series(float("nan"), index=data.index)
        )

        index = data.index
        long_entry = pd.Series(False, index=index)
        long_exit = pd.Series(False, index=index)
        short_entry = pd.Series(False, index=index)
        short_exit = pd.Series(False, index=index)

        entry_price = pd.Series(float("nan"), index=index)
        stop_level = pd.Series(float("nan"), index=index)
        take_profit_level = pd.Series(float("nan"), index=index)
        position = pd.Series(0, index=index, dtype=int)

        in_position = False
        current_entry = float("nan")
        current_stop = float("nan")
        current_take_profit = float("nan")

        crosses_up = (long_condition & ~long_condition.shift(1, fill_value=False)).astype(bool)
        crosses_down = ((~long_condition) & long_condition.shift(1, fill_value=False)).astype(bool)

        for ts in index:
            price_bar = float(price.loc[ts])
            bar_high = float(data["high"].loc[ts])
            bar_low = float(data["low"].loc[ts])
            atr_value = float(atr.loc[ts]) if pd.notna(atr.loc[ts]) else float("nan")

            if in_position:
                if np.isfinite(current_stop) and bar_low <= current_stop:
                    long_exit.loc[ts] = True
                    in_position = False
                elif np.isfinite(current_take_profit) and bar_high >= current_take_profit:
                    long_exit.loc[ts] = True
                    in_position = False
                elif crosses_down.loc[ts]:
                    long_exit.loc[ts] = True
                    in_position = False

            if not in_position and crosses_up.loc[ts]:
                long_entry.loc[ts] = True
                in_position = True
                current_entry = price_bar
                if self.atr_stop_multiplier and np.isfinite(atr_value):
                    current_stop = current_entry - self.atr_stop_multiplier * atr_value
                else:
                    current_stop = float("nan")

                if self.atr_take_profit_multiplier and np.isfinite(atr_value):
                    current_take_profit = current_entry + self.atr_take_profit_multiplier * atr_value
                else:
                    current_take_profit = float("nan")

            if not in_position:
                current_entry = float("nan")
                current_stop = float("nan")
                current_take_profit = float("nan")

            entry_price.loc[ts] = current_entry
            stop_level.loc[ts] = current_stop
            take_profit_level.loc[ts] = current_take_profit
            position.loc[ts] = 1 if in_position else 0

        return pd.DataFrame(
            {
                "long_entry": long_entry,
                "long_exit": long_exit,
                "short_entry": short_entry,
                "short_exit": short_exit,
                "price": price,
                "ema": ema,
                "atr": atr,
                "entry_price": entry_price,
                "stop_level": stop_level,
                "take_profit_level": take_profit_level,
                "position": position,
            },
            index=index,
        )


__all__ = ["generate_signals", "Strategy"]
