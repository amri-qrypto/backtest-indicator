"""EMA(112) trend-following strategy."""
from __future__ import annotations

import pandas as pd

from .strategy_backtest.base import StrategyBase, StrategyMetadata
from .strategy_backtest.utils import build_long_only_signals
from .strategy_ema import ema_trend_context, ema_trend_signals


def generate_signals(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """Return long/cash positions based on a 112-period EMA."""

    return ema_trend_signals(df, span=112, price_col=price_col)


class Strategy(StrategyBase):
    """Long-only EMA(112) trend following strategy."""

    metadata = StrategyMetadata(
        name="ema112",
        description=(
            "Strategi trend following sederhana yang membeli saat harga berada di atas EMA112 "
            "dan kembali ke kas ketika harga turun di bawah EMA.")
        ,
        entry="Entry long ketika penutupan menembus ke atas EMA112.",
        exit="Keluar dari posisi saat penutupan berada di bawah EMA112.",
        parameters={"ema_length": 112},
        context_columns=("price", "ema", "position"),
    )

    def __init__(self, ema_length: int = 112, price_column: str = "close") -> None:
        super().__init__(ema_length=ema_length, price_column=price_column)
        self.ema_length = ema_length
        self.price_column = price_column

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        price, ema, long_condition = ema_trend_context(
            data, span=self.ema_length, price_col=self.price_column
        )

        index = data.index
        long_entry, long_exit, short_entry, short_exit, position = build_long_only_signals(
            long_condition
        )

        return pd.DataFrame(
            {
                "long_entry": long_entry,
                "long_exit": long_exit,
                "short_entry": short_entry,
                "short_exit": short_exit,
                "price": price,
                "ema": ema,
                "position": position,
            },
            index=index,
        )


__all__ = ["generate_signals", "Strategy"]

