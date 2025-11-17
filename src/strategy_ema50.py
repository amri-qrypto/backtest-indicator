"""EMA(50) trend-following strategy."""
from __future__ import annotations

import pandas as pd

from .strategy_backtest.base import StrategyBase, StrategyMetadata
from .strategy_backtest.utils import build_long_only_signals
from .strategy_ema import ema_trend_context, ema_trend_signals


def generate_signals(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """Return long/cash positions based on a 50-period EMA."""

    return ema_trend_signals(df, span=50, price_col=price_col)


class Strategy(StrategyBase):
    """Long-only EMA(50) trend strategy."""

    metadata = StrategyMetadata(
        name="ema50",
        description="Strategi trend following yang menggunakan EMA50 sebagai batas bullish/bearish.",
        entry="Entry long ketika penutupan berada di atas EMA50.",
        exit="Keluar ketika penutupan jatuh di bawah EMA50.",
        parameters={"ema_length": 50},
        context_columns=("price", "ema", "position"),
    )

    def __init__(self, ema_length: int = 50, price_column: str = "close") -> None:
        super().__init__(ema_length=ema_length, price_column=price_column)
        self.ema_length = ema_length
        self.price_column = price_column

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        price, ema, long_condition = ema_trend_context(
            data, span=self.ema_length, price_col=self.price_column
        )

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
            index=data.index,
        )


__all__ = ["generate_signals", "Strategy"]

