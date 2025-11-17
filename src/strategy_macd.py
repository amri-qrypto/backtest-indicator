"""Moving Average Convergence Divergence (MACD) strategy helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy_backtest.base import StrategyBase, StrategyMetadata
from .strategy_backtest.utils import build_long_only_signals


def macd_components(
    close: pd.Series,
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
) -> pd.DataFrame:
    """Return MACD line, signal line, and histogram values."""
    if close.empty:
        raise ValueError("close series must not be empty")

    fast_ema = close.ewm(span=fast_span, adjust=False).mean()
    slow_ema = close.ewm(span=slow_span, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
    histogram = macd_line - signal_line

    data = pd.DataFrame(
        {
            "macd": macd_line.astype(float),
            "macd_signal": signal_line.astype(float),
            "macd_hist": histogram.astype(float),
        },
        index=close.index,
    )
    return data


def macd_signals(
    df: pd.DataFrame,
    price_col: str = "close",
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
) -> pd.Series:
    """Generate long/cash positions based on MACD and signal line crossovers."""
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain a '{price_col}' column")

    macd_df = macd_components(df[price_col].astype(float), fast_span, slow_span, signal_span)

    comparison = np.where(
        macd_df["macd"] > macd_df["macd_signal"],
        1.0,
        np.where(macd_df["macd"] < macd_df["macd_signal"], 0.0, np.nan),
    )
    positions = pd.Series(comparison, index=macd_df.index, dtype=float)
    positions = positions.ffill().fillna(0.0)
    positions.name = "position"

    return positions


class Strategy(StrategyBase):
    """MACD crossover strategy compatible with the signal backtester."""

    metadata = StrategyMetadata(
        name="macd",
        description="Membeli ketika garis MACD berada di atas garis sinyal dan kembali kas sebaliknya.",
        entry="Entry long ketika MACD memotong ke atas garis sinyal.",
        exit="Keluar saat MACD turun di bawah garis sinyal.",
        parameters={"fast_span": 12, "slow_span": 26, "signal_span": 9},
        context_columns=("price", "macd", "macd_signal", "macd_hist", "position"),
    )

    def __init__(
        self,
        price_column: str = "close",
        fast_span: int = 12,
        slow_span: int = 26,
        signal_span: int = 9,
    ) -> None:
        super().__init__(
            price_column=price_column,
            fast_span=fast_span,
            slow_span=slow_span,
            signal_span=signal_span,
        )
        self.price_column = price_column
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.signal_span = signal_span

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.price_column not in data.columns:
            raise KeyError(f"DataFrame must contain '{self.price_column}' column")

        price_series = data[self.price_column].astype(float)
        macd_df = macd_components(
            price_series,
            fast_span=self.fast_span,
            slow_span=self.slow_span,
            signal_span=self.signal_span,
        )

        long_condition = macd_df["macd"] > macd_df["macd_signal"]
        long_entry, long_exit, short_entry, short_exit, position = build_long_only_signals(
            long_condition
        )

        return pd.DataFrame(
            {
                "long_entry": long_entry,
                "long_exit": long_exit,
                "short_entry": short_entry,
                "short_exit": short_exit,
                "price": price_series,
                "macd": macd_df["macd"],
                "macd_signal": macd_df["macd_signal"],
                "macd_hist": macd_df["macd_hist"],
                "position": position,
            },
            index=macd_df.index,
        )


__all__ = ["macd_components", "macd_signals", "Strategy"]
