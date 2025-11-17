"""Trend-following strategy conditioned on ATR-driven volatility filter."""
from __future__ import annotations

import pandas as pd

from .indicators import calculate_ema
from .strategy_backtest.base import StrategyBase, StrategyMetadata
from .strategy_backtest.utils import build_long_only_signals


def average_true_range(
    df: pd.DataFrame,
    period: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """Compute the Average True Range (ATR)."""
    required_columns = {high_col, low_col, close_col}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns for ATR: {sorted(missing)}")

    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    close = df[close_col].astype(float)

    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    atr.name = f"ATR_{period}"
    return atr


def atr_filter_signals(
    df: pd.DataFrame,
    ema_span: int = 50,
    atr_period: int = 14,
    median_window: int = 100,
) -> pd.Series:
    """Return long/cash positions where trend and volatility filters align."""
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column")

    ema = calculate_ema(df["close"].astype(float), ema_span)
    atr = average_true_range(df, atr_period)
    atr_median = atr.rolling(window=median_window, min_periods=1).median()

    long_condition = (df["close"].astype(float) > ema) & (atr > atr_median)
    positions = long_condition.astype(float)
    positions = positions.ffill().fillna(0.0)
    positions.name = "position"

    return positions


class Strategy(StrategyBase):
    """Trend-following strategy gated by ATR volatility filter."""

    metadata = StrategyMetadata(
        name="atr_filter",
        description=(
            "Mengikuti tren ketika harga berada di atas EMA dan volatilitas (ATR) lebih tinggi dari median historis."
        ),
        entry="Entry long saat penutupan di atas EMA dan ATR di atas median rolling.",
        exit="Keluar ketika salah satu kondisi tidak terpenuhi.",
        parameters={"ema_span": 50, "atr_period": 14, "median_window": 100},
        context_columns=("price", "ema", "atr", "atr_median", "position"),
    )

    def __init__(
        self,
        ema_span: int = 50,
        atr_period: int = 14,
        median_window: int = 100,
        price_column: str = "close",
    ) -> None:
        super().__init__(
            ema_span=ema_span,
            atr_period=atr_period,
            median_window=median_window,
            price_column=price_column,
        )
        self.ema_span = ema_span
        self.atr_period = atr_period
        self.median_window = median_window
        self.price_column = price_column

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        required = {"high", "low", self.price_column}
        missing = required - set(data.columns)
        if missing:
            raise KeyError(f"DataFrame missing columns: {sorted(missing)}")

        price = data[self.price_column].astype(float)
        ema = calculate_ema(price, self.ema_span)
        atr = average_true_range(
            data,
            self.atr_period,
            high_col="high",
            low_col="low",
            close_col=self.price_column,
        )
        atr_median = atr.rolling(window=self.median_window, min_periods=1).median()

        long_condition = (price > ema) & (atr > atr_median)
        signals = build_long_only_signals(long_condition)
        long_entry, long_exit, short_entry, short_exit, position = signals

        return pd.DataFrame(
            {
                "long_entry": long_entry,
                "long_exit": long_exit,
                "short_entry": short_entry,
                "short_exit": short_exit,
                "price": price,
                "ema": ema,
                "atr": atr,
                "atr_median": atr_median,
                "position": position,
            },
            index=data.index,
        )


__all__ = ["average_true_range", "atr_filter_signals", "Strategy"]
