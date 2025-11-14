"""EMA 112 trend-following strategy with ATR-based exit."""
from __future__ import annotations

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
    return tr.rolling(window=window, min_periods=1).mean()


class Strategy(StrategyBase):
    """Go long on EMA 50 crossing above EMA 112; exit with ATR trailing stop."""

    metadata = StrategyMetadata(
        name="ema112_atr",
        description=(
            "Strategi trend-following yang menggunakan EMA 50 dan EMA 112 sebagai filter arah dan"
            " trailing stop berbasis ATR untuk manajemen risiko."
        ),
        entry=(
            "Entry long ketika EMA fast (50) menyilang ke atas EMA slow (112)."
            " Sinyal ini menandakan perubahan tren ke arah bullish."
        ),
        exit=(
            "Exit long ketika harga penutupan jatuh di bawah EMA slow dikurangi ATR * multiplier,"
            " memberikan ruang bernapas sambil melindungi profit."
        ),
        parameters={
            "fast_span": 50,
            "slow_span": 112,
            "atr_window": 14,
            "atr_multiplier": 1.5,
        },
        context_columns=("ema_fast", "ema_slow", "atr", "atr_trailing_stop"),
    )

    def __init__(
        self,
        fast_span: int = 50,
        slow_span: int = 112,
        atr_window: int = 14,
        atr_multiplier: float = 1.5,
    ) -> None:
        super().__init__(
            fast_span=fast_span,
            slow_span=slow_span,
            atr_window=atr_window,
            atr_multiplier=atr_multiplier,
        )
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.atr_window = atr_window
        self.atr_multiplier = atr_multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"open", "high", "low", "close"}
        missing = required_cols - set(data.columns)
        if missing:
            raise KeyError(f"DataFrame tidak memiliki kolom harga: {sorted(missing)}")

        close = data["close"].astype(float)
        ema_fast = _ema(close, self.fast_span)
        ema_slow = _ema(close, self.slow_span)
        atr = _atr(data, self.atr_window)

        crossover = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        trailing_stop = ema_slow - self.atr_multiplier * atr
        exit_signal = close < trailing_stop

        signals = pd.DataFrame(
            {
                "long_entry": crossover,
                "long_exit": exit_signal,
                "short_entry": False,
                "short_exit": False,
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "atr": atr,
                "atr_trailing_stop": trailing_stop,
            },
            index=data.index,
        )

        return signals


__all__ = ["Strategy"]
