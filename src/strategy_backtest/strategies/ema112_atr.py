"""EMA 112 trend-following strategy with ATR-based risk management."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..base import StrategyBase, StrategyMetadata


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Return Wilder's ATR (RMA) to match common charting platforms."""

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
    """Go long or short on EMA 112 momentum with ATR stop/target management."""

    metadata = StrategyMetadata(
        name="ema112_atr",
        description=(
            "Strategi trend-following berbasis EMA 112 dengan fokus pada momentum harga dan"
            " pengelolaan risiko fixed-R multiple menggunakan ATR harian."
        ),
        entry=(
            "Entry ketika harga menembus EMA 112 searah tren jangka menengah dan volatilitas"
            " berada di rentang yang diizinkan. Posisi dapat long maupun short."
        ),
        exit=(
            "Exit ketika harga menyentuh stop-loss ATR (1R) atau target profit (risk_reward × R)"
            " sesuai arah posisi."
        ),
        parameters={
            "slow_span": 112,
            "atr_window": 14,
            "atr_multiplier": 1.5,
            "risk_reward": 2.0,
            "trend_lookback": 5,
            "volatility_low_pct": 0.01,
            "volatility_high_pct": 0.06,
            "allow_short": True,
        },
        context_columns=(
            "ema_trend",
            "atr",
            "atr_entry",
            "active_entry_price",
            "stop_level",
            "target_level",
            "exit_flag",
            "position",
        ),
    )

    def __init__(
        self,
        slow_span: int = 112,
        atr_window: int = 14,
        atr_multiplier: float = 1.5,
        risk_reward: float = 2.0,
        trend_lookback: int = 5,
        volatility_low_pct: float = 0.01,
        volatility_high_pct: float = 0.06,
        allow_short: bool = True,
    ) -> None:
        super().__init__(
            slow_span=slow_span,
            atr_window=atr_window,
            atr_multiplier=atr_multiplier,
            risk_reward=risk_reward,
            trend_lookback=trend_lookback,
            volatility_low_pct=volatility_low_pct,
            volatility_high_pct=volatility_high_pct,
            allow_short=allow_short,
        )
        self.slow_span = slow_span
        self.atr_window = atr_window
        self.atr_multiplier = atr_multiplier
        self.risk_reward = risk_reward
        self.trend_lookback = trend_lookback
        self.volatility_low_pct = volatility_low_pct
        self.volatility_high_pct = volatility_high_pct
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"open", "high", "low", "close"}
        missing = required_cols - set(data.columns)
        if missing:
            raise KeyError(f"DataFrame tidak memiliki kolom harga: {sorted(missing)}")

        close = data["close"].astype(float)
        ema_trend = _ema(close, self.slow_span)
        atr = _atr(data, self.atr_window)

        index = data.index
        cross_above = (close > ema_trend) & (close.shift(1) <= ema_trend.shift(1))
        cross_below = (close < ema_trend) & (close.shift(1) >= ema_trend.shift(1))
        if self.trend_lookback > 0:
            slope_filter_long = ema_trend > ema_trend.shift(self.trend_lookback)
            slope_filter_short = ema_trend < ema_trend.shift(self.trend_lookback)
        else:
            slope_filter_long = pd.Series(True, index=index)
            slope_filter_short = pd.Series(True, index=index)

        close_safe = close.replace(0, np.nan)
        vol_norm = atr / close_safe
        low, high = sorted((self.volatility_low_pct, self.volatility_high_pct))
        volatility_filter = vol_norm.between(low, high).fillna(False)

        entry_condition = cross_above & slope_filter_long & volatility_filter
        if self.allow_short:
            short_condition = cross_below & slope_filter_short & volatility_filter
        else:
            short_condition = pd.Series(False, index=index)

        long_entry = pd.Series(False, index=index)
        long_exit = pd.Series(False, index=index)
        short_entry = pd.Series(False, index=index)
        short_exit = pd.Series(False, index=index)
        active_entry_price = pd.Series(float("nan"), index=index)
        stop_level = pd.Series(float("nan"), index=index)
        target_level = pd.Series(float("nan"), index=index)
        atr_entry = pd.Series(float("nan"), index=index)
        exit_flag = pd.Series("", index=index, dtype="object")
        position_context = pd.Series("flat", index=index, dtype="object")

        position: Optional[str] = None
        current_entry = float("nan")
        current_stop = float("nan")
        current_target = float("nan")
        current_atr = float("nan")

        for ts in index:
            price = float(close.loc[ts])
            atr_value = float(atr.loc[ts])

            # Snapshot of current state before evaluating signals
            active_entry_price.loc[ts] = current_entry
            stop_level.loc[ts] = current_stop
            target_level.loc[ts] = current_target
            atr_entry.loc[ts] = current_atr
            position_context.loc[ts] = position or "flat"

            if position is not None:
                if position == "long":
                    stop_hit = price <= current_stop
                    target_hit = price >= current_target
                else:
                    stop_hit = price >= current_stop
                    target_hit = price <= current_target

                if stop_hit or target_hit:
                    if position == "long":
                        long_exit.loc[ts] = True
                    else:
                        short_exit.loc[ts] = True
                    exit_flag.loc[ts] = "target" if target_hit and not stop_hit else "stop"
                    position = None
                    current_entry = float("nan")
                    current_stop = float("nan")
                    current_target = float("nan")
                    current_atr = float("nan")

            if position is None and bool(entry_condition.loc[ts]):
                long_entry.loc[ts] = True
                position = "long"
                current_entry = price
                current_atr = atr_value
                risk_distance = self.atr_multiplier * current_atr
                current_stop = current_entry - risk_distance
                current_target = current_entry + self.risk_reward * risk_distance

                active_entry_price.loc[ts] = current_entry
                stop_level.loc[ts] = current_stop
                target_level.loc[ts] = current_target
                atr_entry.loc[ts] = current_atr
                position_context.loc[ts] = position
                continue

            if position is None and bool(short_condition.loc[ts]):
                short_entry.loc[ts] = True
                position = "short"
                current_entry = price
                current_atr = atr_value
                risk_distance = self.atr_multiplier * current_atr
                current_stop = current_entry + risk_distance
                current_target = current_entry - self.risk_reward * risk_distance

                active_entry_price.loc[ts] = current_entry
                stop_level.loc[ts] = current_stop
                target_level.loc[ts] = current_target
                atr_entry.loc[ts] = current_atr
                position_context.loc[ts] = position

        context = pd.DataFrame(
            {
                "ema_trend": ema_trend,
                "atr": atr,
                "atr_entry": atr_entry,
                "active_entry_price": active_entry_price,
                "stop_level": stop_level,
                "target_level": target_level,
                "exit_flag": exit_flag.where(exit_flag != "", other=pd.NA),
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
