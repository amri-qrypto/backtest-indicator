"""VWAP-based counter-trend and trend-aligned fade strategy."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..base import StrategyBase, StrategyMetadata


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


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _macd(series: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


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


class Strategy(StrategyBase):
    """VWAP mean-reversion longs dan trend-fade shorts sesuai rulebook pengguna."""

    metadata = StrategyMetadata(
        name="vwap",
        description=(
            "Strategi VWAP yang mencari pantulan counter-trend di bawah VWAP dan fade rally di atas VWAP "
            "saat tren turun."
        ),
        entry=(
            "Long ketika harga pertama kali menyelam di bawah VWAP dan ada konfirmasi bullish (wick bawah, volume naik, RSI > 30) "
            "untuk potensi reversion. Short ketika harga memantul di atas VWAP dalam tren turun dengan candle rejection, volume melemah, "
            "RSI < 70 dan MACD histogram melemah."
        ),
        exit=(
            "Keluar utama di level VWAP sebagai target reversion. Stop-loss menggunakan ATR. Jika target VWAP tercapai, posisi ditutup penuh."
        ),
        parameters={
            "rsi_window": 14,
            "volume_ma_window": 20,
            "volume_spike_ratio": 1.1,
            "volume_fade_ratio": 0.9,
            "atr_window": 14,
            "atr_stop_multiplier": 1.5,
            "min_reversion_atr": 0.5,
            "session_frequency": "1D",
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "wick_ratio_threshold": 0.6,
            "allow_counter_trend_long": True,
            "allow_trend_short": True,
        },
        context_columns=(
            "vwap",
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "atr",
            "volume_ma",
            "active_entry_price",
            "stop_level",
            "target_level",
            "exit_flag",
            "position",
        ),
    )

    def __init__(
        self,
        rsi_window: int = 14,
        volume_ma_window: int = 20,
        volume_spike_ratio: float = 1.1,
        volume_fade_ratio: float = 0.9,
        atr_window: int = 14,
        atr_stop_multiplier: float = 1.5,
        min_reversion_atr: float = 0.5,
        session_frequency: str = "1D",
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        wick_ratio_threshold: float = 0.6,
        allow_counter_trend_long: bool = True,
        allow_trend_short: bool = True,
    ) -> None:
        super().__init__(
            rsi_window=rsi_window,
            volume_ma_window=volume_ma_window,
            volume_spike_ratio=volume_spike_ratio,
            volume_fade_ratio=volume_fade_ratio,
            atr_window=atr_window,
            atr_stop_multiplier=atr_stop_multiplier,
            min_reversion_atr=min_reversion_atr,
            session_frequency=session_frequency,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            wick_ratio_threshold=wick_ratio_threshold,
            allow_counter_trend_long=allow_counter_trend_long,
            allow_trend_short=allow_trend_short,
        )
        self.rsi_window = rsi_window
        self.volume_ma_window = volume_ma_window
        self.volume_spike_ratio = volume_spike_ratio
        self.volume_fade_ratio = volume_fade_ratio
        self.atr_window = atr_window
        self.atr_stop_multiplier = atr_stop_multiplier
        self.min_reversion_atr = min_reversion_atr
        self.session_frequency = session_frequency
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.wick_ratio_threshold = wick_ratio_threshold
        self.allow_counter_trend_long = allow_counter_trend_long
        self.allow_trend_short = allow_trend_short

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(data.columns)
        if missing:
            raise KeyError(f"DataFrame tidak memiliki kolom harga/volume: {sorted(missing)}")

        open_ = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)
        index = data.index

        vwap = _session_vwap(data, self.session_frequency)
        atr = _atr(data, self.atr_window)
        rsi = _rsi(close, self.rsi_window)
        macd, macd_signal, macd_hist = _macd(close, self.macd_fast, self.macd_slow, self.macd_signal)

        volume_ma = volume.rolling(self.volume_ma_window, min_periods=1).mean()

        candle_range = (high - low).replace(0.0, np.nan)
        body_low = np.minimum(open_, close)
        body_high = np.maximum(open_, close)
        lower_wick = body_low - low
        upper_wick = high - body_high
        lower_wick_ratio = (lower_wick / candle_range).fillna(0.0)
        upper_wick_ratio = (upper_wick / candle_range).fillna(0.0)

        bullish_body = (close >= open_).fillna(False)
        bearish_body = (close <= open_).fillna(False)

        volume_rise = (volume_ma.gt(0) & (volume > volume_ma * self.volume_spike_ratio)).fillna(False)
        volume_fade = (volume_ma.gt(0) & (volume < volume_ma * self.volume_fade_ratio)).fillna(False)

        rsi_cross_up = (rsi >= 50) & (rsi.shift(1) < 50)
        rsi_cross_down = (rsi <= 50) & (rsi.shift(1) > 50)

        long_rsi_ok = (
            (rsi > 30)
            & (rsi >= rsi.shift(1))
            & (rsi_cross_up | (rsi >= 50) | (rsi.rolling(3, min_periods=1).mean() > rsi.shift(1)))
        ).fillna(False)
        short_rsi_ok = (
            (rsi < 70)
            & (rsi <= rsi.shift(1))
            & (rsi_cross_down | (rsi <= 50) | (rsi.rolling(3, min_periods=1).mean() < rsi.shift(1)))
        ).fillna(False)

        macd_weaken = ((macd_hist < macd_hist.shift(1)) | (macd < macd_signal)).fillna(False)

        below_vwap = (close < vwap).fillna(False)
        above_vwap = (close > vwap).fillna(False)
        cross_below = below_vwap & (close.shift(1) >= vwap.shift(1)).fillna(False)
        cross_above = above_vwap & (close.shift(1) <= vwap.shift(1)).fillna(False)

        wick_support = lower_wick_ratio >= self.wick_ratio_threshold
        wick_reject = upper_wick_ratio >= self.wick_ratio_threshold

        if self.allow_counter_trend_long:
            long_condition = (
                cross_below
                & (bullish_body | wick_support)
                & volume_rise
                & long_rsi_ok
                & atr.notna()
                & vwap.notna()
            )
        else:
            long_condition = pd.Series(False, index=index)

        if self.allow_trend_short:
            short_condition = (
                cross_above
                & (bearish_body | wick_reject)
                & volume_fade
                & short_rsi_ok
                & macd_weaken
                & atr.notna()
                & vwap.notna()
            )
        else:
            short_condition = pd.Series(False, index=index)

        long_entry = pd.Series(False, index=index)
        long_exit = pd.Series(False, index=index)
        short_entry = pd.Series(False, index=index)
        short_exit = pd.Series(False, index=index)

        active_entry_price = pd.Series(float("nan"), index=index)
        stop_level = pd.Series(float("nan"), index=index)
        target_level = pd.Series(float("nan"), index=index)
        exit_flag = pd.Series("", index=index, dtype="object")
        position_context = pd.Series("flat", index=index, dtype="object")

        position: Optional[str] = None
        current_entry = float("nan")
        current_stop = float("nan")
        current_target = float("nan")

        for ts in index:
            price = float(close.loc[ts])
            atr_value = float(atr.loc[ts]) if pd.notna(atr.loc[ts]) else float("nan")
            vwap_value = float(vwap.loc[ts]) if pd.notna(vwap.loc[ts]) else float("nan")

            # snapshot existing state
            active_entry_price.loc[ts] = current_entry
            stop_level.loc[ts] = current_stop
            target_level.loc[ts] = current_target if np.isfinite(current_target) else vwap_value
            position_context.loc[ts] = position or "flat"

            if position is not None:
                if np.isfinite(vwap_value):
                    current_target = vwap_value
                if position == "long":
                    if price <= current_stop:
                        long_exit.loc[ts] = True
                        exit_flag.loc[ts] = "stop"
                        position = None
                    elif np.isfinite(current_target) and price >= current_target:
                        long_exit.loc[ts] = True
                        exit_flag.loc[ts] = "target_vwap"
                        position = None
                else:  # short
                    if price >= current_stop:
                        short_exit.loc[ts] = True
                        exit_flag.loc[ts] = "stop"
                        position = None
                    elif np.isfinite(current_target) and price <= current_target:
                        short_exit.loc[ts] = True
                        exit_flag.loc[ts] = "target_vwap"
                        position = None

                if position is None:
                    current_entry = float("nan")
                    current_stop = float("nan")
                    current_target = float("nan")
                    continue

            if position is None and bool(long_condition.loc[ts]):
                if not np.isfinite(atr_value) or atr_value <= 0:
                    continue
                long_entry.loc[ts] = True
                position = "long"
                current_entry = price
                stop_buffer = self.atr_stop_multiplier * atr_value
                current_stop = current_entry - stop_buffer
                target_candidate = vwap_value
                if not np.isfinite(target_candidate) or target_candidate <= current_entry:
                    target_candidate = current_entry + self.min_reversion_atr * atr_value
                current_target = target_candidate

                active_entry_price.loc[ts] = current_entry
                stop_level.loc[ts] = current_stop
                target_level.loc[ts] = current_target
                position_context.loc[ts] = position
                continue

            if position is None and bool(short_condition.loc[ts]):
                if not np.isfinite(atr_value) or atr_value <= 0:
                    continue
                short_entry.loc[ts] = True
                position = "short"
                current_entry = price
                stop_buffer = self.atr_stop_multiplier * atr_value
                current_stop = current_entry + stop_buffer
                target_candidate = vwap_value
                if not np.isfinite(target_candidate) or target_candidate >= current_entry:
                    target_candidate = current_entry - self.min_reversion_atr * atr_value
                current_target = target_candidate

                active_entry_price.loc[ts] = current_entry
                stop_level.loc[ts] = current_stop
                target_level.loc[ts] = current_target
                position_context.loc[ts] = position

        context = pd.DataFrame(
            {
                "vwap": vwap,
                "rsi": rsi,
                "macd": macd,
                "macd_signal": macd_signal,
                "macd_hist": macd_hist,
                "atr": atr,
                "volume_ma": volume_ma,
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
