"""Signal-driven backtesting pipeline used by the TradingView notebook."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..qflib_metrics import qflib_metrics_from_returns


@dataclass
class TradeRecord:
    trade_id: int
    direction: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_currency: float
    bars_held: int
    exit_reason: str
    entry_context: Dict[str, float] = field(default_factory=dict)
    exit_context: Dict[str, float] = field(default_factory=dict)


@dataclass
class BacktestOutputs:
    """Structured result returned by :class:`SignalBacktester`."""

    positions: pd.Series
    trades: pd.DataFrame
    results: pd.DataFrame
    metrics: Dict[str, float]
    trade_summary: Dict[str, float]


class SignalBacktester:
    """Iterate over pre-computed entry/exit signals and run a vectorised backtest."""

    def __init__(
        self,
        data: pd.DataFrame,
        price_column: str = "close",
        assume_same_bar_execution: bool = True,
    ) -> None:
        if price_column not in data.columns:
            raise KeyError(f"Kolom harga '{price_column}' tidak ditemukan dalam dataset")

        self.data = data.copy()
        self.price_column = price_column
        self.assume_same_bar_execution = assume_same_bar_execution

    def run(self, signals: pd.DataFrame) -> BacktestOutputs:
        """Jalankan backtest berbasis sinyal dengan output posisi, trade log, dan metrik."""

        if signals.empty:
            raise ValueError("Signals DataFrame must not be empty")

        signals = signals.reindex(self.data.index)
        context_columns = [
            column
            for column in signals.columns
            if column not in {"long_entry", "long_exit", "short_entry", "short_exit"}
        ]

        index = self.data.index
        long_entry = self._to_bool(signals.get("long_entry"), index)
        long_exit = self._to_bool(signals.get("long_exit"), index)
        short_entry = self._to_bool(signals.get("short_entry"), index)
        short_exit = self._to_bool(signals.get("short_exit"), index)

        positions, trades = self._simulate_trades(
            long_entry,
            long_exit,
            short_entry,
            short_exit,
            signals[context_columns] if context_columns else None,
        )

        results_df = self._build_backtest_frame(positions)
        metrics = qflib_metrics_from_returns(results_df["strategy_return"])

        trade_summary = {
            "total_trades": int(len(trades)),
            "long_trades": int((trades["direction"] == "Long").sum()) if not trades.empty else 0,
            "short_trades": int((trades["direction"] == "Short").sum()) if not trades.empty else 0,
            "win_rate": float((trades["pnl_pct"] > 0).mean()) if not trades.empty else np.nan,
            "avg_pnl_pct": float(trades["pnl_pct"].mean()) if not trades.empty else np.nan,
            "median_bars": float(trades["bars_held"].median()) if not trades.empty else np.nan,
        }

        return BacktestOutputs(
            positions=positions,
            trades=trades,
            results=results_df,
            metrics=metrics,
            trade_summary=trade_summary,
        )

    def _simulate_trades(
        self,
        long_entry: pd.Series,
        long_exit: pd.Series,
        short_entry: pd.Series,
        short_exit: pd.Series,
        context: Optional[pd.DataFrame],
    ) -> Tuple[pd.Series, pd.DataFrame]:
        index = self.data.index
        position_values = np.zeros(len(index), dtype=float)
        trades: List[TradeRecord] = []

        current_pos = 0
        entry_idx: Optional[int] = None
        entry_price: Optional[float] = None
        entry_direction: Optional[str] = None
        entry_context: Optional[pd.Series] = None

        for i in range(len(index)):
            price = float(self.data.iloc[i][self.price_column])

            long_entry_signal = bool(long_entry.iloc[i])
            long_exit_signal = bool(long_exit.iloc[i])
            short_entry_signal = bool(short_entry.iloc[i])
            short_exit_signal = bool(short_exit.iloc[i])

            exit_trade = False
            exit_reason = ""

            if current_pos > 0:
                if long_exit_signal:
                    exit_trade = True
                    exit_reason = "long_exit_signal"
                elif short_entry_signal:
                    exit_trade = True
                    exit_reason = "short_reversal"
            elif current_pos < 0:
                if short_exit_signal:
                    exit_trade = True
                    exit_reason = "short_exit_signal"
                elif long_entry_signal:
                    exit_trade = True
                    exit_reason = "long_reversal"

            if exit_trade and entry_idx is not None and entry_price is not None and entry_direction is not None:
                exit_context = context.iloc[i] if context is not None else None
                detailed_exit_reason = exit_reason
                if exit_context is not None and "exit_flag" in exit_context.index:
                    flag = exit_context.get("exit_flag")
                    if flag is not None and not pd.isna(flag) and str(flag):
                        detailed_exit_reason = f"{exit_reason}:{flag}"
                trades.append(
                    self._build_trade_record(
                        trade_id=len(trades) + 1,
                        direction=entry_direction,
                        entry_index=entry_idx,
                        exit_index=i,
                        exit_price=price,
                        exit_reason=detailed_exit_reason,
                        entry_price=entry_price,
                        entry_context=entry_context,
                        exit_context=exit_context,
                    )
                )
                current_pos = 0
                entry_idx = None
                entry_price = None
                entry_direction = None
                entry_context = None

            if current_pos == 0:
                if long_entry_signal:
                    current_pos = 1
                    entry_idx = i
                    entry_price = price
                    entry_direction = "Long"
                    entry_context = context.iloc[i] if context is not None else None
                elif short_entry_signal:
                    current_pos = -1
                    entry_idx = i
                    entry_price = price
                    entry_direction = "Short"
                    entry_context = context.iloc[i] if context is not None else None

            position_values[i] = current_pos

        if current_pos != 0 and entry_idx is not None and entry_price is not None and entry_direction is not None:
            exit_context = context.iloc[-1] if context is not None else None
            trades.append(
                self._build_trade_record(
                    trade_id=len(trades) + 1,
                    direction=entry_direction,
                    entry_index=entry_idx,
                    exit_index=len(index) - 1,
                    exit_price=float(self.data.iloc[-1][self.price_column]),
                    exit_reason="forced_exit_at_end",
                    entry_price=entry_price,
                    entry_context=entry_context,
                    exit_context=exit_context,
                )
            )

        positions = pd.Series(position_values, index=index, name="position")
        trades_df = self._trades_to_frame(trades)
        return positions, trades_df

    def _build_trade_record(
        self,
        trade_id: int,
        direction: str,
        entry_index: int,
        exit_index: int,
        exit_price: float,
        exit_reason: str,
        entry_price: float,
        entry_context: Optional[pd.Series],
        exit_context: Optional[pd.Series],
    ) -> TradeRecord:
        entry_price = float(entry_price)
        exit_price = float(exit_price)

        pnl_pct = exit_price / entry_price - 1.0
        if direction == "Short":
            pnl_pct = -pnl_pct
        pnl_currency = pnl_pct * entry_price
        bars_held = exit_index - entry_index

        entry_time = self.data.index[entry_index]
        exit_time = self.data.index[exit_index]

        entry_ctx = self._context_to_dict(entry_context)
        exit_ctx = self._context_to_dict(exit_context)

        return TradeRecord(
            trade_id=trade_id,
            direction=direction,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
            pnl_currency=pnl_currency,
            bars_held=bars_held,
            exit_reason=exit_reason,
            entry_context=entry_ctx,
            exit_context=exit_ctx,
        )

    def _build_backtest_frame(self, positions: pd.Series) -> pd.DataFrame:
        df = self.data.copy()
        asset_returns = df[self.price_column].pct_change().fillna(0.0)
        strategy_returns = positions.shift(1).fillna(0.0) * asset_returns
        equity_curve = (1.0 + strategy_returns).cumprod()

        results = pd.DataFrame(
            {
                "close": df[self.price_column],
                "asset_return": asset_returns,
                "position": positions,
                "strategy_return": strategy_returns,
                "equity_curve": equity_curve,
            }
        )

        rolling_max = results["equity_curve"].cummax()
        results["drawdown"] = results["equity_curve"] / rolling_max - 1.0
        results["cumulative_pnl"] = results["equity_curve"] - results["equity_curve"].iloc[0]
        return results

    @staticmethod
    def _context_to_dict(context: Optional[pd.Series]) -> Dict[str, object]:
        if context is None:
            return {}
        cleaned: Dict[str, object] = {}
        for key, value in context.items():
            if pd.isna(value):
                cleaned[key] = np.nan
                continue
            if isinstance(value, str):
                cleaned[key] = value
                continue
            try:
                cleaned[key] = float(value)
            except (TypeError, ValueError):
                cleaned[key] = str(value)
        return cleaned

    @staticmethod
    def _to_bool(series: Optional[pd.Series], index: pd.Index) -> pd.Series:
        if series is None:
            return pd.Series(False, index=index)
        series = series.reindex(index)
        if series.dtype == bool:
            return series.fillna(False)
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            return numeric.fillna(0.0).astype(float).abs() > 0.0
        return series.astype(str).str.strip().ne("")

    @staticmethod
    def _trades_to_frame(trades: Iterable[TradeRecord]) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for trade in trades:
            row: Dict[str, object] = {
                "trade_id": trade.trade_id,
                "direction": trade.direction,
                "entry_time": trade.entry_time,
                "exit_time": trade.exit_time,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "pnl_pct": trade.pnl_pct,
                "pnl_currency": trade.pnl_currency,
                "bars_held": trade.bars_held,
                "exit_reason": trade.exit_reason,
            }
            for key, value in trade.entry_context.items():
                row[f"entry_{key}"] = value
            for key, value in trade.exit_context.items():
                row[f"exit_{key}"] = value
            rows.append(row)

        if not rows:
            return pd.DataFrame(
                columns=[
                    "trade_id",
                    "direction",
                    "entry_time",
                    "exit_time",
                    "entry_price",
                    "exit_price",
                    "pnl_pct",
                    "pnl_currency",
                    "bars_held",
                    "exit_reason",
                ]
            )

        return pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)


__all__ = ["SignalBacktester", "BacktestOutputs", "TradeRecord"]
