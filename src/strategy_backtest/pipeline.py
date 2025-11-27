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
    mae: float = 0.0
    mfe: float = 0.0
    long_pnl_pct: float = 0.0
    short_pnl_pct: float = 0.0


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

    def run(
        self, signals: pd.DataFrame, position_scale: Optional[pd.Series] = None
    ) -> BacktestOutputs:
        """Jalankan backtest berbasis sinyal dengan output posisi, trade log, dan metrik."""

        if signals.empty:
            raise ValueError("Signals DataFrame must not be empty")

        signals = signals.reindex(self.data.index)
        context_columns = [
            column
            for column in signals.columns
            if column
            not in {"long_entry", "long_exit", "short_entry", "short_exit", "position_scale"}
        ]

        if position_scale is None and "position_scale" in signals.columns:
            position_scale = signals["position_scale"]

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

        scaled_positions = self._apply_position_scale(positions, position_scale)

        results_df = self._build_backtest_frame(scaled_positions)
        metrics = qflib_metrics_from_returns(results_df["strategy_return"])
        trade_summary = self._summarise_trades(trades, results_df["position"])

        return BacktestOutputs(
            positions=results_df["position"],
            trades=trades,
            results=results_df,
            metrics=metrics,
            trade_summary=trade_summary,
        )

    @staticmethod
    def _max_streak(mask: pd.Series) -> int:
        max_streak = 0
        current = 0
        for flag in mask.astype(bool):
            if flag:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    def _summarise_trades(self, trades: pd.DataFrame, positions: pd.Series) -> Dict[str, float]:
        total_trades = int(len(trades))
        summary: Dict[str, float] = {"total_trades": total_trades}

        total_bars = int(len(positions))
        if total_bars > 0:
            in_market = int((positions != 0).sum())
            long_bars = int((positions > 0).sum())
            short_bars = int((positions < 0).sum())
            flat_bars = total_bars - in_market

            summary.update(
                {
                    "bars_in_market": float(in_market),
                    "bars_flat": float(flat_bars),
                    "time_in_market_pct": float(in_market / total_bars),
                    "flat_time_pct": float(flat_bars / total_bars),
                    "long_exposure_pct": float(long_bars / total_bars),
                    "short_exposure_pct": float(short_bars / total_bars),
                    "avg_position": float(positions.mean()),
                    "avg_abs_position": float(positions.abs().mean()),
                }
            )
        else:
            summary.update(
                {
                    "bars_in_market": np.nan,
                    "bars_flat": np.nan,
                    "time_in_market_pct": np.nan,
                    "flat_time_pct": np.nan,
                    "long_exposure_pct": np.nan,
                    "short_exposure_pct": np.nan,
                    "avg_position": np.nan,
                    "avg_abs_position": np.nan,
                }
            )

        if trades.empty:
            summary.update(
                {
                    "long_trades": 0,
                    "short_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "breakeven_trades": 0,
                    "win_rate": np.nan,
                    "loss_rate": np.nan,
                    "breakeven_rate": np.nan,
                    "avg_pnl_pct": np.nan,
                    "avg_pnl_currency": np.nan,
                    "median_pnl_pct": np.nan,
                    "net_profit_pct": np.nan,
                    "net_profit_currency": np.nan,
                    "gross_profit_pct": np.nan,
                    "gross_loss_pct": np.nan,
                    "gross_profit_currency": np.nan,
                    "gross_loss_currency": np.nan,
                    "profit_factor_pct": np.nan,
                    "profit_factor_currency": np.nan,
                    "avg_win_pct": np.nan,
                    "avg_loss_pct": np.nan,
                    "avg_win_currency": np.nan,
                    "avg_loss_currency": np.nan,
                    "avg_bars": np.nan,
                    "median_bars": np.nan,
                    "avg_bars_winning": np.nan,
                    "avg_bars_losing": np.nan,
                    "max_consecutive_wins": 0,
                    "max_consecutive_losses": 0,
                    "long_short_ratio": np.nan,
                    "best_trade_pct": np.nan,
                    "worst_trade_pct": np.nan,
                    "best_trade_currency": np.nan,
                    "worst_trade_currency": np.nan,
                }
            )
            return summary

        long_trades = int((trades["direction"] == "Long").sum())
        short_trades = int((trades["direction"] == "Short").sum())

        pnl_pct = trades["pnl_pct"].astype(float)
        pnl_currency = trades["pnl_currency"].astype(float)
        bars_held = trades["bars_held"].astype(float)

        winning_mask = pnl_pct > 0
        losing_mask = pnl_pct < 0
        breakeven_mask = ~(winning_mask | losing_mask)

        winning_trades = int(winning_mask.sum())
        losing_trades = int(losing_mask.sum())
        breakeven_trades = int(breakeven_mask.sum())

        wins_pct = pnl_pct[winning_mask]
        losses_pct = pnl_pct[losing_mask]
        wins_cur = pnl_currency[winning_mask]
        losses_cur = pnl_currency[losing_mask]

        gross_profit_pct = float(wins_pct.sum()) if winning_trades else 0.0
        gross_loss_pct = float(-losses_pct.sum()) if losing_trades else 0.0
        gross_profit_currency = float(wins_cur.sum()) if winning_trades else 0.0
        gross_loss_currency = float(-losses_cur.sum()) if losing_trades else 0.0

        profit_factor_pct = gross_profit_pct / gross_loss_pct if gross_loss_pct > 0 else np.nan
        profit_factor_currency = (
            gross_profit_currency / gross_loss_currency if gross_loss_currency > 0 else np.nan
        )

        summary.update(
            {
                "long_trades": long_trades,
                "short_trades": short_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "breakeven_trades": breakeven_trades,
                "win_rate": float(winning_trades / total_trades) if total_trades else np.nan,
                "loss_rate": float(losing_trades / total_trades) if total_trades else np.nan,
                "breakeven_rate": float(breakeven_trades / total_trades) if total_trades else np.nan,
                "avg_pnl_pct": float(pnl_pct.mean()),
                "avg_pnl_currency": float(pnl_currency.mean()),
                "median_pnl_pct": float(pnl_pct.median()),
                "net_profit_pct": float(pnl_pct.sum()),
                "net_profit_currency": float(pnl_currency.sum()),
                "gross_profit_pct": gross_profit_pct,
                "gross_loss_pct": gross_loss_pct,
                "gross_profit_currency": gross_profit_currency,
                "gross_loss_currency": gross_loss_currency,
                "profit_factor_pct": profit_factor_pct,
                "profit_factor_currency": profit_factor_currency,
                "avg_win_pct": float(wins_pct.mean()) if winning_trades else np.nan,
                "avg_loss_pct": float(losses_pct.mean()) if losing_trades else np.nan,
                "avg_win_currency": float(wins_cur.mean()) if winning_trades else np.nan,
                "avg_loss_currency": float(losses_cur.mean()) if losing_trades else np.nan,
                "avg_bars": float(bars_held.mean()),
                "median_bars": float(bars_held.median()),
                "avg_bars_winning": float(bars_held[winning_mask].mean()) if winning_trades else np.nan,
                "avg_bars_losing": float(bars_held[losing_mask].mean()) if losing_trades else np.nan,
                "max_consecutive_wins": float(self._max_streak(winning_mask)),
                "max_consecutive_losses": float(self._max_streak(losing_mask)),
                "long_short_ratio": float(long_trades / short_trades) if short_trades else np.nan,
                "best_trade_pct": float(pnl_pct.max()),
                "worst_trade_pct": float(pnl_pct.min()),
                "best_trade_currency": float(pnl_currency.max()),
                "worst_trade_currency": float(pnl_currency.min()),
            }
        )

        return summary

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
                        price_path=self.data.iloc[entry_idx : i + 1][self.price_column],
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
                    price_path=self.data.iloc[entry_idx:][self.price_column],
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
        price_path: pd.Series,
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

        price_path = price_path.astype(float)
        if direction == "Short":
            pnl_path = entry_price / price_path - 1.0
        else:
            pnl_path = price_path / entry_price - 1.0
        mae = float(pnl_path.min()) if not pnl_path.empty else 0.0
        mfe = float(pnl_path.max()) if not pnl_path.empty else 0.0

        long_pnl_pct = float(pnl_pct) if direction == "Long" else 0.0
        short_pnl_pct = float(pnl_pct) if direction == "Short" else 0.0

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
            mae=mae,
            mfe=mfe,
            long_pnl_pct=long_pnl_pct,
            short_pnl_pct=short_pnl_pct,
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
    def _apply_position_scale(
        positions: pd.Series, scale: Optional[pd.Series]
    ) -> pd.Series:
        if scale is None:
            return positions

        aligned = pd.to_numeric(scale.reindex(positions.index), errors="coerce").fillna(0.0)
        scaled = (positions.astype(float) * aligned).rename("position")
        return scaled

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
                "mae": trade.mae,
                "mfe": trade.mfe,
                "long_pnl_pct": trade.long_pnl_pct,
                "short_pnl_pct": trade.short_pnl_pct,
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
