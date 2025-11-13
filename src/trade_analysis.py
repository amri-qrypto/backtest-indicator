"""Utilities to derive trade-level analytics from backtest outputs."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(slots=True)
class TradeSummary:
    """Aggregated statistics calculated from a trade log."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    net_profit: float
    average_return_pct: float
    average_trade_duration: float


def _signal_label(direction: float) -> str:
    direction_int = int(np.sign(direction))
    return {1: "Long", -1: "Short"}.get(direction_int, "Flat")


def generate_trade_log(
    backtest: pd.DataFrame,
    *,
    initial_capital: float,
) -> pd.DataFrame:
    """Return a trade-by-trade breakdown derived from ``run_backtest`` output."""

    required_columns = {"position", "close", "strategy_ret", "equity_curve", "trading_cost"}
    missing = required_columns.difference(backtest.columns)
    if missing:
        raise ValueError(f"Backtest output missing required columns: {sorted(missing)}")

    if backtest.empty:
        return pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "direction",
                "signal",
                "entry_price",
                "exit_price",
                "entry_equity",
                "exit_equity",
                "return_pct",
                "net_pnl",
                "gross_profit",
                "gross_loss",
                "bars_held",
                "max_drawdown",
                "trading_cost_value",
                "position_size",
                "stop_loss",
                "is_win",
                "cumulative_pnl",
            ]
        )

    positions = backtest["position"].fillna(0.0)
    equity = backtest["equity_curve"].astype(float)
    notional_equity = equity.shift(1).fillna(initial_capital)

    in_position = positions != 0
    groups = (in_position != in_position.shift()).cumsum()

    log_records = []
    cumulative_pnl = 0.0

    for _, mask_series in groups.groupby(groups):
        segment_index = mask_series.index
        if not in_position.loc[segment_index[0]]:
            continue

        trade_df = backtest.loc[segment_index]
        entry_idx = trade_df.index[0]
        exit_idx = trade_df.index[-1]

        direction = float(positions.loc[entry_idx])
        signal = _signal_label(direction)

        entry_equity = float(notional_equity.loc[entry_idx])

        trade_returns = (1.0 + trade_df["strategy_ret"]).prod()
        exit_equity = entry_equity * trade_returns
        net_pnl = exit_equity - entry_equity
        cumulative_pnl += net_pnl

        max_equity = (trade_df["equity_curve"] / entry_equity).cummax()
        relative_equity = trade_df["equity_curve"] / entry_equity
        max_drawdown = float((relative_equity / max_equity - 1.0).min())

        trade_cost_value = float(
            (trade_df["trading_cost"] * notional_equity.loc[trade_df.index]).sum()
        )

        record = {
            "entry_time": entry_idx,
            "exit_time": exit_idx,
            "direction": direction,
            "signal": signal,
            "entry_price": float(trade_df["close"].iloc[0]),
            "exit_price": float(trade_df["close"].iloc[-1]),
            "entry_equity": entry_equity,
            "exit_equity": exit_equity,
            "return_pct": float(trade_returns - 1.0),
            "net_pnl": net_pnl,
            "gross_profit": max(net_pnl, 0.0),
            "gross_loss": min(net_pnl, 0.0),
            "bars_held": int(len(trade_df)),
            "max_drawdown": max_drawdown,
            "trading_cost_value": trade_cost_value,
            "position_size": float(notional_equity.loc[entry_idx] * abs(direction)),
            "stop_loss": float(trade_df.get("stop_loss", pd.Series([np.nan])).iloc[0])
            if "stop_loss" in trade_df
            else np.nan,
            "is_win": bool(net_pnl > 0.0),
            "cumulative_pnl": cumulative_pnl,
        }

        log_records.append(record)

    return pd.DataFrame(log_records)


def summarise_trades(trade_log: pd.DataFrame) -> TradeSummary:
    """Compute aggregated metrics for a given trade log."""

    if trade_log.empty:
        return TradeSummary(0, 0, 0, float("nan"), 0.0, 0.0, 0.0, float("nan"), float("nan"))

    total = int(len(trade_log))
    wins = int((trade_log["net_pnl"] > 0).sum())
    losses = int((trade_log["net_pnl"] < 0).sum())
    win_rate = wins / total if total else float("nan")
    gross_profit = float(trade_log.loc[trade_log["net_pnl"] > 0, "net_pnl"].sum())
    gross_loss = float(trade_log.loc[trade_log["net_pnl"] < 0, "net_pnl"].sum())
    net_profit = float(trade_log["net_pnl"].sum())
    average_return = float(trade_log["return_pct"].mean())
    average_duration = float(trade_log["bars_held"].mean())

    return TradeSummary(
        total_trades=total,
        winning_trades=wins,
        losing_trades=losses,
        win_rate=win_rate,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_profit=net_profit,
        average_return_pct=average_return,
        average_trade_duration=average_duration,
    )


def failed_entries(trade_log: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Filter trades with P&L below ``threshold`` for post-trade evaluation."""

    if trade_log.empty:
        return trade_log.copy()

    failures = trade_log.loc[trade_log["net_pnl"] <= threshold].copy()
    failures["failure_reason"] = np.where(
        failures["net_pnl"] < 0,
        "Negative net P&L",
        "Net P&L equals threshold",
    )
    return failures


__all__ = ["TradeSummary", "generate_trade_log", "summarise_trades", "failed_entries"]

