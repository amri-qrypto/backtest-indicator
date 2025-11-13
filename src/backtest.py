"""Lightweight backtesting utilities compatible with QF-Lib containers."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Optional

import numpy as np
import pandas as pd

from qflib_adapters import to_qfseries


@dataclass
class BacktestResult:
    """Container storing the backtest output."""
    data: pd.DataFrame

    @property
    def equity(self) -> pd.Series:
        return self.data["equity_curve"]

    def to_qfseries(self) -> Dict[str, object]:
        return {column: to_qfseries(self.data[column]) for column in self.data.columns}


def run_backtest(
    df: pd.DataFrame,
    positions: pd.Series,
    initial_capital: float = 10000.0,
    trading_cost_bps: float = 0.0,
    stop_loss: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Run a vectorized backtest for a long/cash strategy."""
    if not df.index.equals(positions.index):
        positions = positions.reindex(df.index)

    positions = positions.fillna(0.0).astype(float)

    result = df.copy()
    result["asset_ret"] = result["close"].pct_change().fillna(0.0)

    shifted_positions = positions.shift(1).fillna(0.0)
    result["position"] = positions
    result["strategy_ret"] = shifted_positions * result["asset_ret"]

    trades = positions.diff().abs().fillna(0.0)
    if trading_cost_bps:
        cost = trades * (trading_cost_bps / 10000.0)
        result["strategy_ret"] -= cost
    else:
        cost = pd.Series(np.zeros(len(result)), index=result.index, dtype=float)

    result["trading_cost"] = cost
    result["equity_curve"] = initial_capital * (1.0 + result["strategy_ret"]).cumprod()

    notional_equity = result["equity_curve"].shift(1).fillna(initial_capital)
    result["position_value"] = notional_equity * positions.abs()

    if stop_loss is not None:
        aligned_stop = stop_loss.reindex(result.index).astype(float)
        result["stop_loss"] = aligned_stop
    else:
        result["stop_loss"] = np.nan

    signal_map = {1: "Long", -1: "Short", 0: "Flat"}
    result["signal"] = positions.round().astype(int).map(signal_map).fillna("Flat")

    rolling_max = result["equity_curve"].cummax()
    result["drawdown"] = result["equity_curve"] / rolling_max - 1.0
    result["cumulative_pnl"] = result["equity_curve"] - initial_capital
    result["net_pnl"] = result["cumulative_pnl"]

    return result


def performance_metrics(equity: pd.Series) -> Dict[str, float]:
    """Compute performance statistics from an equity curve."""
    if equity.empty:
        raise ValueError("Equity series must not be empty")

    equity = equity.astype(float)
    returns = equity.pct_change().fillna(0.0)

    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    periods = len(equity)
    annual_factor = 252
    periods_for_return = max(periods - 1, 1)
    annualized_return = (1.0 + total_return) ** (annual_factor / periods_for_return) - 1.0
    annualized_volatility = returns.std(ddof=0) * sqrt(annual_factor)
    sharpe_ratio = (
        annualized_return / annualized_volatility if annualized_volatility > 0 else np.nan
    )

    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1.0
    max_drawdown = float(drawdown.min())

    drawdown_durations = (drawdown != 0).astype(int)
    duration_groups = (drawdown_durations != drawdown_durations.shift()).cumsum()
    duration_stretch = drawdown_durations.groupby(duration_groups).cumsum()
    max_drawdown_duration = int(duration_stretch.max()) if not duration_stretch.empty else 0

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio) if np.isfinite(sharpe_ratio) else np.nan,
        "max_drawdown": max_drawdown,
        "max_drawdown_duration": max_drawdown_duration,
    }


__all__ = ["BacktestResult", "run_backtest", "performance_metrics"]
