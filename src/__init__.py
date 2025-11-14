"""Convenience imports for the backtest toolkit."""

from .backtest import BacktestResult, performance_metrics, run_backtest
from .data_loader import load_ohlcv_csv
from .properties import StrategyProperties
from .strategy import ema_vs_price_signals
from .strategy_ema import ema_trend_signals
from .trade_analysis import TradeSummary, failed_entries, generate_trade_log, summarise_trades
from .strategy_registry import StrategyFunction, available_strategies, get_strategy

__all__ = [
    "BacktestResult",
    "performance_metrics",
    "run_backtest",
    "load_ohlcv_csv",
    "StrategyProperties",
    "ema_vs_price_signals",
    "ema_trend_signals",
    "TradeSummary",
    "failed_entries",
    "generate_trade_log",
    "summarise_trades",
    "StrategyFunction",
    "available_strategies",
    "get_strategy",
]

