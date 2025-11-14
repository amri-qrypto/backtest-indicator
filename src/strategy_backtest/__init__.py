"""Signal-driven backtesting helpers for TradingView CSV exports."""

from .base import StrategyBase, StrategyMetadata
from .registry import get_strategy, list_strategies
from .pipeline import BacktestOutputs, SignalBacktester
from .utils import load_strategy_csv, sanitise_column_name, sanitise_columns

__all__ = [
    "StrategyBase",
    "StrategyMetadata",
    "get_strategy",
    "list_strategies",
    "BacktestOutputs",
    "SignalBacktester",
    "load_strategy_csv",
    "sanitise_column_name",
    "sanitise_columns",
]
