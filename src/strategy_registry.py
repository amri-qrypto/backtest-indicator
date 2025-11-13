"""Central registry of strategy callables for quick experimentation."""
from __future__ import annotations

from typing import Callable, Dict, Iterable

import pandas as pd

from strategy import ema_vs_price_signals
from strategy_atr_filter import atr_filter_signals
from strategy_ema import ema_trend_signals
from strategy_macd import macd_signals
from strategy_oversold import oversold_signals

StrategyFunction = Callable[[pd.DataFrame], pd.Series]


def _ema_strategy(span: int) -> StrategyFunction:
    """Return a strategy function locking in a specific EMA ``span``."""

    def _strategy(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
        return ema_trend_signals(df, span=span, price_col=price_col)

    _strategy.__name__ = f"ema_{span}_signals"
    return _strategy


_STRATEGIES: Dict[str, StrategyFunction] = {
    "ema_vs_price": ema_vs_price_signals,
    "ema112": _ema_strategy(112),
    "ema50": _ema_strategy(50),
    "ema45": _ema_strategy(45),
    "macd": macd_signals,
    "atr_filter": atr_filter_signals,
    "oversold": oversold_signals,
}


def available_strategies() -> Iterable[str]:
    """Return an iterable of registered strategy names."""

    return tuple(sorted(_STRATEGIES))


def get_strategy(name: str) -> StrategyFunction:
    """Return the callable associated with ``name``."""

    try:
        return _STRATEGIES[name]
    except KeyError as exc:  # pragma: no cover - defensive programming
        choices = ", ".join(sorted(_STRATEGIES))
        raise KeyError(f"Unknown strategy '{name}'. Available strategies: {choices}") from exc


__all__ = ["StrategyFunction", "available_strategies", "get_strategy"]

