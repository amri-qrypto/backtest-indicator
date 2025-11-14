"""Registry of TradingView signal strategies."""
from __future__ import annotations

from importlib import import_module
from typing import Dict, Iterable, Type

from .base import StrategyBase

_STRATEGY_MODULES = {
    "ema112_atr": "src.strategy_backtest.strategies.ema112_atr",
    "vwap": "src.strategy_backtest.strategies.vwap",
}

_CACHE: Dict[str, Type[StrategyBase]] = {}


def _load_strategy_class(name: str) -> Type[StrategyBase]:
    try:
        module_path = _STRATEGY_MODULES[name]
    except KeyError as exc:  # pragma: no cover - defensive mapping lookup
        available = ", ".join(sorted(_STRATEGY_MODULES))
        raise KeyError(f"Strategi '{name}' tidak terdaftar. Pilihan: {available}") from exc

    if name not in _CACHE:
        module = import_module(module_path)
        if not hasattr(module, "Strategy"):
            raise AttributeError(f"Modul {module_path} tidak mendefinisikan kelas 'Strategy'")
        strategy_cls = getattr(module, "Strategy")
        if not issubclass(strategy_cls, StrategyBase):
            raise TypeError(f"Strategy '{name}' harus mewarisi StrategyBase")
        _CACHE[name] = strategy_cls
    return _CACHE[name]


def get_strategy(name: str, **kwargs) -> StrategyBase:
    """Instantiate a registered strategy by ``name``."""

    strategy_cls = _load_strategy_class(name)
    return strategy_cls(**kwargs)


def list_strategies() -> Iterable[str]:
    """Return all registered strategy names."""

    return tuple(sorted(_STRATEGY_MODULES))


__all__ = ["get_strategy", "list_strategies"]
