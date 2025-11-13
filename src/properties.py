"""Configuration objects describing a backtest setup."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(slots=True)
class StrategyProperties:
    """Collect frequently used parameters for a strategy run.

    The object centralises configuration that is typically adjusted between experiments,
    such as the traded symbol, timeframe resolution, starting capital, and trading costs.
    """

    symbol: str
    timeframe: str = "1H"
    initial_capital: float = 10_000.0
    trading_cost_bps: float = 0.0
    stop_loss_pct: Optional[float] = None
    position_size: Optional[float] = None
    notes: Dict[str, str] = field(default_factory=dict)

    def to_kwargs(self) -> Dict[str, object]:
        """Return a dictionary suitable for unpacking into ``run_backtest``."""

        return {
            "initial_capital": self.initial_capital,
            "trading_cost_bps": self.trading_cost_bps,
        }


__all__ = ["StrategyProperties"]

