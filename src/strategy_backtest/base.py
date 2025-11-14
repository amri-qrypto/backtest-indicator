"""Base classes for reusable TradingView signal strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, MutableMapping

import pandas as pd


@dataclass
class StrategyMetadata:
    """Human-readable description of a registered strategy."""

    name: str
    description: str
    entry: str
    exit: str
    parameters: MutableMapping[str, float] = field(default_factory=dict)
    context_columns: Iterable[str] = field(default_factory=tuple)


class StrategyBase(ABC):
    """Abstract base class for strategies that emit entry/exit signals."""

    metadata: StrategyMetadata

    def __init__(self, **kwargs: float) -> None:
        self.params: Dict[str, float] = dict(kwargs)

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame containing long/short entry & exit signals."""

    def describe(self) -> Mapping[str, object]:
        """Return metadata merged with the current parameter set."""

        info = {
            "name": self.metadata.name,
            "description": self.metadata.description,
            "entry": self.metadata.entry,
            "exit": self.metadata.exit,
        }
        info.update({f"param_{key}": value for key, value in self.params.items()})
        return info


__all__ = ["StrategyBase", "StrategyMetadata"]
