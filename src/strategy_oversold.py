"""Mean reversion strategy using oversold/overbought hyperwave oscillators."""
from __future__ import annotations

import pandas as pd

from .strategy_backtest.base import StrategyBase, StrategyMetadata
from .strategy_backtest.utils import build_long_only_signals

DEFAULT_ENTRY_COLUMN = "Oversold HWO Up"
DEFAULT_EXIT_COLUMN = "Overbought HWO Down"


def oversold_signals(
    df: pd.DataFrame,
    entry_column: str = DEFAULT_ENTRY_COLUMN,
    exit_column: str = DEFAULT_EXIT_COLUMN,
) -> pd.Series:
    """Generate a long/cash signal series from oversold/overbought markers."""
    missing = [col for col in (entry_column, exit_column) if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required oversold columns: {missing}")

    entry_signal = pd.to_numeric(df[entry_column], errors="coerce").fillna(0.0)
    exit_signal = pd.to_numeric(df[exit_column], errors="coerce").fillna(0.0)

    positions = []
    current_position = 0.0
    for entry, exit_ in zip(entry_signal, exit_signal):
        if entry >= 1.0:
            current_position = 1.0
        elif exit_ >= 1.0:
            current_position = 0.0
        positions.append(current_position)

    series = pd.Series(positions, index=df.index, dtype=float, name="position")
    return series


class Strategy(StrategyBase):
    """Mean reversion strategy driven by oversold/overbought markers."""

    metadata = StrategyMetadata(
        name="oversold",
        description="Membeli ketika indikator oversold aktif dan keluar ketika overbought muncul.",
        entry="Entry long ketika kolom oversold bernilai >= 1.",
        exit="Exit ketika kolom overbought bernilai >= 1.",
        parameters={
            "entry_column": DEFAULT_ENTRY_COLUMN,
            "exit_column": DEFAULT_EXIT_COLUMN,
        },
        context_columns=("entry_signal", "exit_signal", "position"),
    )

    def __init__(
        self,
        entry_column: str = DEFAULT_ENTRY_COLUMN,
        exit_column: str = DEFAULT_EXIT_COLUMN,
    ) -> None:
        super().__init__(entry_column=entry_column, exit_column=exit_column)
        self.entry_column = entry_column
        self.exit_column = exit_column

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        missing = [
            column
            for column in (self.entry_column, self.exit_column)
            if column not in data.columns
        ]
        if missing:
            raise KeyError(f"DataFrame missing required oversold columns: {missing}")

        entry_signal = pd.to_numeric(data[self.entry_column], errors="coerce").fillna(0.0)
        exit_signal = pd.to_numeric(data[self.exit_column], errors="coerce").fillna(0.0)
        positions = oversold_signals(data, self.entry_column, self.exit_column)
        long_condition = positions.astype(float) > 0.5

        long_entry, long_exit, short_entry, short_exit, position = build_long_only_signals(
            long_condition
        )

        return pd.DataFrame(
            {
                "long_entry": long_entry,
                "long_exit": long_exit,
                "short_entry": short_entry,
                "short_exit": short_exit,
                "entry_signal": entry_signal,
                "exit_signal": exit_signal,
                "position": position,
            },
            index=data.index,
        )


__all__ = ["oversold_signals", "Strategy"]
