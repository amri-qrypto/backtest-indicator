"""Mean reversion strategy using oversold/overbought hyperwave oscillators."""
from __future__ import annotations

import pandas as pd

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


__all__ = ["oversold_signals"]
