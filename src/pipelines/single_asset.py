"""Single-asset backtesting pipeline that glues loaders, indicators, and strategies."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable, Dict, Mapping, MutableMapping, Optional, Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import pandas as pd

from ..data_loader import load_ohlcv_csv
from ..indicators import calculate_ema
from ..strategy_backtest.pipeline import BacktestOutputs, SignalBacktester
from ..strategy_backtest.registry import get_strategy
from ..utils.metrics import portfolio_metrics_from_returns

IndicatorFunction = Callable[..., pd.Series]


@dataclass(frozen=True)
class IndicatorConfig:
    """Describe how an indicator should be computed for the dataset."""

    name: str
    source_column: str = "close"
    target_column: Optional[str] = None
    params: Mapping[str, object] = field(default_factory=dict)

    def resolve_target_column(self) -> str:
        return self.target_column or self.name


@dataclass(frozen=True)
class SingleAssetPipelineConfig:
    """Configuration bundle for the single asset pipeline."""

    data_path: str
    strategy_name: str
    strategy_kwargs: Mapping[str, object] = field(default_factory=dict)
    horizon_bars: Optional[int] = None
    indicators: Sequence[IndicatorConfig] = field(default_factory=tuple)
    price_column: str = "close"
    external_signals: Optional["ExternalSignalModulationConfig"] = None


@dataclass(frozen=True)
class ExternalSignalModulationConfig:
    """Optional external signal that scales/gates the base strategy positions."""

    path: str
    value_column: str = "signal"
    timestamp_column: str = "time"
    clip_min: float = 0.0
    clip_max: float = 1.0
    block_threshold: float = 0.0


_INDICATOR_FUNCTIONS: Dict[str, IndicatorFunction] = {
    "ema": calculate_ema,
}


def _apply_indicators(data: pd.DataFrame, indicators: Sequence[IndicatorConfig]) -> pd.DataFrame:
    if not indicators:
        return data

    enriched = data.copy()
    for indicator in indicators:
        try:
            func = _INDICATOR_FUNCTIONS[indicator.name]
        except KeyError as exc:
            available = ", ".join(sorted(_INDICATOR_FUNCTIONS))
            raise KeyError(
                f"Indicator '{indicator.name}' tidak dikenal. Pilihan: {available}"
            ) from exc

        source_column = indicator.source_column
        if source_column not in enriched.columns:
            raise KeyError(f"Kolom sumber indikator '{source_column}' tidak ditemukan")

        series = enriched[source_column].astype(float)
        target = indicator.resolve_target_column()
        enriched[target] = func(series, **indicator.params)

    return enriched


def _load_external_signal(
    cfg: ExternalSignalModulationConfig, index: pd.DatetimeIndex
) -> pd.Series:
    series = pd.read_csv(cfg.path)
    if cfg.timestamp_column not in series.columns:
        raise KeyError(
            f"CSV eksternal harus memiliki kolom waktu '{cfg.timestamp_column}'"
        )
    if cfg.value_column not in series.columns:
        raise KeyError(
            f"CSV eksternal harus memiliki kolom sinyal '{cfg.value_column}'"
        )

    series[cfg.timestamp_column] = pd.to_datetime(
        series[cfg.timestamp_column], utc=True, errors="coerce"
    )
    series = series.dropna(subset=[cfg.timestamp_column])
    series = series.set_index(cfg.timestamp_column).sort_index()
    values = pd.to_numeric(series[cfg.value_column], errors="coerce")
    aligned = values.reindex(index).ffill()
    clipped = aligned.fillna(0.0).clip(lower=cfg.clip_min, upper=cfg.clip_max)
    return clipped.rename("position_scale")


def _apply_external_modulation(
    signals: pd.DataFrame,
    cfg: Optional[ExternalSignalModulationConfig],
    index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    if cfg is None:
        return signals, None

    scale = _load_external_signal(cfg, index)
    if signals.empty:
        return signals, scale

    blocked = scale.abs() <= cfg.block_threshold
    gated = signals.copy()
    for column in ("long_entry", "short_entry"):
        if column in gated.columns:
            gated[column] = gated[column] & ~blocked

    gated["position_scale"] = scale
    return gated, scale


def run_single_asset_pipeline(config: SingleAssetPipelineConfig) -> BacktestOutputs:
    """Load data, compute indicators, and execute the requested strategy."""

    df = load_ohlcv_csv(config.data_path)
    if config.horizon_bars is not None and config.horizon_bars > 0:
        df = df.tail(config.horizon_bars).copy()

    df = _apply_indicators(df, config.indicators)

    strategy = get_strategy(config.strategy_name, **dict(config.strategy_kwargs))
    raw_signals = strategy.generate_signals(df)
    signals, position_scale = _apply_external_modulation(
        raw_signals, config.external_signals, df.index
    )

    backtester = SignalBacktester(df, price_column=config.price_column)
    outputs = backtester.run(signals, position_scale=position_scale)
    return outputs


def save_backtest_outputs(
    outputs: BacktestOutputs,
    output_dir: str | Path,
    prefix: str = "backtest",
) -> Dict[str, Path]:
    """Persist metrics, trade log, and an equity plot for later reuse."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    standard_metrics = portfolio_metrics_from_returns(
        outputs.results["strategy_return"],
    )
    metrics_payload: MutableMapping[str, object] = {
        "metrics": outputs.metrics,
        "trade_summary": outputs.trade_summary,
        "standard_metrics": standard_metrics,
    }
    metrics_path = destination / f"{prefix}_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))

    trades_path = destination / f"{prefix}_trades.csv"
    outputs.trades.to_csv(trades_path, index=False)

    plot_path = destination / f"{prefix}_equity.png"
    _plot_equity_curve(outputs.results, plot_path)

    return {
        "metrics": metrics_path,
        "trades": trades_path,
        "plot": plot_path,
    }


def _plot_equity_curve(results: pd.DataFrame, path: Path) -> None:
    if results.empty:
        raise ValueError("Results frame is empty; cannot plot equity curve")

    fig = Figure(figsize=(10, 4))
    FigureCanvasAgg(fig)  # Ensure a non-interactive canvas is attached
    ax = fig.add_subplot(1, 1, 1)

    equity = results["equity_curve"]
    ax.plot(equity.index, equity.values, label="Equity Curve")
    ax.set_title("Backtest Equity Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity (relative)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)


__all__ = [
    "IndicatorConfig",
    "ExternalSignalModulationConfig",
    "SingleAssetPipelineConfig",
    "run_single_asset_pipeline",
    "save_backtest_outputs",
]
