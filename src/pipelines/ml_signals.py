"""Inference helpers that convert saved ML models into trading signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from ..backtest import performance_metrics, run_backtest
from ..performance.time_utils import DEFAULT_BARS_PER_YEAR
from ..trade_analysis import generate_trade_log, summarise_trades
from .practical_crypto_ml import FeatureStore


@dataclass(frozen=True)
class MLSignalInferenceConfig:
    """Configuration for loading a trained model and running inference."""

    model_path: str
    feature_store_path: Optional[str] = None
    feature_columns: Optional[Sequence[str]] = None
    price_column: str = "close"
    initial_capital: float = 10000.0
    trading_cost_bps: float = 0.0
    probability_floor: float = 0.0
    probability_cap: float = 1.0

    def resolve_model_path(self) -> Path:
        return Path(self.model_path)

    def resolve_feature_store_path(self) -> Optional[Path]:
        if self.feature_store_path is None:
            return None
        return Path(self.feature_store_path)


@dataclass(frozen=True)
class MLSignalWeightingConfig:
    """Control how raw ML scores are converted into portfolio weights."""

    mode: str = "threshold"  # "threshold" | "risk_normalized"
    entry_threshold: float = 0.1
    target_volatility: float = 0.01
    volatility_lookback: int = 24
    max_abs_position: float = 1.0


@dataclass
class MLSignalBacktestResult:
    """Artifacts returned by :func:`run_ml_signal_backtest`."""

    probabilities: pd.Series
    signals: pd.Series
    positions: pd.Series
    backtest: pd.DataFrame
    metrics: Mapping[str, float]
    trade_log: pd.DataFrame
    trade_summary: Mapping[str, float]
    monitoring_artifacts: Mapping[str, pd.DataFrame]


def _load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path!s}")
    return joblib.load(path)


def _load_feature_store(path: Optional[Path]) -> Optional[FeatureStore]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"FeatureStore file not found at {path!s}")
    store = joblib.load(path)
    if not isinstance(store, FeatureStore):
        raise TypeError(
            "Loaded object is not a FeatureStore. Ensure you saved the correct artifact."
        )
    return store


def _prepare_features(
    features: pd.DataFrame,
    config: MLSignalInferenceConfig,
    feature_store: Optional[FeatureStore],
) -> pd.DataFrame:
    if feature_store is not None:
        return feature_store.transform(features)

    if config.feature_columns is not None:
        missing = [col for col in config.feature_columns if col not in features.columns]
        if missing:
            raise KeyError(f"Missing columns for inference: {missing}")
        return features.loc[:, list(config.feature_columns)]

    return features.copy()


def _predict_probabilities(model, features: pd.DataFrame) -> pd.Series:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features.values)[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(features.values)
        probs = 1.0 / (1.0 + np.exp(-decision))
    else:
        preds = model.predict(features.values)
        probs = preds.astype(float)

    return pd.Series(probs, index=features.index, name="prob_up")


def _threshold_positions(signals: pd.Series, cfg: MLSignalWeightingConfig) -> pd.Series:
    positions = pd.Series(0.0, index=signals.index)
    active = signals >= cfg.entry_threshold
    positions.loc[active] = cfg.max_abs_position
    return positions


def _risk_normalized_positions(
    signals: pd.Series, price_series: pd.Series, cfg: MLSignalWeightingConfig
) -> pd.Series:
    returns = price_series.pct_change().fillna(0.0)
    realized_vol = returns.rolling(cfg.volatility_lookback).std(ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaling = cfg.target_volatility / realized_vol
    scaled = signals * scaling
    clipped = scaled.clip(lower=-cfg.max_abs_position, upper=cfg.max_abs_position)
    return clipped.fillna(0.0)


def _apply_weighting(
    signals: pd.Series,
    price_series: pd.Series,
    cfg: Optional[MLSignalWeightingConfig],
) -> pd.Series:
    if cfg is None:
        cfg = MLSignalWeightingConfig()

    mode = cfg.mode.lower()
    if mode == "threshold":
        return _threshold_positions(signals, cfg)
    if mode == "risk_normalized":
        return _risk_normalized_positions(signals, price_series, cfg)

    raise ValueError(
        "Unknown weighting mode. Use 'threshold' or 'risk_normalized'."
    )


def _clip_probabilities(series: pd.Series, config: MLSignalInferenceConfig) -> pd.Series:
    floor = config.probability_floor
    cap = config.probability_cap
    clipped = series
    if floor is not None:
        clipped = clipped.clip(lower=floor)
    if cap is not None:
        clipped = clipped.clip(upper=cap)
    return clipped


def _align_inputs(market_data: pd.DataFrame, features: pd.DataFrame) -> pd.DatetimeIndex:
    if market_data.empty or features.empty:
        raise ValueError("Market data and features must not be empty")
    index = market_data.index.intersection(features.index)
    if index.empty:
        raise ValueError("No overlapping timestamps between market data and features")
    return index


def run_ml_signal_backtest(
    market_data: pd.DataFrame,
    features: pd.DataFrame,
    model_config: MLSignalInferenceConfig,
    weighting: Optional[MLSignalWeightingConfig] = None,
) -> MLSignalBacktestResult:
    """Load a saved model, compute signals, and evaluate them via backtest."""

    model = _load_model(model_config.resolve_model_path())
    feature_store = _load_feature_store(model_config.resolve_feature_store_path())

    index = _align_inputs(market_data, features)
    features = features.loc[index]
    price_series = market_data.loc[index, model_config.price_column].astype(float)

    prepared = _prepare_features(features, model_config, feature_store)
    probabilities = _clip_probabilities(_predict_probabilities(model, prepared), model_config)
    signals = 2.0 * probabilities - 1.0

    positions = _apply_weighting(signals, price_series, weighting)

    price_frame = pd.DataFrame({"close": price_series})
    backtest = run_backtest(
        price_frame,
        positions,
        initial_capital=model_config.initial_capital,
        trading_cost_bps=model_config.trading_cost_bps,
    )

    metrics = performance_metrics(
        backtest["equity_curve"], bars_per_year=DEFAULT_BARS_PER_YEAR
    )
    trade_log = generate_trade_log(backtest, initial_capital=model_config.initial_capital)
    trade_summary = asdict(summarise_trades(trade_log))

    monitoring: Dict[str, pd.DataFrame] = {
        "prediction_log": pd.DataFrame(
            {
                "prob_up": probabilities,
                "signal": signals,
                "position": positions,
            }
        ),
        "pnl": backtest[["strategy_ret", "equity_curve", "drawdown"]].copy(),
        "turnover": positions.diff().abs().rename("turnover").to_frame(),
    }

    return MLSignalBacktestResult(
        probabilities=probabilities,
        signals=signals,
        positions=positions,
        backtest=backtest,
        metrics=metrics,
        trade_log=trade_log,
        trade_summary=trade_summary,
        monitoring_artifacts=monitoring,
    )


__all__ = [
    "MLSignalInferenceConfig",
    "MLSignalWeightingConfig",
    "MLSignalBacktestResult",
    "run_ml_signal_backtest",
]
