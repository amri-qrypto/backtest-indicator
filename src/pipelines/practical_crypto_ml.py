"""End-to-end pipeline for a practical crypto trading ML stack."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from ..data_loader import load_ohlcv_csv


# ---------------------------------------------------------------------------
# Configuration dataclasses


@dataclass(frozen=True)
class AuxiliarySourceConfig:
    """Describe an additional data source such as funding, order book, or sentiment."""

    path: Optional[str] = None
    prefix: str = ""
    columns: Optional[Sequence[str]] = None
    resample_rule: str = "1H"
    aggregation: str = "last"
    forward_fill_limit: Optional[int] = 6


@dataclass(frozen=True)
class MultiSourceDataConfig:
    """Bundle the required data sources for the ML pipeline."""

    ohlcv_path: str
    funding_rates: AuxiliarySourceConfig = field(default_factory=AuxiliarySourceConfig)
    order_book_depth: AuxiliarySourceConfig = field(default_factory=AuxiliarySourceConfig)
    on_chain_activity: AuxiliarySourceConfig = field(default_factory=AuxiliarySourceConfig)
    sentiment_scores: AuxiliarySourceConfig = field(default_factory=AuxiliarySourceConfig)
    resample_rule: str = "1H"
    fill_limit: int = 3
    clip_zscore: float = 5.0


@dataclass(frozen=True)
class LabelConfig:
    """Specify how training labels should be generated."""

    horizon_bars: int = 24
    task: str = "binary"  # binary | regression
    threshold: float = 0.0
    price_column: str = "close"


@dataclass(frozen=True)
class ModelStackConfig:
    """Toggle which models are trained and how CV is performed."""

    train_linear: bool = True
    train_tree_based: bool = True
    train_deep_learning: bool = False
    cv_splits: int = 5
    random_state: int = 7
    mlp_hidden_layers: Sequence[int] = (128, 64)
    mlp_activation: str = "relu"
    mlp_max_iter: int = 300


@dataclass(frozen=True)
class PortfolioConfig:
    """Define how ML scores are converted into positions."""

    top_k: int = 1
    long_short: bool = False
    max_leverage: float = 1.0
    volatility_target: float = 0.15
    turnover_limit: float = 1.5


@dataclass
class FeatureStore:
    """Keep feature metadata and the fitted scaler for reuse."""

    scaler: StandardScaler
    feature_names: List[str]

    @classmethod
    def fit(cls, features: pd.DataFrame) -> Tuple["FeatureStore", pd.DataFrame]:
        scaler = StandardScaler()
        transformed = scaler.fit_transform(features.values)
        normalized = pd.DataFrame(
            transformed,
            index=features.index,
            columns=features.columns,
        )
        return cls(scaler=scaler, feature_names=list(features.columns)), normalized

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [name for name in self.feature_names if name not in features.columns]
        if missing:
            raise KeyError(f"Missing features for transformation: {missing}")
        transformed = self.scaler.transform(features[self.feature_names].values)
        return pd.DataFrame(transformed, index=features.index, columns=self.feature_names)


@dataclass
class MLPipelineResult:
    """Artifacts produced by the ML pipeline run."""

    feature_store: FeatureStore
    feature_frame: pd.DataFrame
    labels: pd.Series
    models: Dict[str, object]
    cv_metrics: Dict[str, Mapping[str, float]]
    predictions: pd.DataFrame
    signals: pd.Series
    portfolio_weights: pd.Series
    realized_performance: Mapping[str, float]
    guardrail_flags: Mapping[str, bool]


# ---------------------------------------------------------------------------
# Loading and preprocessing helpers


def _load_auxiliary_source(
    config: AuxiliarySourceConfig, base_index: pd.DatetimeIndex
) -> pd.DataFrame:
    if not config.path:
        return pd.DataFrame(index=base_index)

    df = pd.read_csv(config.path)
    if "time" not in df.columns:
        raise ValueError(f"Auxiliary file {config.path} must contain a 'time' column")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()

    keep_columns = [col for col in df.columns if col != "time"]
    if config.columns:
        keep_columns = [col for col in config.columns if col in df.columns]
    df = df[keep_columns]

    if config.resample_rule:
        df = df.resample(config.resample_rule).agg(config.aggregation)

    prefixed = df.add_prefix(config.prefix)
    aligned = prefixed.reindex(base_index)
    aligned = aligned.ffill(limit=config.forward_fill_limit)
    return aligned


def _clip_outliers_zscore(df: pd.DataFrame, zscore: float) -> pd.DataFrame:
    if not np.isfinite(zscore) or zscore <= 0:
        return df

    clipped = df.copy()
    for column in clipped.columns:
        series = clipped[column]
        mean = series.mean()
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            continue
        upper = mean + zscore * std
        lower = mean - zscore * std
        clipped[column] = series.clip(lower, upper)
    return clipped


def _engineer_base_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=ohlcv.index)
    close = ohlcv["close"].astype(float)
    volume = ohlcv.get("volume", pd.Series(index=ohlcv.index, data=np.nan)).astype(float)

    returns_1h = close.pct_change(1)
    returns_4h = close.pct_change(4)
    returns_24h = close.pct_change(24)
    features["ret_1h"] = returns_1h
    features["ret_4h"] = returns_4h
    features["ret_24h"] = returns_24h
    features["momentum_24h"] = returns_1h.rolling(24).mean()
    features["momentum_7d"] = returns_1h.rolling(24 * 7).mean()
    features["vol_24h"] = returns_1h.rolling(24).std()
    features["vol_7d"] = returns_1h.rolling(24 * 7).std()
    features["volume_change"] = volume.pct_change()
    features["volume_zscore_7d"] = (volume - volume.rolling(24 * 7).mean()) / volume.rolling(24 * 7).std()
    features["price_distance_ema_24"] = close / close.ewm(span=24, adjust=False).mean() - 1
    features["price_distance_ema_96"] = close / close.ewm(span=96, adjust=False).mean() - 1

    return features


def _merge_sources(
    ohlcv: pd.DataFrame, data_cfg: MultiSourceDataConfig
) -> pd.DataFrame:
    base_features = _engineer_base_features(ohlcv)

    funding = _load_auxiliary_source(data_cfg.funding_rates, ohlcv.index)
    if not funding.empty:
        base_features = base_features.join(funding, how="left")

    order_book = _load_auxiliary_source(data_cfg.order_book_depth, ohlcv.index)
    if not order_book.empty:
        base_features = base_features.join(order_book, how="left")

    on_chain = _load_auxiliary_source(data_cfg.on_chain_activity, ohlcv.index)
    if not on_chain.empty:
        base_features = base_features.join(on_chain, how="left")

    sentiment = _load_auxiliary_source(data_cfg.sentiment_scores, ohlcv.index)
    if not sentiment.empty:
        base_features = base_features.join(sentiment, how="left")

    base_features = _clip_outliers_zscore(base_features, data_cfg.clip_zscore)
    base_features = base_features.ffill(limit=data_cfg.fill_limit).dropna()
    return base_features


# ---------------------------------------------------------------------------
# Label construction


def build_labels(data: pd.DataFrame, config: LabelConfig) -> pd.Series:
    price = data[config.price_column]
    forward_returns = price.pct_change(config.horizon_bars).shift(-config.horizon_bars)

    if config.task == "binary":
        labels = (forward_returns > config.threshold).astype(int)
    else:
        labels = forward_returns

    return labels.loc[data.index]


# ---------------------------------------------------------------------------
# Model stack and evaluation


def _train_and_score_model(
    model_name: str,
    model,
    features: pd.DataFrame,
    labels: pd.Series,
    cv: TimeSeriesSplit,
) -> Tuple[object, Mapping[str, float], pd.Series]:
    cv_accuracy: List[float] = []
    cv_auc: List[float] = []
    oof_pred = pd.Series(index=features.index, dtype=float, name=model_name)

    for train_idx, test_idx in cv.split(features):
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

        model.fit(X_train, y_train)
        probs = _predict_proba(model, X_test)
        oof_pred.iloc[test_idx] = probs

        cv_accuracy.append(accuracy_score(y_test, (probs > 0.5).astype(int)))
        try:
            auc = roc_auc_score(y_test, probs)
            cv_auc.append(auc)
        except ValueError:
            cv_auc.append(np.nan)

    metrics = {
        "accuracy": float(np.nanmean(cv_accuracy)),
        "roc_auc": float(np.nanmean(cv_auc)),
    }
    model.fit(features, labels)
    return model, metrics, oof_pred


def _predict_proba(model, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        decision = model.decision_function(features)
        return 1 / (1 + np.exp(-decision))
    preds = model.predict(features)
    return preds.astype(float)


def train_model_stack(
    features: pd.DataFrame,
    labels: pd.Series,
    config: ModelStackConfig,
) -> Tuple[Dict[str, object], Dict[str, Mapping[str, float]], pd.DataFrame]:
    models: Dict[str, object] = {}
    cv_metrics: Dict[str, Mapping[str, float]] = {}
    predictions = pd.DataFrame(index=features.index)

    cv = TimeSeriesSplit(n_splits=config.cv_splits)

    if config.train_linear:
        linear_model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=1000,
            random_state=config.random_state,
        )
        model, metrics, preds = _train_and_score_model(
            "logistic_l1", linear_model, features, labels, cv
        )
        models["logistic_l1"] = model
        cv_metrics["logistic_l1"] = metrics
        predictions["logistic_l1"] = preds

    if config.train_tree_based:
        gbdt = GradientBoostingClassifier(random_state=config.random_state)
        model, metrics, preds = _train_and_score_model(
            "gbdt", gbdt, features, labels, cv
        )
        models["gbdt"] = model
        cv_metrics["gbdt"] = metrics
        predictions["gbdt"] = preds

    if config.train_deep_learning:
        hidden_layers = tuple(int(layer) for layer in config.mlp_hidden_layers)
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=config.mlp_activation,
            solver="adam",
            learning_rate_init=0.001,
            max_iter=config.mlp_max_iter,
            random_state=config.random_state,
        )
        model, metrics, preds = _train_and_score_model(
            "mlp", mlp, features, labels, cv
        )
        models["mlp"] = model
        cv_metrics["mlp"] = metrics
        predictions["mlp"] = preds

    if predictions.empty:
        raise ValueError("Model stack is empty. Enable at least one model type")

    return models, cv_metrics, predictions


# ---------------------------------------------------------------------------
# Signal generation and portfolio construction


def _combine_predictions(predictions: pd.DataFrame) -> pd.Series:
    ensemble = predictions.mean(axis=1)
    return 2 * ensemble - 1  # map probability to [-1, 1]


def _build_portfolio(
    signals: pd.Series,
    config: PortfolioConfig,
) -> Tuple[pd.Series, Mapping[str, bool]]:
    guardrails: MutableMapping[str, bool] = {
        "max_leverage": False,
        "turnover": False,
    }

    latest_signals = signals.dropna()
    if latest_signals.empty:
        return pd.Series(dtype=float), guardrails

    ranked = latest_signals.rank(pct=True)
    positions = pd.Series(0.0, index=latest_signals.index)
    universe_size = len(ranked)
    bucket = max(1, min(config.top_k, universe_size))

    if config.long_short:
        long_cutoff = ranked.quantile(1 - bucket / universe_size)
        short_cutoff = ranked.quantile(bucket / universe_size)
        positions[ranked >= long_cutoff] = 1 / bucket
        positions[ranked <= short_cutoff] = -1 / bucket
    else:
        threshold = ranked.quantile(1 - bucket / universe_size)
        winners = ranked >= threshold
        positions[winners] = 1 / winners.sum() if winners.sum() else 0

    gross_leverage = positions.abs().sum()
    if gross_leverage > config.max_leverage and gross_leverage > 0:
        positions *= config.max_leverage / gross_leverage
        guardrails["max_leverage"] = True

    turnover = positions.diff().abs().sum()
    if turnover > config.turnover_limit:
        scale = config.turnover_limit / turnover
        positions *= scale
        guardrails["turnover"] = True

    return positions, guardrails


def _evaluate_realized_performance(
    signals: pd.Series,
    labels: pd.Series,
    horizon_bars: int,
    task: str,
) -> Mapping[str, float]:
    aligned_labels = labels.loc[signals.index]
    if task == "binary":
        target_direction = aligned_labels.replace({0: -1, 1: 1})
        accuracy = (np.sign(signals) == target_direction).mean()
        extra = {"directional_accuracy": float(accuracy)}
    else:
        correlation = float(signals.corr(aligned_labels)) if aligned_labels.std() else 0.0
        extra = {"signal_label_corr": correlation}

    sharpe = _information_ratio(signals, horizon_bars)
    payload = {"signal_information_ratio": float(sharpe)}
    payload.update(extra)
    return payload


def _information_ratio(signals: pd.Series, horizon_bars: int) -> float:
    hourly_factor = np.sqrt(24 * 365 / max(horizon_bars, 1))
    mean = signals.mean()
    std = signals.std(ddof=0)
    if std == 0 or np.isnan(std):
        return 0.0
    return mean / std * hourly_factor


# ---------------------------------------------------------------------------
# High-level orchestrator


def run_practical_crypto_ml_pipeline(
    data_cfg: MultiSourceDataConfig,
    label_cfg: LabelConfig,
    model_cfg: ModelStackConfig,
    portfolio_cfg: PortfolioConfig,
) -> MLPipelineResult:
    """Execute the multi-source ML pipeline and return rich artifacts."""

    ohlcv = load_ohlcv_csv(data_cfg.ohlcv_path)
    merged_features = _merge_sources(ohlcv, data_cfg)

    labels = build_labels(ohlcv, label_cfg)
    labels = labels.loc[merged_features.index].dropna()
    merged_features = merged_features.loc[labels.index]

    feature_store, normalized = FeatureStore.fit(merged_features)
    models, cv_metrics, oof_predictions = train_model_stack(
        normalized, labels, model_cfg
    )
    signals = _combine_predictions(oof_predictions)
    portfolio_weights, guardrails = _build_portfolio(signals, portfolio_cfg)
    realized_performance = _evaluate_realized_performance(
        signals, labels, label_cfg.horizon_bars, label_cfg.task
    )

    return MLPipelineResult(
        feature_store=feature_store,
        feature_frame=merged_features,
        labels=labels,
        models=models,
        cv_metrics=cv_metrics,
        predictions=oof_predictions,
        signals=signals,
        portfolio_weights=portfolio_weights,
        realized_performance=realized_performance,
        guardrail_flags=guardrails,
    )


__all__ = [
    "AuxiliarySourceConfig",
    "MultiSourceDataConfig",
    "LabelConfig",
    "ModelStackConfig",
    "PortfolioConfig",
    "FeatureStore",
    "MLPipelineResult",
    "run_practical_crypto_ml_pipeline",
]

