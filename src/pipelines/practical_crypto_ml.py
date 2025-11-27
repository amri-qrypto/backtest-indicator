"""End-to-end pipeline for a practical crypto trading ML stack."""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
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
class FeatureEngineeringConfig:
    """Control which optional features are engineered from available sources."""

    realized_vol_window: Optional[int] = None
    funding_skew_window: Optional[int] = None
    funding_skew_column: Optional[str] = None
    order_book_imbalance_window: Optional[int] = None
    order_book_bid_column: str = "bid_volume"
    order_book_ask_column: str = "ask_volume"


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
    feature_engineering: FeatureEngineeringConfig = field(
        default_factory=FeatureEngineeringConfig
    )


@dataclass(frozen=True)
class LabelConfig:
    """Specify how training labels should be generated."""

    horizon_bars: int = 24
    task: str = "binary"  # binary | regression
    threshold: float = 0.0
    price_column: str = "close"
    rolling_vol_window: Optional[int] = None
    rolling_vol_multiplier: float = 1.0


@dataclass(frozen=True)
class ModelStackConfig:
    """Toggle which models are trained and how CV is performed."""

    train_linear: bool = True
    train_logistic_elasticnet: bool = False
    train_probit: bool = False
    train_sgd: bool = False
    train_tree_based: bool = True
    train_deep_learning: bool = False
    cv_splits: int = 5
    random_state: int = 7
    logistic_l1_cs: Sequence[float] = (0.5, 1.0, 2.0)
    logistic_max_iter: int = 1200
    logistic_tol: float = 1e-4
    logistic_elasticnet_cs: Sequence[float] = (0.2, 1.0)
    logistic_elasticnet_l1_ratios: Sequence[float] = (0.2, 0.5, 0.8)
    logistic_elasticnet_max_iter: int = 2000
    sgd_losses: Sequence[str] = ("log_loss", "hinge")
    sgd_alphas: Sequence[float] = (0.0001, 0.001)
    sgd_max_iter: int = 1000
    sgd_early_stopping: bool = True
    sgd_n_iter_no_change: int = 5
    sgd_validation_fraction: float = 0.1
    sgd_tol: float = 1e-3
    mlp_hidden_layers: Sequence[int] = (128, 64)
    mlp_activation: str = "relu"
    mlp_max_iter: int = 300
    mlp_alpha: float = 1e-4
    mlp_beta1: float = 0.9
    mlp_beta2: float = 0.999
    mlp_validation_fraction: float = 0.1
    mlp_early_stopping: bool = True


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
    cv_report: Dict[str, Mapping[str, object]]
    cv_artifacts: Mapping[str, Path]
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


def _engineer_base_features(
    ohlcv: pd.DataFrame,
    feature_cfg: FeatureEngineeringConfig,
    auxiliary_sources: Optional[Mapping[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    features = pd.DataFrame(index=ohlcv.index)
    close = ohlcv["close"].astype(float)
    volume = ohlcv.get("volume", pd.Series(index=ohlcv.index, data=np.nan)).astype(float)
    auxiliary_sources = auxiliary_sources or {}

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

    if feature_cfg.realized_vol_window and feature_cfg.realized_vol_window > 0:
        high = ohlcv.get("high", pd.Series(index=ohlcv.index, data=np.nan)).astype(float)
        low = ohlcv.get("low", pd.Series(index=ohlcv.index, data=np.nan)).astype(float)
        intrabar_range = np.log(high / low).replace([np.inf, -np.inf], np.nan)
        features["realized_vol_intrabar"] = intrabar_range.rolling(
            feature_cfg.realized_vol_window
        ).std()

    funding = auxiliary_sources.get("funding")
    if (
        feature_cfg.funding_skew_window
        and feature_cfg.funding_skew_window > 0
        and funding is not None
        and not funding.empty
    ):
        if feature_cfg.funding_skew_column and feature_cfg.funding_skew_column in funding.columns:
            funding_series = funding[feature_cfg.funding_skew_column]
        else:
            funding_series = funding.mean(axis=1)
        features["funding_skew"] = funding_series
        features[f"funding_skew_roll_{feature_cfg.funding_skew_window}"] = funding_series.rolling(
            feature_cfg.funding_skew_window
        ).mean()

    order_book = auxiliary_sources.get("order_book")
    if (
        feature_cfg.order_book_imbalance_window
        and feature_cfg.order_book_imbalance_window > 0
        and order_book is not None
        and not order_book.empty
    ):
        bid_volume = order_book.get(feature_cfg.order_book_bid_column)
        ask_volume = order_book.get(feature_cfg.order_book_ask_column)
        if bid_volume is not None and ask_volume is not None:
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume).replace(0, np.nan)
            features["order_book_imbalance"] = imbalance
            features[
                f"order_book_imbalance_roll_{feature_cfg.order_book_imbalance_window}"
            ] = imbalance.rolling(feature_cfg.order_book_imbalance_window).mean()

    return features


def _merge_sources(
    ohlcv: pd.DataFrame, data_cfg: MultiSourceDataConfig
) -> pd.DataFrame:
    funding = _load_auxiliary_source(data_cfg.funding_rates, ohlcv.index)
    order_book = _load_auxiliary_source(data_cfg.order_book_depth, ohlcv.index)
    on_chain = _load_auxiliary_source(data_cfg.on_chain_activity, ohlcv.index)
    sentiment = _load_auxiliary_source(data_cfg.sentiment_scores, ohlcv.index)

    base_features = _engineer_base_features(
        ohlcv,
        data_cfg.feature_engineering,
        auxiliary_sources={"funding": funding, "order_book": order_book},
    )
    if not funding.empty:
        base_features = base_features.join(funding, how="left")
    if not order_book.empty:
        base_features = base_features.join(order_book, how="left")
    if not on_chain.empty:
        base_features = base_features.join(on_chain, how="left")
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

    threshold: float | pd.Series = config.threshold
    if config.rolling_vol_window and config.rolling_vol_window > 0:
        realized_vol = price.pct_change().rolling(config.rolling_vol_window).std()
        threshold = realized_vol * config.rolling_vol_multiplier + config.threshold

    if config.task == "binary":
        labels = (forward_returns > threshold).astype(int)
    else:
        labels = forward_returns

    return labels.loc[data.index]


def _population_stability_index(
    train: pd.Series, test: pd.Series, *, bins: int = 10
) -> float:
    combined = pd.concat([train, test]).dropna()
    if combined.empty:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.nanquantile(combined, quantiles))
    if len(edges) < 2:
        return 0.0

    train_counts, _ = np.histogram(train.dropna(), bins=edges)
    test_counts, _ = np.histogram(test.dropna(), bins=edges)

    train_dist = train_counts / max(train_counts.sum(), 1)
    test_dist = test_counts / max(test_counts.sum(), 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        psi = (test_dist - train_dist) * np.log(np.divide(test_dist, train_dist))
    psi = np.nan_to_num(psi, nan=0.0, posinf=0.0, neginf=0.0)
    return float(psi.sum())


def _serialize_label_distribution(labels: pd.Series) -> Dict[str, float]:
    counts = labels.value_counts(dropna=False)
    total = counts.sum()
    return {str(idx): float(val / total) if total else 0.0 for idx, val in counts.items()}


def _replace_nan(value):
    if isinstance(value, dict):
        return {key: _replace_nan(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_replace_nan(item) for item in value]
    if isinstance(value, float) or isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    return value


def _export_cv_report(
    cv_report: Mapping[str, Mapping[str, object]], output_path: str | Path
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = _replace_nan(cv_report)
    path.write_text(json.dumps(cleaned, indent=2, sort_keys=True))
    return path


def _cv_stability_table(cv_report: Mapping[str, Mapping[str, object]]) -> pd.DataFrame:
    records: List[Mapping[str, object]] = []
    for model, report in cv_report.items():
        for fold in report.get("folds", []):
            drift = fold.get("drift", {})
            records.append(
                {
                    "model": model,
                    "fold": fold.get("fold"),
                    "average_psi": drift.get("average_psi", 0.0),
                    "max_psi": drift.get("max_psi", 0.0),
                }
            )
    return pd.DataFrame.from_records(records)


def _plot_cv_stability(
    cv_report: Mapping[str, Mapping[str, object]], output_path: str | Path
) -> Optional[Path]:
    table = _cv_stability_table(cv_report)
    if table.empty:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    for model, group in table.groupby("model"):
        plt.plot(group["fold"], group["average_psi"], marker="o", label=model)
    plt.axhline(0.1, color="red", linestyle="--", linewidth=1, label="PSI 0.1")
    plt.xlabel("Fold")
    plt.ylabel("Average PSI")
    plt.title("CV Feature Stability (Train vs Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def _export_cv_artifacts(
    cv_report: Mapping[str, Mapping[str, object]],
    report_path: Optional[str | Path],
    plot_path: Optional[str | Path],
) -> Dict[str, Path]:
    artifacts: Dict[str, Path] = {}
    if report_path is not None:
        artifacts["report"] = _export_cv_report(cv_report, report_path)
    if plot_path is not None:
        plotted = _plot_cv_stability(cv_report, plot_path)
        if plotted is not None:
            artifacts["stability_plot"] = plotted
    return artifacts


# ---------------------------------------------------------------------------
# Model stack and evaluation


def _train_and_score_model(
    model_name: str,
    model,
    features: pd.DataFrame,
    labels: pd.Series,
    cv: TimeSeriesSplit,
    task: str,
) -> Tuple[object, Mapping[str, float], pd.Series, Mapping[str, object]]:
    cv_accuracy: List[float] = []
    cv_auc: List[float] = []
    cv_mae: List[float] = []
    cv_r2: List[float] = []
    oof_pred = pd.Series(index=features.index, dtype=float, name=model_name)
    fold_details: List[Dict[str, object]] = []

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(features)):
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

        model.fit(X_train, y_train)
        fold_payload: Dict[str, object] = {
            "fold": fold_id,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "train_range": {
                "start": str(X_train.index.min()),
                "end": str(X_train.index.max()),
            },
            "test_range": {
                "start": str(X_test.index.min()),
                "end": str(X_test.index.max()),
            },
            "label_distribution": {
                "train": _serialize_label_distribution(y_train),
                "test": _serialize_label_distribution(y_test),
            },
        }

        feature_stats = {
            "train_mean": X_train.mean().astype(float).to_dict(),
            "test_mean": X_test.mean().astype(float).to_dict(),
            "train_std": X_train.std(ddof=0).astype(float).to_dict(),
            "test_std": X_test.std(ddof=0).astype(float).to_dict(),
        }
        psi_scores = {
            col: _population_stability_index(X_train[col], X_test[col])
            for col in features.columns
        }
        drift_summary = {
            "average_psi": float(np.nanmean(list(psi_scores.values()))),
            "max_psi": float(np.nanmax(list(psi_scores.values())))
            if psi_scores
            else 0.0,
            "feature_psi": psi_scores,
        }

        if task == "binary":
            probs = _predict_proba(model, X_test)
            oof_pred.iloc[test_idx] = probs

            fold_accuracy = accuracy_score(y_test, (probs > 0.5).astype(int))
            cv_accuracy.append(fold_accuracy)
            try:
                auc = roc_auc_score(y_test, probs)
                cv_auc.append(auc)
            except ValueError:
                auc = np.nan
                cv_auc.append(np.nan)

            fold_payload["scores"] = {
                "accuracy": float(fold_accuracy),
                "roc_auc": float(auc) if not np.isnan(auc) else np.nan,
            }
        elif task == "regression":
            preds = model.predict(X_test)
            oof_pred.iloc[test_idx] = preds
            fold_mae = mean_absolute_error(y_test, preds)
            cv_mae.append(fold_mae)
            try:
                fold_r2 = r2_score(y_test, preds)
                cv_r2.append(fold_r2)
            except ValueError:
                fold_r2 = np.nan
                cv_r2.append(np.nan)

            fold_payload["scores"] = {
                "mae": float(fold_mae),
                "r2": float(fold_r2) if not np.isnan(fold_r2) else np.nan,
            }
        else:  # pragma: no cover - defensive branch
            raise ValueError(f"Unsupported task '{task}' for model scoring")

        fold_payload["feature_stats"] = feature_stats
        fold_payload["drift"] = drift_summary
        fold_details.append(fold_payload)

    if task == "binary":
        metrics = {
            "accuracy": float(np.nanmean(cv_accuracy)),
            "roc_auc": float(np.nanmean(cv_auc)),
        }
    else:
        metrics = {
            "mae": float(np.nanmean(cv_mae)),
            "r2": float(np.nanmean(cv_r2)),
        }
    model.fit(features, labels)

    cv_report = {
        "model": model_name,
        "task": task,
        "folds": fold_details,
        "aggregate": metrics,
    }
    return model, metrics, oof_pred, cv_report


def _predict_proba(model, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        decision = model.decision_function(features)
        return 1 / (1 + np.exp(-decision))
    preds = model.predict(features)
    return preds.astype(float)


class StatsmodelsProbitClassifier:
    """Light wrapper to make statsmodels Probit mimic scikit-learn API."""

    def fit(self, features: pd.DataFrame, labels: pd.Series):
        import statsmodels.api as sm

        features_const = sm.add_constant(features, has_constant="add")
        self._model = sm.Probit(labels, features_const)
        self._result = self._model.fit(disp=0)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        import statsmodels.api as sm

        features_const = sm.add_constant(features, has_constant="add")
        return self._result.predict(features_const)


def train_model_stack(
    features: pd.DataFrame,
    labels: pd.Series,
    config: ModelStackConfig,
    *,
    task: str = "binary",
) -> Tuple[
    Dict[str, object], Dict[str, Mapping[str, float]], pd.DataFrame, Dict[str, Mapping[str, object]]
]:
    models: Dict[str, object] = {}
    cv_metrics: Dict[str, Mapping[str, float]] = {}
    cv_reports: Dict[str, Mapping[str, object]] = {}
    predictions = pd.DataFrame(index=features.index)

    cv = TimeSeriesSplit(n_splits=config.cv_splits)

    non_null_labels = labels.dropna()
    unique_values = pd.unique(non_null_labels)
    is_binary_label = len(unique_values) <= 2 and np.isin(unique_values, [0, 1]).all()
    if task == "binary" and not is_binary_label:
        raise ValueError("LabelConfig.task='binary' tetapi label bersifat kontinu. Gunakan 'regression'.")
    if task == "regression" and len(unique_values) <= 2:
        raise ValueError(
            "LabelConfig.task='regression' tetapi label hanya berisi dua nilai unik. Gunakan 'binary'."
        )

    if task == "binary":
        if config.train_linear:
            for c in config.logistic_l1_cs:
                name = f"logistic_l1_C{c}"
                linear_model = LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=c,
                    tol=config.logistic_tol,
                    max_iter=config.logistic_max_iter,
                    random_state=config.random_state,
                )
                model, metrics, preds, cv_report = _train_and_score_model(
                    name, linear_model, features, labels, cv, task
                )
                models[name] = model
                cv_metrics[name] = metrics
                cv_reports[name] = cv_report
                predictions[name] = preds

        if config.train_logistic_elasticnet:
            for c in config.logistic_elasticnet_cs:
                for l1_ratio in config.logistic_elasticnet_l1_ratios:
                    name = f"logistic_en_C{c}_l1r{l1_ratio}"
                    elastic = LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=l1_ratio,
                        C=c,
                        tol=config.logistic_tol,
                        max_iter=config.logistic_elasticnet_max_iter,
                        random_state=config.random_state,
                    )
                    model, metrics, preds, cv_report = _train_and_score_model(
                        name, elastic, features, labels, cv, task
                    )
                    models[name] = model
                    cv_metrics[name] = metrics
                    cv_reports[name] = cv_report
                    predictions[name] = preds

        if config.train_probit:
            if importlib.util.find_spec("statsmodels") is None:
                raise ImportError(
                    "statsmodels tidak tersedia. Install untuk mengaktifkan model Probit."
                )

            probit = StatsmodelsProbitClassifier()
            model, metrics, preds, cv_report = _train_and_score_model(
                "probit", probit, features, labels, cv, task
            )
            models["probit"] = model
            cv_metrics["probit"] = metrics
            cv_reports["probit"] = cv_report
            predictions["probit"] = preds

        if config.train_sgd:
            for loss in config.sgd_losses:
                for alpha in config.sgd_alphas:
                    name = f"sgd_{loss}_alpha{alpha}"
                    sgd = SGDClassifier(
                        loss=loss,
                        penalty="elasticnet" if loss == "log_loss" else "l2",
                        alpha=alpha,
                        max_iter=config.sgd_max_iter,
                        tol=config.sgd_tol,
                        early_stopping=config.sgd_early_stopping,
                        n_iter_no_change=config.sgd_n_iter_no_change,
                        validation_fraction=config.sgd_validation_fraction,
                        random_state=config.random_state,
                    )
                    model, metrics, preds, cv_report = _train_and_score_model(
                        name, sgd, features, labels, cv, task
                    )
                    models[name] = model
                    cv_metrics[name] = metrics
                    cv_reports[name] = cv_report
                    predictions[name] = preds

        if config.train_tree_based:
            gbdt = GradientBoostingClassifier(random_state=config.random_state)
            model, metrics, preds, cv_report = _train_and_score_model(
                "gbdt", gbdt, features, labels, cv, task
            )
            models["gbdt"] = model
            cv_metrics["gbdt"] = metrics
            cv_reports["gbdt"] = cv_report
            predictions["gbdt"] = preds

        if config.train_deep_learning:
            hidden_layers = tuple(int(layer) for layer in config.mlp_hidden_layers)
            mlp = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation=config.mlp_activation,
                solver="adam",
                learning_rate_init=0.001,
                alpha=config.mlp_alpha,
                beta_1=config.mlp_beta1,
                beta_2=config.mlp_beta2,
                validation_fraction=config.mlp_validation_fraction,
                early_stopping=config.mlp_early_stopping,
                max_iter=config.mlp_max_iter,
                random_state=config.random_state,
            )
            model, metrics, preds, cv_report = _train_and_score_model(
                "mlp", mlp, features, labels, cv, task
            )
            models["mlp"] = model
            cv_metrics["mlp"] = metrics
            cv_reports["mlp"] = cv_report
            predictions["mlp"] = preds
    elif task == "regression":
        if config.train_linear:
            linear_variants = {
                "linreg": LinearRegression(),
                "ridge": Ridge(random_state=config.random_state),
                "lasso": Lasso(random_state=config.random_state),
            }
            for name, model_instance in linear_variants.items():
                model, metrics, preds, cv_report = _train_and_score_model(
                    name, model_instance, features, labels, cv, task
                )
                models[name] = model
                cv_metrics[name] = metrics
                cv_reports[name] = cv_report
                predictions[name] = preds

        if config.train_tree_based:
            gbdt_reg = GradientBoostingRegressor(random_state=config.random_state)
            model, metrics, preds, cv_report = _train_and_score_model(
                "gbdt_reg", gbdt_reg, features, labels, cv, task
            )
            models["gbdt_reg"] = model
            cv_metrics["gbdt_reg"] = metrics
            cv_reports["gbdt_reg"] = cv_report
            predictions["gbdt_reg"] = preds
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported task '{task}'")

    if predictions.empty:
        raise ValueError("Model stack is empty. Enable at least one model type")

    return models, cv_metrics, predictions, cv_reports


# ---------------------------------------------------------------------------
# Signal generation and portfolio construction


def _combine_predictions(predictions: pd.DataFrame, task: str) -> pd.Series:
    ensemble = predictions.mean(axis=1)
    if task == "binary":
        return 2 * ensemble - 1  # map probability to [-1, 1]

    if predictions.empty:
        return pd.Series(dtype=float)

    # Direction diambil dari sign prediksi, magnitude memakai peringkat absolut (quantile)
    strength = ensemble.abs().rank(pct=True)
    return np.sign(ensemble) * strength.clip(0.0, 1.0)


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
    cv_report_path: Optional[str | Path] = Path("outputs/cv_report.json"),
    cv_plot_path: Optional[str | Path] = Path("outputs/cv_stability.png"),
) -> MLPipelineResult:
    """Execute the multi-source ML pipeline and return rich artifacts."""

    ohlcv = load_ohlcv_csv(data_cfg.ohlcv_path)
    merged_features = _merge_sources(ohlcv, data_cfg)

    labels = build_labels(ohlcv, label_cfg)
    labels = labels.loc[merged_features.index].dropna()
    merged_features = merged_features.loc[labels.index]

    feature_store, normalized = FeatureStore.fit(merged_features)
    models, cv_metrics, oof_predictions, cv_report = train_model_stack(
        normalized, labels, model_cfg, task=label_cfg.task
    )
    cv_artifacts = _export_cv_artifacts(cv_report, cv_report_path, cv_plot_path)
    signals = _combine_predictions(oof_predictions, label_cfg.task)
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
        cv_report=cv_report,
        cv_artifacts=cv_artifacts,
        predictions=oof_predictions,
        signals=signals,
        portfolio_weights=portfolio_weights,
        realized_performance=realized_performance,
        guardrail_flags=guardrails,
    )


__all__ = [
    "AuxiliarySourceConfig",
    "MultiSourceDataConfig",
    "FeatureEngineeringConfig",
    "LabelConfig",
    "ModelStackConfig",
    "PortfolioConfig",
    "FeatureStore",
    "MLPipelineResult",
    "run_practical_crypto_ml_pipeline",
]

