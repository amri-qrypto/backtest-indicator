"""Regression-based ML executor that maps predicted returns into trading signals."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..base import StrategyBase, StrategyMetadata
from ...pipelines.practical_crypto_ml import (
    AuxiliarySourceConfig,
    LabelConfig,
    ModelStackConfig,
    MultiSourceDataConfig,
    PortfolioConfig,
    run_practical_crypto_ml_pipeline,
)


def _build_aux_source(path: Optional[str], prefix: str) -> AuxiliarySourceConfig:
    if path:
        return AuxiliarySourceConfig(path=path, prefix=prefix)
    return AuxiliarySourceConfig()


class Strategy(StrategyBase):
    """Train linear regressors to predict forward returns and emit direction/strength signals."""

    metadata = StrategyMetadata(
        name="ml_linreg",
        description=(
            "Pipeline ML multi-sumber untuk tugas regresi: memprediksi forward returns dengan "
            "LinearRegression/Ridge/Lasso lalu mengubahnya menjadi sinyal arah [-1, 1] berbasis sign"
            " & quantile."
        ),
        entry=(
            "Masuk long saat sinyal positif dan posisi sebelumnya tidak long; masuk short saat sinyal "
            "negatif dan posisi sebelumnya tidak short. Magnitudo sinyal mewakili kekuatan (quantile) "
            "tetapi entri/keluar tetap mengikuti perubahan arah."
        ),
        exit=(
            "Keluar ketika sinyal berbalik atau kembali ke netral. Guardrail leverage/turnover ikut "
            "dicatat di konteks sinyal."
        ),
        parameters={
            "label_horizon": 24,
            "top_k": 1,
            "long_short": True,
            "max_leverage": 1.0,
            "turnover_limit": 1.5,
            "cv_splits": 4,
            "train_tree_based": False,
        },
        context_columns=(
            "ml_signal",
            "predicted_return",
            "signal_strength",
            "guardrail_max_leverage",
            "guardrail_turnover",
        ),
    )

    def __init__(
        self,
        data_path: Optional[str] = None,
        *,
        label_horizon: int = 24,
        top_k: int = 1,
        long_short: bool = True,
        max_leverage: float = 1.0,
        turnover_limit: float = 1.5,
        cv_splits: int = 4,
        funding_path: Optional[str] = None,
        order_book_path: Optional[str] = None,
        on_chain_path: Optional[str] = None,
        sentiment_path: Optional[str] = None,
        train_tree_based: bool = False,
    ) -> None:
        super().__init__(
            label_horizon=label_horizon,
            top_k=top_k,
            long_short=long_short,
            max_leverage=max_leverage,
            turnover_limit=turnover_limit,
            cv_splits=cv_splits,
            train_tree_based=train_tree_based,
        )
        self.data_path = data_path
        self.label_horizon = label_horizon
        self.top_k = top_k
        self.long_short = long_short
        self.max_leverage = max_leverage
        self.turnover_limit = turnover_limit
        self.cv_splits = cv_splits
        self.funding_path = funding_path
        self.order_book_path = order_book_path
        self.on_chain_path = on_chain_path
        self.sentiment_path = sentiment_path
        self.train_tree_based = train_tree_based

    def _build_configs(self) -> tuple[
        MultiSourceDataConfig, LabelConfig, ModelStackConfig, PortfolioConfig
    ]:
        if not self.data_path:
            raise ValueError(
                "Parameter 'data_path' wajib diisi agar pipeline ML bisa memuat OHLCV. "
                "Isi field ini di strategy_kwargs dan samakan dengan data_path pada konfigurasi pipeline."
            )
        data_cfg = MultiSourceDataConfig(
            ohlcv_path=self.data_path,
            funding_rates=_build_aux_source(self.funding_path, "fund_"),
            order_book_depth=_build_aux_source(self.order_book_path, "ob_"),
            on_chain_activity=_build_aux_source(self.on_chain_path, "onchain_"),
            sentiment_scores=_build_aux_source(self.sentiment_path, "sent_"),
        )
        label_cfg = LabelConfig(horizon_bars=self.label_horizon, task="regression", threshold=0.0)
        model_cfg = ModelStackConfig(
            train_linear=True,
            train_tree_based=self.train_tree_based,
            train_deep_learning=False,
            cv_splits=self.cv_splits,
        )
        portfolio_cfg = PortfolioConfig(
            top_k=self.top_k,
            long_short=self.long_short,
            max_leverage=self.max_leverage,
            turnover_limit=self.turnover_limit,
        )
        return data_cfg, label_cfg, model_cfg, portfolio_cfg

    def _to_entry_exit(self, desired_position: pd.Series) -> pd.DataFrame:
        aligned = desired_position.astype(int)
        previous = aligned.shift(1).fillna(0).astype(int)

        long_entry = (aligned == 1) & (previous <= 0)
        long_exit = (previous > 0) & (aligned <= 0)
        short_entry = (aligned == -1) & (previous >= 0)
        short_exit = (previous < 0) & (aligned >= 0)

        return pd.DataFrame(
            {
                "long_entry": long_entry,
                "long_exit": long_exit,
                "short_entry": short_entry,
                "short_exit": short_exit,
            }
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data_cfg, label_cfg, model_cfg, portfolio_cfg = self._build_configs()
        result = run_practical_crypto_ml_pipeline(
            data_cfg=data_cfg,
            label_cfg=label_cfg,
            model_cfg=model_cfg,
            portfolio_cfg=portfolio_cfg,
        )

        ml_signal = result.signals.reindex(data.index).fillna(0.0)
        predicted_return = result.predictions.mean(axis=1).reindex(data.index).fillna(0.0)
        signal_strength = ml_signal.abs()
        desired_position = np.sign(ml_signal)

        signals = self._to_entry_exit(desired_position)
        signals["ml_signal"] = ml_signal
        signals["predicted_return"] = predicted_return
        signals["signal_strength"] = signal_strength
        signals["guardrail_max_leverage"] = bool(result.guardrail_flags.get("max_leverage", False))
        signals["guardrail_turnover"] = bool(result.guardrail_flags.get("turnover", False))

        return signals


__all__ = ["Strategy"]
