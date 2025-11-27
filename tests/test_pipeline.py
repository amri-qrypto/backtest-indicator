from __future__ import annotations

from pathlib import Path

import numpy as np

from src.pipelines.single_asset import (
    IndicatorConfig,
    SingleAssetPipelineConfig,
    run_single_asset_pipeline,
    save_backtest_outputs,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _fixture_path(name: str) -> str:
    return str(FIXTURES / name)


def test_pipeline_runs_on_mini_dataset(tmp_path):
    config = SingleAssetPipelineConfig(
        data_path=_fixture_path("mini_ohlcv.csv"),
        strategy_name="ema50",
        strategy_kwargs={"ema_length": 50},
        indicators=(
            IndicatorConfig(
                name="ema",
                source_column="close",
                target_column="ema_fast",
                params={"span": 25},
            ),
        ),
        horizon_bars=120,
        price_column="close",
    )

    outputs = run_single_asset_pipeline(config)

    assert outputs.trade_summary["total_trades"] > 0
    assert np.isfinite(outputs.metrics["sharpe_ratio"])
    assert not outputs.trades["pnl_pct"].isna().any()

    artifacts = save_backtest_outputs(outputs, tmp_path, prefix="mini")
    metrics = (tmp_path / "mini_metrics.json").read_text()
    assert (tmp_path / "mini_metrics.json") == artifacts["metrics"]
    assert (tmp_path / "mini_trades.csv").exists()
    assert (tmp_path / "mini_equity.png").exists()
    assert "standard_metrics" in metrics
