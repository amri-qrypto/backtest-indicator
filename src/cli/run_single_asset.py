"""CLI entrypoint to execute the single asset pipeline from a config file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML may be missing at runtime
    yaml = None  # type: ignore

from ..pipelines.single_asset import (
    ExternalSignalModulationConfig,
    IndicatorConfig,
    SingleAssetPipelineConfig,
    run_single_asset_pipeline,
    save_backtest_outputs,
)


def _load_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError(
                "PyYAML belum terpasang. Instal dependensi 'PyYAML' untuk membaca file YAML."
            )
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)

    if not isinstance(data, Mapping):
        raise TypeError("Konfigurasi harus berupa objek mapping/dictionary.")

    return data


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def _build_indicator_configs(payload: Sequence[Mapping[str, Any]]) -> Sequence[IndicatorConfig]:
    indicators = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise TypeError("Setiap entri indikator harus berupa mapping.")
        indicator_kwargs: MutableMapping[str, Any] = {
            key: value for key, value in item.items() if key in {"name", "source_column", "target_column", "params"}
        }
        indicators.append(IndicatorConfig(**indicator_kwargs))
    return tuple(indicators)


def _build_external_signal_config(
    payload: Mapping[str, Any] | None, base_dir: Path
) -> ExternalSignalModulationConfig | None:
    if payload is None:
        return None

    if not isinstance(payload, Mapping):
        raise TypeError("external_signals harus berupa mapping jika disediakan.")

    resolved: MutableMapping[str, Any] = dict(payload)
    if "path" in resolved:
        resolved["path"] = str(_resolve_path(resolved["path"], base_dir))

    return ExternalSignalModulationConfig(**resolved)


def _build_pipeline_config(payload: Mapping[str, Any], base_dir: Path) -> SingleAssetPipelineConfig:
    required_fields = {"data_path", "strategy_name"}
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise KeyError(f"Field konfigurasi wajib hilang: {missing}")

    data_path = _resolve_path(payload["data_path"], base_dir)
    strategy_name = payload["strategy_name"]

    indicators_payload = payload.get("indicators", [])
    if indicators_payload is None:
        indicators_payload = []

    indicators = _build_indicator_configs(indicators_payload)

    external_signals_cfg = _build_external_signal_config(
        payload.get("external_signals"), base_dir
    )

    return SingleAssetPipelineConfig(
        data_path=str(data_path),
        strategy_name=str(strategy_name),
        strategy_kwargs=dict(payload.get("strategy_kwargs", {})),
        horizon_bars=payload.get("horizon_bars"),
        indicators=indicators,
        price_column=str(payload.get("price_column", "close")),
        external_signals=external_signals_cfg,
    )


def _output_settings(payload: Mapping[str, Any], args: argparse.Namespace, base_dir: Path) -> tuple[Path | None, str]:
    output_dir_value = args.output_dir or payload.get("output_dir")
    output_prefix = args.prefix or payload.get("output_prefix", "backtest")

    if output_dir_value is None:
        return None, str(output_prefix)

    output_dir = _resolve_path(output_dir_value, base_dir)
    return output_dir, str(output_prefix)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jalankan pipeline single asset via file konfigurasi.")
    parser.add_argument("config", type=Path, help="Path menuju file konfigurasi JSON/YAML.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override folder output untuk metrics/trades.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Prefix nama file hasil ekspor.",
    )
    parser.add_argument(
        "--skip-save",
        action="store_true",
        help="Jika di-set, tidak menyimpan metrics/trade log ke disk.",
    )
    parser.add_argument(
        "--external-signal",
        type=Path,
        default=None,
        help="Path CSV berisi kolom waktu + sinyal eksternal untuk scaling/gating posisi.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = args.config.resolve()
    payload = _load_mapping(config_path)
    if args.external_signal:
        payload["external_signals"] = dict(payload.get("external_signals", {}))
        payload["external_signals"]["path"] = str(args.external_signal)

    pipeline_config = _build_pipeline_config(payload, config_path.parent)

    outputs = run_single_asset_pipeline(pipeline_config)

    total_trades = outputs.trade_summary.get("total_trades", 0)
    sharpe_ratio = outputs.metrics.get("sharpe_ratio")
    cagr = outputs.metrics.get("cagr")
    total_return = outputs.metrics.get("total_return")

    print("Backtest selesai.")
    print(f"  Data: {pipeline_config.data_path}")
    print(f"  Strategi: {pipeline_config.strategy_name}")
    print(f"  Total trades: {total_trades}")
    print(f"  CAGR: {cagr:.4f}" if cagr is not None else "  CAGR: N/A")
    print(f"  Sharpe Ratio: {sharpe_ratio:.4f}" if sharpe_ratio is not None else "  Sharpe Ratio: N/A")
    print(f"  Total Return: {total_return:.4f}" if total_return is not None else "  Total Return: N/A")

    if args.skip_save:
        return

    output_dir, prefix = _output_settings(payload, args, config_path.parent)
    if output_dir is None:
        print("Lewati penyimpanan karena output_dir tidak ditentukan.")
        return

    artifacts = save_backtest_outputs(outputs, output_dir, prefix=prefix)
    print("Artifacts tersimpan:")
    for name, path in artifacts.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
