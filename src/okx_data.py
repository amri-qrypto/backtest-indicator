"""Helpers for downloading and caching OHLCV data from the OKX public API."""
from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib import error, parse, request

import pandas as pd

from .strategy_backtest.utils import load_strategy_csv, sanitise_columns

OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"
OKX_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; BacktestIndicator/1.0; "
        "+https://github.com/tegar90/backtest-indicator)"
    ),
    "Accept": "application/json",
}


@dataclass
class OKXDataset:
    """Container describing the downloaded or cached dataset."""

    data: pd.DataFrame
    column_mapping: Dict[str, str]
    source: str
    path: Optional[Path] = None

    def __post_init__(self) -> None:  # pragma: no cover - simple runtime guard
        if self.source not in {"okx", "local"}:
            raise ValueError("source must be either 'okx' or 'local'")


def _to_millis(value: object | None) -> Optional[int]:
    if value in (None, ""):
        return None
    timestamp = pd.Timestamp(value, tz="UTC")
    return int(timestamp.timestamp() * 1000)


def _normalise_label(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text)


def _format_timestamp_label(timestamp: pd.Timestamp) -> str:
    ts = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    return ts.strftime("%Y%m%d%H%M%S")


def _load_local_dataset(path: Path) -> OKXDataset:
    df, mapping = load_strategy_csv(path)
    df = df.sort_index()
    return OKXDataset(data=df, column_mapping=mapping, source="local", path=path)


def _ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _discover_latest_cached(
    directory: Path, inst_id: str, bar: str
) -> Optional[Path]:  # pragma: no cover - filesystem side effect
    pattern = f"sample-{_normalise_label(inst_id)}-{_normalise_label(bar)}-*.csv"
    candidates = sorted(directory.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _export_dataset(
    df: pd.DataFrame, directory: Path, inst_id: str, bar: str
) -> Path:  # pragma: no cover - filesystem side effect
    directory.mkdir(parents=True, exist_ok=True)
    if df.empty:
        raise ValueError("Cannot export an empty dataset")

    start_label = _format_timestamp_label(pd.Timestamp(df.index.min()))
    end_label = _format_timestamp_label(pd.Timestamp(df.index.max()))
    filename = f"sample-{_normalise_label(inst_id)}-{_normalise_label(bar)}-{start_label}-{end_label}.csv"
    export_path = directory / filename

    time_index = pd.DatetimeIndex(df.index)
    if time_index.tz is None:
        time_index = time_index.tz_localize("UTC")
    else:
        time_index = time_index.tz_convert("UTC")

    export_df = df.copy()
    export_df.insert(0, "time", (time_index.view("int64") // 10**9))
    export_df.to_csv(export_path, index=False)
    return export_path


def fetch_okx_ohlcv(
    inst_id: str,
    bar: str,
    start: object | None = None,
    end: object | None = None,
    *,
    limit: int = 200,
    pause: float = 0.2,
    request_timeout: float = 10.0,
    fallback_path: str | Path | None = None,
    cache_directory: str | Path | None = "data",
) -> OKXDataset:
    """Fetch OHLCV candles from OKX, optionally caching the results to disk."""

    start_ms = _to_millis(start)
    end_ms = _to_millis(end)
    if end_ms is None:
        end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)

    cache_dir = Path(cache_directory) if cache_directory is not None else None
    fallback = Path(fallback_path) if fallback_path is not None else None

    params = {"instId": inst_id, "bar": bar, "limit": str(limit)}
    all_rows: List[List[str]] = []
    cursor = end_ms

    def _resolve_fallback() -> Optional[Path]:  # pragma: no cover - filesystem side effect
        if fallback is not None:
            return fallback
        if cache_dir is not None:
            return _discover_latest_cached(cache_dir, inst_id, bar)
        return None

    def _use_fallback(reason: str, exc: Exception | None = None) -> OKXDataset:
        candidate = _resolve_fallback()
        if candidate is None:
            if exc is not None:
                raise RuntimeError(reason) from exc
            raise RuntimeError(reason)
        print(f"⚠️ {reason}. Menggunakan dataset lokal: {candidate}")
        return _load_local_dataset(candidate)

    while True:
        query = params | {"before": str(cursor)}
        url = f"{OKX_CANDLES_URL}?{parse.urlencode(query)}"
        req = request.Request(url, headers=OKX_REQUEST_HEADERS)
        try:
            with request.urlopen(req, timeout=request_timeout) as resp:
                raw_body = resp.read().decode("utf-8")
        except (error.URLError, TimeoutError, socket.timeout) as exc:
            return _use_fallback(f"Gagal terhubung ke OKX ({exc})", exc)

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            return _use_fallback("Respon OKX tidak valid", exc)

        if payload.get("code") != "0":
            message = f"Gagal mengambil data OKX: {payload.get('msg')} (code={payload.get('code')})"
            return _use_fallback(message)

        rows = payload.get("data", [])
        if not rows:
            break

        all_rows.extend(rows)
        earliest = min(int(row[0]) for row in rows)
        reached_start = start_ms is not None and earliest <= start_ms
        exhausted = len(rows) < int(params["limit"])

        if reached_start or (start_ms is None and exhausted):
            break

        next_cursor = earliest - 1
        if next_cursor <= 0:
            break
        cursor = next_cursor
        time.sleep(max(pause, 0))

    if not all_rows:
        return _use_fallback("Tidak ada data OHLCV yang diterima dari OKX")

    df = pd.DataFrame(
        all_rows,
        columns=["ts", "o", "h", "l", "c", "vol", "volCcy", "volCcyQuote", "confirm"],
    )
    df = df.drop_duplicates(subset="ts")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])
    df = df.sort_values("ts")

    if start_ms is not None:
        df = df[df["ts"] >= start_ms]
    if end_ms is not None:
        df = df[df["ts"] <= end_ms]

    numeric_cols = ["o", "h", "l", "c", "vol", "volCcy", "volCcyQuote"]
    df = _ensure_numeric(df, numeric_cols)
    df["confirm"] = df["confirm"].astype(int)

    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True).tz_convert(None)
    df = df.drop(columns=["ts"])
    df = df.dropna(subset=["time"])

    rename_map = {
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "vol": "volume",
        "volCcy": "volume_ccy",
        "volCcyQuote": "volume_quote",
        "confirm": "confirm",
    }
    df = df.rename(columns=rename_map)
    raw_mapping: Dict[str, str] = {"time": "ts"} | {new: original for original, new in rename_map.items()}

    sanitised_cols, _ = sanitise_columns(df.columns)
    column_mapping = {alias: raw_mapping[col_name] for alias, col_name in zip(sanitised_cols, df.columns)}
    df.columns = sanitised_cols
    df = df.set_index("time").sort_index()

    if start_ms is not None and not df.empty:
        requested_start = pd.Timestamp(start_ms, unit="ms", tz="UTC")
        actual_start = pd.Timestamp(df.index.min(), tz="UTC")
        if actual_start > requested_start:
            print(
                "⚠️ Data OKX tersedia mulai "
                f"{actual_start.tz_convert(None)} (lebih lambat dari START_TIME)."
            )

    export_path: Optional[Path] = None
    if cache_dir is not None:
        export_path = _export_dataset(df, cache_dir, inst_id, bar)

    return OKXDataset(data=df, column_mapping=column_mapping, source="okx", path=export_path)


__all__ = ["OKXDataset", "fetch_okx_ohlcv"]

