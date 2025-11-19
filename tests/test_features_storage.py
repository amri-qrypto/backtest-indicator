import warnings
from pathlib import Path

import pandas as pd
import pytest

from src.features.storage import export_feature_dataset


def test_export_feature_dataset_prefers_parquet(monkeypatch, tmp_path):
    df = pd.DataFrame({"value": [1, 2]})

    def fake_to_parquet(self, path):
        Path(path).write_text("PARQUET")

    def fake_to_csv(self, *args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("CSV fallback should not be triggered when Parquet works")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)

    saved_path = export_feature_dataset(df, tmp_path / "dataset.parquet")

    assert saved_path.name == "dataset.parquet"
    assert saved_path.read_text() == "PARQUET"


def test_export_feature_dataset_falls_back_to_csv(monkeypatch, tmp_path):
    df = pd.DataFrame({"value": [1, 2]})

    def fake_to_parquet(self, path):
        raise ImportError("pyarrow missing")

    def fake_to_csv(self, path, index=False):
        Path(path).write_text("value\n1\n2\n")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)

    with warnings.catch_warnings(record=True) as caught:
        saved_path = export_feature_dataset(df, tmp_path / "dataset.parquet")

    assert caught == []

    assert saved_path.suffix == ".csv"
    assert saved_path.read_text() == "value\n1\n2\n"


def test_export_feature_dataset_resolves_relative_paths(monkeypatch, tmp_path):
    df = pd.DataFrame({"value": [1, 2]})

    def fake_to_parquet(self, path):
        Path(path).write_text("PARQUET")

    def fake_to_csv(self, *args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("CSV fallback should not be triggered when Parquet works")

    import src.features.storage as storage

    monkeypatch.setattr(storage, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)

    saved_path = storage.export_feature_dataset(df, Path("relative/dataset.parquet"))

    assert saved_path == tmp_path / "relative/dataset.parquet"
    assert saved_path.read_text() == "PARQUET"
