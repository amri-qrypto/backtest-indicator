"""Helpers for persisting engineered feature datasets."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def export_feature_dataset(dataset: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist ``dataset`` to ``output_path`` ensuring a usable format is produced.

    The function prefers Parquet for its compression and schema support. However,
    notebooks are often executed in lightweight environments where optional
    dependencies such as ``pyarrow`` or ``fastparquet`` might be absent. If Pandas
    raises :class:`ImportError` while attempting the Parquet export, the dataset is
    transparently written to CSV instead so the pipeline still produces an output
    artefact.
    """

    path = Path(output_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        dataset.to_parquet(path)
        return path
    except ImportError:
        fallback_path = path.with_suffix(".csv")
        dataset.to_csv(fallback_path, index=False)
        logger.info(
            "Missing optional dependency 'pyarrow' or 'fastparquet'; saved features as CSV instead. "
            "Run 'pip install pyarrow' to enable Parquet exports."
        )
        return fallback_path
