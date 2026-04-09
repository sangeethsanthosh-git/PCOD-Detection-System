"""Dataset loading and schema normalization for PCOS prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .utils import detect_target_column, normalize_column_name


IDENTIFIER_COLUMNS = {"sl_no", "patient_file_no"}


def load_dataset(dataset_path: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Load one dataset, normalize column names, detect target, and drop identifiers.

    Required target aliases include:
    - pcos_(y/n)
    - pcos
    - diagnosis
    - outcome
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    dataframe = pd.read_csv(dataset_path)
    raw_shape = dataframe.shape

    # Remove accidental unnamed columns from CSV trailing delimiters.
    dataframe = dataframe.loc[
        :,
        ~dataframe.columns.astype(str).str.contains(r"^unnamed", case=False, regex=True),
    ].copy()

    dataframe.columns = [normalize_column_name(column) for column in dataframe.columns]

    target_column = detect_target_column(dataframe.columns.tolist())
    if target_column is None:
        raise ValueError("Target column not detected from expected aliases.")
    if target_column != "target":
        dataframe = dataframe.rename(columns={target_column: "target"})

    dropped_identifiers = [column for column in dataframe.columns if column in IDENTIFIER_COLUMNS]
    dataframe = dataframe.drop(columns=dropped_identifiers, errors="ignore")

    info = {
        "dataset_path": str(dataset_path),
        "raw_shape": raw_shape,
        "final_shape": dataframe.shape,
        "target_column_detected": target_column,
        "dropped_identifier_columns": dropped_identifiers,
    }
    return dataframe, info
