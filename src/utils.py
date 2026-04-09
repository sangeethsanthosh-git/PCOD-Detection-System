"""Shared helper utilities for the PCOS ML pipeline."""

from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_CANDIDATES = ("pcos_y_n", "pcos", "diagnosis", "outcome")


def ensure_directories(paths: Iterable[Path]) -> None:
    """Create directories if they do not already exist."""
    for directory in paths:
        Path(directory).mkdir(parents=True, exist_ok=True)


def normalize_column_name(column_name: str) -> str:
    """
    Normalize a raw column name.

    Rules:
    - lowercase
    - replace non-alphanumeric characters with underscore
    - collapse multiple underscores
    """
    normalized = str(column_name).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def detect_target_column(columns: Sequence[str]) -> Optional[str]:
    """Detect target column name from a list of normalized column names."""
    normalized_columns = [normalize_column_name(column) for column in columns]

    for candidate in TARGET_CANDIDATES:
        if candidate in normalized_columns:
            return candidate

    for column in normalized_columns:
        if "pcos" in column or "diagnosis" in column or "outcome" in column:
            return column

    return None


def resolve_existing_path(candidate_paths: Sequence[Path]) -> Optional[Path]:
    """Return the first existing path from the provided candidates."""
    for path in candidate_paths:
        if path.exists():
            return path
    return None


def extract_positive_class_scores(model, features) -> Optional[np.ndarray]:
    """Return continuous model scores needed for ROC-AUC when available."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(features)

    return None


def stratified_subsample(
    X: pd.DataFrame,
    y: pd.Series,
    max_samples: int,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return a stratified sample when dataset size exceeds a limit."""
    if max_samples <= 0 or len(X) <= max_samples:
        return X, y

    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=max_samples,
        stratify=y,
        random_state=random_state,
    )
    return X_sample, y_sample


def safe_int_from_env(env_name: str, default_value: int) -> int:
    """Read positive integer from env var with safe fallback."""
    value = str(os.getenv(env_name, "")).strip()
    if not value:
        return default_value
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default_value
    except ValueError:
        return default_value
