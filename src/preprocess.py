"""Leakage-safe preprocessing utilities for tabular PCOS data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from imblearn.over_sampling import SMOTE
except ImportError:  # pragma: no cover - runtime dependency
    SMOTE = None  # type: ignore[assignment]


NUMERIC_CAST_THRESHOLD = 0.90
UNKNOWN_TOKEN = "__unknown__"


@dataclass
class PreprocessorState:
    """Store fitted train-time preprocessing artifacts."""

    numeric_like_columns: List[str] = field(default_factory=list)
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    numeric_medians: Dict[str, float] = field(default_factory=dict)
    categorical_modes: Dict[str, str] = field(default_factory=dict)
    encoders: Dict[str, LabelEncoder] = field(default_factory=dict)
    scaler: StandardScaler = field(default_factory=StandardScaler)


class TabularPreprocessor:
    """Impute, label-encode, and standardize tabular features (fit on train only)."""

    def __init__(self) -> None:
        self.state = PreprocessorState()
        self._is_fitted = False

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessing steps on training features and transform them."""
        transformed = self._fit_transform_features(X_train.copy())
        self._is_fitted = True
        return transformed

    def transform(self, X_data: pd.DataFrame) -> pd.DataFrame:
        """Transform features using train-fitted artifacts."""
        if not self._is_fitted:
            raise RuntimeError("TabularPreprocessor must be fitted before transform.")

        frame = X_data.copy()
        frame = self._cast_numeric_like(frame)
        frame = self._impute(frame, fit=False)
        frame = self._encode_categorical(frame, fit=False)
        if self.state.numeric_columns:
            frame[self.state.numeric_columns] = self.state.scaler.transform(frame[self.state.numeric_columns])
        return frame

    def _fit_transform_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Train preprocessing stack and transform training data."""
        self.state.numeric_like_columns = _detect_numeric_like_columns(frame)
        frame = self._cast_numeric_like(frame)

        self.state.numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
        self.state.categorical_columns = [c for c in frame.columns if c not in self.state.numeric_columns]

        frame = self._impute(frame, fit=True)
        frame = self._encode_categorical(frame, fit=True)

        self.state.scaler = StandardScaler()
        if self.state.numeric_columns:
            frame[self.state.numeric_columns] = self.state.scaler.fit_transform(frame[self.state.numeric_columns])
        return frame

    def _cast_numeric_like(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Cast train-detected numeric-like object columns to numeric."""
        for column in self.state.numeric_like_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    def _impute(self, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Impute numerical with median and categorical with mode."""
        if fit:
            self.state.numeric_medians = {
                column: float(frame[column].median()) for column in self.state.numeric_columns
            }
            self.state.categorical_modes = {}
            for column in self.state.categorical_columns:
                mode_series = frame[column].mode(dropna=True)
                self.state.categorical_modes[column] = str(mode_series.iloc[0]) if not mode_series.empty else "missing"

        for column in self.state.numeric_columns:
            frame[column] = frame[column].fillna(self.state.numeric_medians[column])
        for column in self.state.categorical_columns:
            frame[column] = frame[column].fillna(self.state.categorical_modes[column]).astype(str)
        return frame

    def _encode_categorical(self, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Label-encode categorical columns with unknown-token safety."""
        for column in self.state.categorical_columns:
            if fit:
                encoder = LabelEncoder()
                encoder.fit(frame[column].astype(str).tolist() + [UNKNOWN_TOKEN])
                self.state.encoders[column] = encoder

            encoder = self.state.encoders[column]
            known = set(encoder.classes_.tolist())
            values = frame[column].astype(str)
            values = values.where(values.isin(known), UNKNOWN_TOKEN)
            frame[column] = encoder.transform(values)

        return frame


def remove_duplicates(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate records."""
    return dataframe.drop_duplicates().reset_index(drop=True)


def split_features_target(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into feature matrix and target vector."""
    if "target" not in dataframe.columns:
        raise ValueError("Expected 'target' column in dataframe.")
    X = dataframe.drop(columns=["target"]).copy()
    y = normalize_binary_target(dataframe["target"])
    return X, y


def normalize_binary_target(target: pd.Series) -> pd.Series:
    """Convert mixed target labels (yes/no, 0/1) into strict binary integers."""
    text = target.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1,
        "yes": 1,
        "y": 1,
        "true": 1,
        "positive": 1,
        "0": 0,
        "no": 0,
        "n": 0,
        "false": 0,
        "negative": 0,
    }
    mapped = text.map(mapping)
    numeric = pd.to_numeric(text, errors="coerce")
    mapped = mapped.where(mapped.notna(), np.where(numeric.notna(), (numeric > 0).astype(int), np.nan))

    if mapped.isna().any():
        fill_value = int(mapped.mode(dropna=True).iloc[0]) if mapped.notna().any() else 0
        mapped = mapped.fillna(fill_value)
    return mapped.astype(int)


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train/test split with stratification."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def apply_limited_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """
    Conditionally apply limited SMOTE.

    Applied only if current minority ratio < `sampling_strategy`.
    """
    class_counts = y_train.value_counts().sort_values(ascending=False)
    if class_counts.empty or len(class_counts) < 2:
        return X_train.copy(), y_train.copy(), {"applied": False, "reason": "single_class"}

    majority = int(class_counts.iloc[0])
    minority = int(class_counts.iloc[-1])
    current_ratio = float(minority / majority) if majority > 0 else 1.0

    if current_ratio >= sampling_strategy:
        return (
            X_train.copy(),
            y_train.copy(),
            {"applied": False, "reason": "ratio_already_high", "current_ratio": current_ratio},
        )

    if SMOTE is None:
        raise ImportError("SMOTE requires 'imbalanced-learn'. Install with: pip install imbalanced-learn")

    if minority < 2:
        return (
            X_train.copy(),
            y_train.copy(),
            {"applied": False, "reason": "minority_too_small", "current_ratio": current_ratio},
        )

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=min(5, minority - 1),
    )
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name=y_train.name)

    return (
        X_resampled,
        y_resampled,
        {"applied": True, "reason": "smote_applied", "current_ratio": current_ratio, "target_ratio": sampling_strategy},
    )


def _detect_numeric_like_columns(frame: pd.DataFrame) -> List[str]:
    """Detect object columns that are mostly numeric values."""
    numeric_like: List[str] = []
    for column in frame.columns:
        if frame[column].dtype != object:
            continue
        coerced = pd.to_numeric(frame[column], errors="coerce")
        if coerced.notna().mean() >= NUMERIC_CAST_THRESHOLD:
            numeric_like.append(column)
    return numeric_like
