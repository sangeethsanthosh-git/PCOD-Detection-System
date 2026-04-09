"""SHAP-based explainability helpers for training-time reporting."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None  # type: ignore[assignment]

from .utils import stratified_subsample


TREE_EXPLAINER_CANDIDATES = ("XGBoost", "LightGBM", "RandomForest")


def select_explainer_model_name(
    results: Dict[str, Dict[str, object]],
    candidate_names: tuple[str, ...] = TREE_EXPLAINER_CANDIDATES,
) -> str:
    """Choose the best tree model for SHAP explainability."""
    available = [name for name in candidate_names if name in results]
    if not available:
        raise ValueError("No tree-based model is available for SHAP explainability.")

    return max(
        available,
        key=lambda name: (
            float(results[name].get("f1") or 0.0),
            float(results[name].get("roc_auc") or 0.0),
        ),
    )


def compute_global_shap_importance(
    model: object,
    X: pd.DataFrame,
    y: pd.Series | None = None,
    max_samples: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute mean absolute SHAP importance for a fitted tree model."""
    if shap is None:
        raise ImportError("The shap package is required for SHAP explainability.")

    X_sample = X.copy()
    if y is not None:
        X_sample, _ = stratified_subsample(
            X=X_sample,
            y=y,
            max_samples=max_samples,
            random_state=random_state,
        )
    elif len(X_sample) > max_samples:
        X_sample = X_sample.sample(n=max_samples, random_state=random_state)

    explainer = shap.TreeExplainer(model)  # type: ignore[union-attr]
    shap_values = explainer.shap_values(X_sample)
    values = _coerce_shap_values(shap_values)
    mean_abs = np.mean(np.abs(values), axis=0)

    importance_df = (
        pd.DataFrame({"feature": X_sample.columns, "importance": mean_abs})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return importance_df


def _coerce_shap_values(shap_values: object) -> np.ndarray:
    """Normalize SHAP outputs across library versions."""
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[-1], dtype=float)
    elif hasattr(shap_values, "values"):
        values = np.asarray(shap_values.values, dtype=float)
    else:
        values = np.asarray(shap_values, dtype=float)

    if values.ndim == 3:
        values = values[:, :, -1]
    return values
