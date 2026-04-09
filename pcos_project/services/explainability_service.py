"""Runtime SHAP explainability helpers for prediction responses."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None  # type: ignore[assignment]

_EXPLAINER_CACHE: dict[int, object] = {}


def build_local_explanation(
    model: object,
    X_row: pd.DataFrame,
    feature_labels: dict[str, str],
    top_n: int = 6,
) -> dict[str, Any]:
    """Return top local feature contributions for one prediction row."""
    if shap is None:
        if hasattr(model, "coef_"):
            return _build_linear_explanation(model, X_row, feature_labels, top_n=top_n)
        raise ImportError("The shap package is required for runtime explanations.")

    if X_row.empty:
        return {
            "ai_explanation": {},
            "contribution_chart": {"labels": [], "values": []},
            "contributors": [],
            "backend": "shap_tree",
        }

    if hasattr(model, "coef_"):
        return _build_linear_explanation(model, X_row, feature_labels, top_n=top_n)

    explainer = _get_tree_explainer(model)
    shap_values = explainer.shap_values(X_row)
    values = _coerce_shap_values(shap_values)
    return _build_ranked_explanation(
        values=np.abs(values[0]),
        columns=X_row.columns.tolist(),
        feature_labels=feature_labels,
        top_n=top_n,
        backend="shap_tree",
    )


def fallback_explanation_from_chart(
    contribution_chart: dict[str, list[Any]] | None,
) -> dict[str, Any]:
    """Convert an existing chart into a chart-plus-dict explanation fallback."""
    labels = list((contribution_chart or {}).get("labels", []))
    values = [float(value) for value in (contribution_chart or {}).get("values", [])]
    ai_explanation = {label: round(value, 4) for label, value in zip(labels, values)}
    return {
        "ai_explanation": ai_explanation,
        "contribution_chart": {
            "labels": labels,
            "values": values,
        },
        "contributors": labels[:4],
        "backend": "heuristic_fallback",
    }


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


def _get_tree_explainer(model: object):
    """Cache TreeExplainer instances for loaded model objects."""
    cache_key = id(model)
    if cache_key not in _EXPLAINER_CACHE:
        _EXPLAINER_CACHE[cache_key] = shap.TreeExplainer(model)  # type: ignore[union-attr]
    return _EXPLAINER_CACHE[cache_key]


def _build_linear_explanation(
    model: object,
    X_row: pd.DataFrame,
    feature_labels: dict[str, str],
    top_n: int = 6,
) -> dict[str, Any]:
    """Approximate local importance for linear models with coefficient-weighted contributions."""
    coefficients = np.asarray(getattr(model, "coef_"), dtype=float)
    if coefficients.ndim == 2:
        coefficients = coefficients[-1]
    if coefficients.ndim != 1:
        raise ValueError("Linear explanation expects a 1D coefficient vector.")

    row_values = np.asarray(X_row.iloc[0], dtype=float)
    contributions = np.abs(coefficients[: len(row_values)] * row_values)
    return _build_ranked_explanation(
        values=contributions,
        columns=X_row.columns.tolist(),
        feature_labels=feature_labels,
        top_n=top_n,
        backend="linear_coeff",
    )


def _build_ranked_explanation(
    values: np.ndarray,
    columns: list[str],
    feature_labels: dict[str, str],
    top_n: int,
    backend: str,
) -> dict[str, Any]:
    """Build a consistent explanation payload from ranked contribution values."""
    ranked_indices = np.argsort(values)[::-1]
    top_indices = [index for index in ranked_indices if float(values[index]) > 0][:top_n]

    if not top_indices:
        return {
            "ai_explanation": {},
            "contribution_chart": {"labels": [], "values": []},
            "contributors": [],
            "backend": backend,
        }

    raw_pairs = []
    for index in top_indices:
        feature_name = columns[index]
        label = feature_labels.get(feature_name, feature_name.replace("_", " ").title())
        raw_pairs.append((label, float(values[index])))

    total = sum(value for _, value in raw_pairs) or 1.0
    return {
        "ai_explanation": {label: round(value, 4) for label, value in raw_pairs},
        "contribution_chart": {
            "labels": [label for label, _ in raw_pairs],
            "values": [round((value / total) * 100, 1) for _, value in raw_pairs],
        },
        "contributors": [label for label, _ in raw_pairs],
        "backend": backend,
    }
