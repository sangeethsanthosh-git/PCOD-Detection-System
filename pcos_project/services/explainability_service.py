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
    """Return top local feature contributions for a single-row tree-model prediction."""
    if shap is None:
        raise ImportError("The shap package is required for runtime explanations.")

    if X_row.empty:
        return {
            "ai_explanation": {},
            "contribution_chart": {"labels": [], "values": []},
            "contributors": [],
            "backend": "shap_tree",
        }

    explainer = _get_tree_explainer(model)
    shap_values = explainer.shap_values(X_row)
    values = _coerce_shap_values(shap_values)
    row_values = np.abs(values[0])

    ranked_indices = np.argsort(row_values)[::-1]
    top_indices = [index for index in ranked_indices if row_values[index] > 0][:top_n]

    if not top_indices:
        return {
            "ai_explanation": {},
            "contribution_chart": {"labels": [], "values": []},
            "contributors": [],
            "backend": "shap_tree",
        }

    raw_pairs = []
    for index in top_indices:
        feature_name = X_row.columns[index]
        label = feature_labels.get(feature_name, feature_name.replace("_", " ").title())
        raw_pairs.append((label, float(row_values[index])))

    total = sum(value for _, value in raw_pairs) or 1.0
    ai_explanation = {label: round(value, 4) for label, value in raw_pairs}
    contribution_chart = {
        "labels": [label for label, _ in raw_pairs],
        "values": [round((value / total) * 100, 1) for _, value in raw_pairs],
    }
    return {
        "ai_explanation": ai_explanation,
        "contribution_chart": contribution_chart,
        "contributors": [label for label, _ in raw_pairs],
        "backend": "shap_tree",
    }


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
