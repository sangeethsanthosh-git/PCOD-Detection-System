"""Weighted model blending utilities."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

DEFAULT_BLEND_WEIGHTS = {
    "XGBoost": 0.30,
    "LightGBM": 0.25,
    "RandomForest": 0.20,
    "GradientBoosting": 0.15,
    "CatBoost": 0.10,
}


class WeightedBlendEnsemble(BaseEstimator, ClassifierMixin):
    """Blend model predictions using fixed probability weights."""

    def __init__(self, models: Dict[str, object], weights: Dict[str, float], threshold: float = 0.5) -> None:
        self.models = models
        self.weights = weights
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        No-op fit for pre-trained base models.

        The blended ensemble is constructed from already fitted model objects.
        """
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Compute weighted average class probabilities."""
        probs_pos = np.zeros(len(X), dtype=float)
        for model_name, weight in self.weights.items():
            model = self.models[model_name]
            probs_pos += float(weight) * _positive_probability(model, X)

        probs_pos = np.clip(probs_pos, 0.0, 1.0)
        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary labels using configurable threshold."""
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def create_blended_ensemble(
    trained_models: Dict[str, object],
    requested_weights: Dict[str, float] | None = None,
) -> Tuple[WeightedBlendEnsemble, Dict[str, float], list[str]]:
    """
    Create blended ensemble from available models and normalized weights.

    If some requested models are missing, their weights are redistributed among
    available models proportionally.
    """
    requested_weights = requested_weights or DEFAULT_BLEND_WEIGHTS.copy()

    available_weights = {name: w for name, w in requested_weights.items() if name in trained_models}
    missing_models = [name for name in requested_weights if name not in trained_models]
    if not available_weights:
        raise ValueError("No requested blend models are available for blending.")

    total_weight = float(sum(available_weights.values()))
    normalized_weights = {name: float(weight / total_weight) for name, weight in available_weights.items()}

    ensemble = WeightedBlendEnsemble(models=trained_models, weights=normalized_weights)
    return ensemble, normalized_weights, missing_models


def _positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    """Return estimated positive class probability for any binary classifier."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return np.asarray(probabilities[:, 1], dtype=float)

    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-decision))

    return np.asarray(model.predict(X), dtype=float)
