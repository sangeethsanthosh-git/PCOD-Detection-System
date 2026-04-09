"""Stacking ensemble training utilities."""

from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


STACKING_BASE_MODELS = ("RandomForest", "XGBoost", "LightGBM")


def train_stacking_ensemble(
    tuned_models: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> StackingClassifier:
    """Train stacking ensemble with RF/XGB/LGBM and LogisticRegression meta-model."""
    missing = [name for name in STACKING_BASE_MODELS if name not in tuned_models]
    if missing:
        raise KeyError(f"Missing base models for stacking: {missing}")

    stack_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    for worker_count in [-1, 1]:
        stacking_model = StackingClassifier(
            estimators=[
                ("rf", clone(tuned_models["RandomForest"])),
                ("xgb", clone(tuned_models["XGBoost"])),
                ("lgb", clone(tuned_models["LightGBM"])),
            ],
            final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
            cv=stack_cv,
            n_jobs=worker_count,
            passthrough=True,
        )
        try:
            stacking_model.fit(X_train, y_train)
            return stacking_model
        except PermissionError:
            if worker_count == 1:
                raise
            print("Stacking: parallel training blocked, retrying with n_jobs=1")

    raise RuntimeError("Failed to train stacking ensemble.")
