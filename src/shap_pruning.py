"""SHAP-based feature pruning for model simplification and robustness."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None  # type: ignore[assignment]

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore[assignment]

from .utils import safe_int_from_env, stratified_subsample


def prune_features_with_shap(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    drop_fraction: float = 0.20,
    min_features: int = 20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame, str]:
    """
    Prune weakest features using SHAP importance from an XGBoost model.

    Fallback strategy:
    - If SHAP is unavailable but XGBoost exists, use XGBoost feature importances.
    """
    if XGBClassifier is None:
        raise ImportError("XGBoost is required for SHAP-based pruning.")

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    if shap is not None:
        shap_sample_size = safe_int_from_env("PCOS_SHAP_MAX_SAMPLES", 20000)
        X_shap, _ = stratified_subsample(
            X=X_train,
            y=y_train,
            max_samples=shap_sample_size,
            random_state=random_state,
        )
        importance = _compute_shap_importance(model, X_shap)
        method = "shap"
    else:
        importance = pd.Series(model.feature_importances_, index=X_train.columns, name="importance")
        method = "xgboost_feature_importance_fallback"

    ranked = importance.fillna(0.0).sort_values(ascending=False)
    keep_count = _resolve_keep_count(
        n_features=len(ranked),
        drop_fraction=drop_fraction,
        min_features=min_features,
    )
    selected_features = ranked.head(keep_count).index.tolist()

    importance_df = pd.DataFrame(
        {
            "feature": ranked.index,
            "importance": ranked.values,
        }
    )
    importance_df["selected"] = importance_df["feature"].isin(selected_features)
    importance_df["method"] = method

    return (
        X_train[selected_features].copy(),
        X_test[selected_features].copy(),
        selected_features,
        importance_df,
        method,
    )


def _compute_shap_importance(model, X: pd.DataFrame) -> pd.Series:
    """Compute mean absolute SHAP values per feature for tree model."""
    explainer = shap.TreeExplainer(model)  # type: ignore[union-attr]
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        # For binary classification, use class-1 explanation when returned as list.
        values = np.asarray(shap_values[-1], dtype=float)
    elif hasattr(shap_values, "values"):
        values = np.asarray(shap_values.values, dtype=float)
    else:
        values = np.asarray(shap_values, dtype=float)

    if values.ndim == 3:
        # SHAP may return (n_samples, n_features, n_classes)
        values = values[:, :, -1]

    mean_abs = np.mean(np.abs(values), axis=0)
    return pd.Series(mean_abs, index=X.columns, name="importance")


def _resolve_keep_count(n_features: int, drop_fraction: float, min_features: int) -> int:
    """Compute number of features to keep after pruning."""
    min_features = max(1, int(min_features))
    drop_fraction = float(np.clip(drop_fraction, 0.0, 0.95))
    keep_count = int(round(n_features * (1.0 - drop_fraction)))
    keep_count = max(min_features, keep_count)
    keep_count = min(n_features, keep_count)
    return keep_count
