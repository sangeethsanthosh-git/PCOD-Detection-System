"""Training utilities for strong baseline models used in the ensemble stack."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore[assignment]

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None  # type: ignore[assignment]

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - optional dependency
    CatBoostClassifier = None  # type: ignore[assignment]

from .utils import safe_int_from_env, stratified_subsample


DEFAULT_MAX_CV_SAMPLES = 30000
DEFAULT_MAX_SVM_SAMPLES = 20000


def train_base_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tuned_params: Dict[str, Dict[str, object]] | None = None,
    cv_folds: int = 5,
    random_state: int = 42,
    n_jobs: int | None = None,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """
    Train and cross-validate strong base models for ensemble learning.

    Models:
    - RandomForestClassifier
    - GradientBoostingClassifier
    - XGBClassifier
    - LightGBMClassifier
    - SupportVectorMachine
    - CatBoostClassifier
    """
    resolved_n_jobs = _resolve_n_jobs(n_jobs)
    tuned_params = tuned_params or {}
    is_binary = len(np.unique(y_train)) == 2

    model_defs = build_base_model_definitions(
        random_state=random_state,
        n_jobs=resolved_n_jobs,
        tuned_params=tuned_params,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "f1": make_scorer(f1_score, average="binary" if is_binary else "weighted", zero_division=0),
        "roc_auc": "roc_auc" if is_binary else "roc_auc_ovr",
    }

    trained_models: Dict[str, object] = {}
    summary_rows = []

    for model_name, estimator in model_defs.items():
        if estimator is None:
            summary_rows.append(
                {
                    "model": model_name,
                    "status": "skipped",
                    "cv_f1_mean": np.nan,
                    "cv_f1_std": np.nan,
                    "cv_roc_auc_mean": np.nan,
                    "cv_roc_auc_std": np.nan,
                    "train_samples_used": 0,
                    "note": "missing dependency",
                }
            )
            continue

        # Keep CV tractable on large data; SVM gets stricter cap.
        sample_limit = _resolve_cv_sample_limit(model_name)
        X_cv, y_cv = stratified_subsample(
            X=X_train,
            y=y_train,
            max_samples=sample_limit,
            random_state=random_state,
        )

        cv_results = cross_validate(
            estimator=clone(estimator),
            X=X_cv,
            y=y_cv,
            scoring=scoring,
            cv=cv,
            n_jobs=resolved_n_jobs,
            error_score="raise",
        )

        # Fit final model on all training samples except SVM which is capped.
        fit_limit = _resolve_fit_sample_limit(model_name)
        X_fit, y_fit = stratified_subsample(
            X=X_train,
            y=y_train,
            max_samples=fit_limit,
            random_state=random_state,
        )
        fitted = clone(estimator)
        fitted.fit(X_fit, y_fit)
        trained_models[model_name] = fitted

        summary_rows.append(
            {
                "model": model_name,
                "status": "trained",
                "cv_f1_mean": float(np.mean(cv_results["test_f1"])),
                "cv_f1_std": float(np.std(cv_results["test_f1"])),
                "cv_roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
                "cv_roc_auc_std": float(np.std(cv_results["test_roc_auc"])),
                "train_samples_used": int(len(X_fit)),
                "note": "",
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["status", "cv_f1_mean"],
        ascending=[True, False],
        na_position="last",
    )
    return trained_models, summary_df


def build_base_model_definitions(
    random_state: int = 42,
    n_jobs: int = 1,
    tuned_params: Dict[str, Dict[str, object]] | None = None,
) -> Dict[str, object | None]:
    """Create model objects with optional tuned parameters."""
    tuned_params = tuned_params or {}

    def _apply_params(model_name: str, estimator):
        params = tuned_params.get(model_name, {})
        if params:
            estimator.set_params(**params)
        return estimator

    models: Dict[str, object | None] = {
        "RandomForest": _apply_params(
            "RandomForest",
            RandomForestClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_split=2,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=n_jobs,
            ),
        ),
        "GradientBoosting": _apply_params(
            "GradientBoosting",
            GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=random_state,
            ),
        ),
        "SupportVectorMachine": _apply_params(
            "SupportVectorMachine",
            SVC(
                C=10.0,
                kernel="rbf",
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = _apply_params(
            "XGBoost",
            XGBClassifier(
                n_estimators=600,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.8,
                min_child_weight=1,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=random_state,
                n_jobs=n_jobs,
                verbosity=0,
            ),
        )
    else:
        models["XGBoost"] = None

    if LGBMClassifier is not None:
        models["LightGBM"] = _apply_params(
            "LightGBM",
            LGBMClassifier(
                n_estimators=600,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.8,
                objective="binary",
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=-1,
            ),
        )
    else:
        models["LightGBM"] = None

    if CatBoostClassifier is not None:
        models["CatBoost"] = _apply_params(
            "CatBoost",
            CatBoostClassifier(
                iterations=600,
                depth=6,
                learning_rate=0.03,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=random_state,
                verbose=False,
            ),
        )
    else:
        models["CatBoost"] = None

    return models


def _resolve_n_jobs(n_jobs: int | None) -> int:
    """Resolve parallel worker count from input/env."""
    if n_jobs is not None:
        return max(1, int(n_jobs))
    return safe_int_from_env("PCOS_N_JOBS", 1)


def _resolve_cv_sample_limit(model_name: str) -> int:
    """Resolve max CV sample size to keep folds computationally feasible."""
    base_limit = safe_int_from_env("PCOS_MAX_CV_SAMPLES", DEFAULT_MAX_CV_SAMPLES)
    if model_name == "SupportVectorMachine":
        return min(base_limit, safe_int_from_env("PCOS_MAX_SVM_CV_SAMPLES", DEFAULT_MAX_SVM_SAMPLES))
    return base_limit


def _resolve_fit_sample_limit(model_name: str) -> int:
    """Resolve max final-fit sample size for expensive estimators."""
    if model_name == "SupportVectorMachine":
        return safe_int_from_env("PCOS_MAX_SVM_FIT_SAMPLES", DEFAULT_MAX_SVM_SAMPLES)
    return safe_int_from_env("PCOS_MAX_FINAL_FIT_SAMPLES", 0)
