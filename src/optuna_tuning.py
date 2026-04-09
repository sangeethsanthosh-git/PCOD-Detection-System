"""Bayesian hyperparameter optimization with Optuna."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency
    optuna = None  # type: ignore[assignment]

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


def tune_models_with_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    cv_folds: int = 5,
    random_state: int = 42,
    n_jobs: int = 1,
) -> Tuple[Dict[str, Dict[str, object]], pd.DataFrame]:
    """
    Tune model hyperparameters using Optuna Bayesian optimization.

    Tuned models:
    - RandomForest
    - XGBoost
    - LightGBM
    - CatBoost
    """
    requested_trials = max(1, int(n_trials))
    max_samples = safe_int_from_env("PCOS_OPTUNA_MAX_SAMPLES", 30000)
    X_tune, y_tune = stratified_subsample(
        X=X_train,
        y=y_train,
        max_samples=max_samples,
        random_state=random_state,
    )

    if optuna is None:
        summary = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "status": "skipped",
                    "best_cv_f1": np.nan,
                    "n_trials": requested_trials,
                    "note": "optuna not installed",
                }
                for model_name in ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]
            ]
        )
        return {}, summary

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    is_binary = len(np.unique(y_tune)) == 2
    scorer = make_scorer(f1_score, average="binary" if is_binary else "weighted", zero_division=0)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    studies: Dict[str, Tuple[dict | None, float | None, str]] = {}

    studies["RandomForest"] = _run_study(
        model_name="RandomForest",
        objective_fn=lambda trial: _rf_objective(trial, X_tune, y_tune, cv, scorer, random_state, n_jobs),
        n_trials=requested_trials,
        random_state=random_state,
    )

    if XGBClassifier is not None:
        studies["XGBoost"] = _run_study(
            model_name="XGBoost",
            objective_fn=lambda trial: _xgb_objective(trial, X_tune, y_tune, cv, scorer, random_state, n_jobs),
            n_trials=requested_trials,
            random_state=random_state,
        )
    else:
        studies["XGBoost"] = (None, None, "xgboost not installed")

    if LGBMClassifier is not None:
        studies["LightGBM"] = _run_study(
            model_name="LightGBM",
            objective_fn=lambda trial: _lgb_objective(trial, X_tune, y_tune, cv, scorer, random_state, n_jobs),
            n_trials=requested_trials,
            random_state=random_state,
        )
    else:
        studies["LightGBM"] = (None, None, "lightgbm not installed")

    if CatBoostClassifier is not None:
        studies["CatBoost"] = _run_study(
            model_name="CatBoost",
            objective_fn=lambda trial: _cat_objective(trial, X_tune, y_tune, cv, scorer, random_state),
            n_trials=requested_trials,
            random_state=random_state,
        )
    else:
        studies["CatBoost"] = (None, None, "catboost not installed")

    tuned_params: Dict[str, Dict[str, object]] = {}
    rows = []
    for model_name, (best_params, best_score, note) in studies.items():
        status = "tuned" if best_params is not None else "skipped"
        if best_params is not None:
            tuned_params[model_name] = best_params
        rows.append(
            {
                "model": model_name,
                "status": status,
                "best_cv_f1": float(best_score) if best_score is not None else np.nan,
                "n_trials": requested_trials,
                "note": note,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("best_cv_f1", ascending=False, na_position="last")
    return tuned_params, summary_df


def _run_study(
    model_name: str,
    objective_fn,
    n_trials: int,
    random_state: int,
) -> Tuple[dict | None, float | None, str]:
    """Run one Optuna study with error-safe fallback."""
    try:
        sampler = optuna.samplers.TPESampler(seed=random_state)  # type: ignore[union-attr]
        study = optuna.create_study(direction="maximize", sampler=sampler)  # type: ignore[union-attr]
        study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
        return dict(study.best_trial.params), float(study.best_value), ""
    except Exception as error:  # pragma: no cover - runtime safety
        return None, None, f"{model_name} tuning failed: {type(error).__name__}: {error}"


def _rf_objective(trial, X_train, y_train, cv, scorer, random_state: int, n_jobs: int) -> float:
    """Optuna objective for RandomForest."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900),
        "max_depth": trial.suggest_int("max_depth", 3, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }
    model = RandomForestClassifier(
        **params,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=n_jobs,
    )
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scorer, n_jobs=n_jobs)
    return float(scores.mean())


def _xgb_objective(trial, X_train, y_train, cv, scorer, random_state: int, n_jobs: int) -> float:
    """Optuna objective for XGBoost."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 250, 1200),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
    }
    model = XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=0,
    )
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scorer, n_jobs=n_jobs)
    return float(scores.mean())


def _lgb_objective(trial, X_train, y_train, cv, scorer, random_state: int, n_jobs: int) -> float:
    """Optuna objective for LightGBM."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 250, 1200),
        "num_leaves": trial.suggest_int("num_leaves", 20, 120),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
    }
    model = LGBMClassifier(
        **params,
        objective="binary",
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=-1,
    )
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scorer, n_jobs=n_jobs)
    return float(scores.mean())


def _cat_objective(trial, X_train, y_train, cv, scorer, random_state: int) -> float:
    """Optuna objective for CatBoost."""
    params = {
        "iterations": trial.suggest_int("iterations", 250, 1200),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
    }
    model = CatBoostClassifier(
        **params,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_state,
        verbose=False,
    )
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scorer, n_jobs=1)
    return float(scores.mean())
