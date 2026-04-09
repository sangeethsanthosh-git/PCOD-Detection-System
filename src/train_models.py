"""Fast model training and hyperparameter tuning utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier


SEARCH_N_ITER = 8
SEARCH_CV = 3
SEARCH_N_JOBS = -1


def build_model_search_space(random_state: int = 42, n_jobs: int = -1) -> Dict[str, Tuple[object, Dict[str, List[object]]]]:
    """Define fast model set and randomized search spaces."""
    return {
        "RandomForest": (
            RandomForestClassifier(
                random_state=random_state,
                class_weight="balanced_subsample",
                n_jobs=n_jobs,
            ),
            {
                "n_estimators": [200, 400],
                "max_depth": [6, 10, None],
                "min_samples_split": [2, 4, 6],
                "min_samples_leaf": [1, 2, 3],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                random_state=random_state,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=n_jobs,
                verbosity=0,
            ),
            {
                "n_estimators": [200],
                "learning_rate": [0.03, 0.05],
                "max_depth": [3, 4],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        ),
        "LightGBM": (
            LGBMClassifier(
                random_state=random_state,
                objective="binary",
                n_jobs=n_jobs,
                verbose=-1,
            ),
            {
                "n_estimators": [200],
                "learning_rate": [0.05, 0.1],
                "num_leaves": [31, 70],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        ),
    }


def train_and_tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Tune and train all base models with randomized search."""
    cv = StratifiedKFold(n_splits=SEARCH_CV, shuffle=True, random_state=random_state)
    tuned_models: Dict[str, object] = {}
    summary_rows = []

    for model_name in ["RandomForest", "XGBoost", "LightGBM"]:
        best_estimator = None
        best_params = None
        best_score = None

        # Default to n_jobs=-1; fallback to n_jobs=1 for restricted environments.
        for worker_count in [SEARCH_N_JOBS, 1]:
            model, param_dist = build_model_search_space(
                random_state=random_state,
                n_jobs=worker_count,
            )[model_name]

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist,
                n_iter=SEARCH_N_ITER,
                scoring="f1",
                cv=cv,
                n_jobs=worker_count,
                random_state=random_state,
                verbose=0,
                refit=True,
            )
            try:
                search.fit(X_train, y_train)
                best_estimator = search.best_estimator_
                best_params = search.best_params_
                best_score = float(search.best_score_)
                break
            except PermissionError:
                if worker_count == 1:
                    raise
                print(f"{model_name}: parallel search blocked, retrying with n_jobs=1")

        if best_estimator is None:
            raise RuntimeError(f"Failed to train {model_name}.")

        tuned_models[model_name] = best_estimator
        summary_rows.append(
            {
                "model": model_name,
                "best_cv_f1": best_score,
                "best_params": str(best_params),
            }
        )
        print(f"{model_name} best params: {best_params}")

    summary_df = pd.DataFrame(summary_rows).sort_values("best_cv_f1", ascending=False)
    return tuned_models, summary_df
