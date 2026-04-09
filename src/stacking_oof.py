"""Out-of-fold (OOF) stacking utilities and meta-learner ensemble."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .utils import extract_positive_class_scores


def generate_oof_predictions(
    base_models: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame | None = None,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Generate leakage-safe out-of-fold predictions for stacking features.

    Returns:
    - oof_train_features: one OOF prediction column per base model
    - oof_test_features: averaged fold predictions for test set (if X_test provided)
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    oof_train = pd.DataFrame(index=X_train.index)
    oof_test = pd.DataFrame(index=X_test.index) if X_test is not None else None

    for model_name, model in base_models.items():
        train_preds = np.zeros(len(X_train), dtype=float)
        test_fold_preds = []

        for train_idx, valid_idx in cv.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_valid = X_train.iloc[valid_idx]

            fold_model = clone(model)
            fold_model.fit(X_fold_train, y_fold_train)

            valid_scores = _scores_from_model(fold_model, X_fold_valid)
            train_preds[valid_idx] = valid_scores

            if X_test is not None:
                test_scores = _scores_from_model(fold_model, X_test)
                test_fold_preds.append(test_scores)

        column_name = f"{model_name}_oof"
        oof_train[column_name] = train_preds
        if oof_test is not None and test_fold_preds:
            oof_test[column_name] = np.mean(np.vstack(test_fold_preds), axis=0)

    return oof_train, oof_test


class OOFStackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Stacking ensemble trained on OOF meta-features with LogisticRegression meta-model.

    Base model scores (OOF) are used as meta-features during training to avoid leakage.
    """

    def __init__(
        self,
        base_models: Dict[str, object],
        meta_model: object | None = None,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.base_models = base_models
        self.meta_model = (
            meta_model
            if meta_model is not None
            else LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state)
        )
        self.cv_folds = cv_folds
        self.random_state = random_state

        self.fitted_base_models_: Dict[str, object] = {}
        self.meta_model_: object | None = None
        self.oof_train_features_: pd.DataFrame | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit OOF meta-model and base models on full training data."""
        oof_train, _ = generate_oof_predictions(
            base_models=self.base_models,
            X_train=X,
            y_train=y,
            X_test=None,
            cv_folds=self.cv_folds,
            random_state=self.random_state,
        )
        self.oof_train_features_ = oof_train

        self.meta_model_ = clone(self.meta_model)
        self.meta_model_.fit(oof_train, y)

        self.fitted_base_models_ = {}
        for model_name, model in self.base_models.items():
            fitted = clone(model)
            fitted.fit(X, y)
            self.fitted_base_models_[model_name] = fitted

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities through stacked base-model scores."""
        if self.meta_model_ is None:
            raise RuntimeError("OOFStackingEnsemble must be fitted before predict_proba.")

        meta_features = self._meta_features(X)
        if hasattr(self.meta_model_, "predict_proba"):
            return self.meta_model_.predict_proba(meta_features)

        decision = self.meta_model_.decision_function(meta_features)
        probs_pos = 1.0 / (1.0 + np.exp(-decision))
        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels using 0.5 probability threshold."""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] >= 0.5).astype(int)

    def _meta_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Build meta-feature dataframe from fitted base model scores."""
        if not self.fitted_base_models_:
            raise RuntimeError("OOFStackingEnsemble must be fitted before prediction.")

        features = pd.DataFrame(index=X.index)
        for model_name, model in self.fitted_base_models_.items():
            features[f"{model_name}_oof"] = _scores_from_model(model, X)
        return features


def build_stacking_base_models(trained_models: Dict[str, object]) -> Dict[str, object]:
    """Select required base estimators for stacking."""
    required = ["RandomForest", "XGBoost", "LightGBM", "GradientBoosting"]
    missing = [name for name in required if name not in trained_models]
    if missing:
        raise KeyError(f"Missing base models required for stacking: {missing}")
    return {name: trained_models[name] for name in required}


def train_stacking_classifier(
    base_models: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
    n_jobs: int = 1,
) -> StackingClassifier:
    """Train sklearn StackingClassifier with required base models and logistic meta-model."""
    required = ["RandomForest", "XGBoost", "LightGBM", "GradientBoosting"]
    missing = [name for name in required if name not in base_models]
    if missing:
        raise KeyError(f"Missing base models for StackingClassifier: {missing}")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    stacking_model = StackingClassifier(
        estimators=[
            ("rf", clone(base_models["RandomForest"])),
            ("xgb", clone(base_models["XGBoost"])),
            ("lgb", clone(base_models["LightGBM"])),
            ("gb", clone(base_models["GradientBoosting"])),
        ],
        final_estimator=LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        ),
        cv=cv,
        n_jobs=n_jobs,
        passthrough=True,
    )
    stacking_model.fit(X_train, y_train)
    return stacking_model


def _scores_from_model(model, X: pd.DataFrame) -> np.ndarray:
    """Get continuous scores for stacking meta-features."""
    scores = extract_positive_class_scores(model, X)
    if scores is not None:
        return np.asarray(scores, dtype=float)
    return np.asarray(model.predict(X), dtype=float)
