"""Feature selection with RandomForest feature importance."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def select_top_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    top_k: int = 20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame]:
    """Fit RF importance model on training data and keep top-k features."""
    if X_train.empty:
        raise ValueError("Feature selection received an empty feature matrix.")

    selector = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=1,
    )
    selector.fit(X_train, y_train)

    importance_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": selector.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    selected_features = importance_df.head(min(top_k, X_train.shape[1]))["feature"].tolist()
    importance_df["selected"] = importance_df["feature"].isin(selected_features)

    return (
        X_train[selected_features].copy(),
        X_test[selected_features].copy(),
        selected_features,
        importance_df,
    )
