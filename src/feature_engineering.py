"""Clinical feature engineering and correlation-based pruning."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def engineer_features(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, bool]]:
    """Create additional domain-driven features when source columns are present."""
    engineered = dataframe.copy()
    created_flags = {
        "beta_hcg_ratio": False,
        "lh_fsh_ratio": False,
        "bmi_category": False,
        "follicle_total": False,
    }

    if "target" in engineered.columns:
        features = engineered.drop(columns=["target"]).copy()
    else:
        features = engineered

    # Ensure candidate numeric columns are castable for arithmetic operations.
    for column in features.columns:
        coerced = pd.to_numeric(features[column], errors="coerce")
        if coerced.notna().mean() >= 0.90:
            features[column] = coerced

    # 1) beta_hcg_ratio = I / (II + 1)
    col_i_hcg = _find_first_existing(
        features.columns.tolist(),
        ["i_beta_hcg_miu_ml", "i_beta_hcg", "i_beta_hcg_ml"],
    )
    col_ii_hcg = _find_first_existing(
        features.columns.tolist(),
        ["ii_beta_hcg_miu_ml", "ii_beta_hcg", "ii_beta_hcg_ml"],
    )
    if col_i_hcg and col_ii_hcg:
        i_hcg = pd.to_numeric(features[col_i_hcg], errors="coerce")
        ii_hcg = pd.to_numeric(features[col_ii_hcg], errors="coerce")
        features["beta_hcg_ratio"] = i_hcg / (ii_hcg + 1.0)
        created_flags["beta_hcg_ratio"] = True

    # 2) lh_fsh_ratio = LH / (FSH + 1)
    col_lh = _find_first_existing(features.columns.tolist(), ["lh_miu_ml", "lh"])
    col_fsh = _find_first_existing(features.columns.tolist(), ["fsh_miu_ml", "fsh"])
    if col_lh and col_fsh:
        lh_vals = pd.to_numeric(features[col_lh], errors="coerce")
        fsh_vals = pd.to_numeric(features[col_fsh], errors="coerce")
        features["lh_fsh_ratio"] = lh_vals / (fsh_vals + 1.0)
        created_flags["lh_fsh_ratio"] = True

    # 3) bmi_category from BMI.
    col_bmi = _find_first_existing(features.columns.tolist(), ["bmi"])
    if col_bmi:
        bmi_values = pd.to_numeric(features[col_bmi], errors="coerce")
        # Handle both numeric BMI and textual BMI categories.
        if bmi_values.notna().mean() >= 0.50:
            bins = [-np.inf, 18.5, 25.0, 30.0, np.inf]
            labels = ["underweight", "normal", "overweight", "obese"]
            features["bmi_category"] = pd.cut(bmi_values, bins=bins, labels=labels).astype(str)
        else:
            features["bmi_category"] = features[col_bmi].astype(str)
        created_flags["bmi_category"] = True

    # 4) follicle_total = left + right follicle count.
    col_fol_l = _find_first_existing(features.columns.tolist(), ["follicle_no_left", "follicle_no_l"])
    col_fol_r = _find_first_existing(features.columns.tolist(), ["follicle_no_right", "follicle_no_r"])
    if col_fol_l and col_fol_r:
        left = pd.to_numeric(features[col_fol_l], errors="coerce")
        right = pd.to_numeric(features[col_fol_r], errors="coerce")
        features["follicle_total"] = left + right
        created_flags["follicle_total"] = True

    if "target" in engineered.columns:
        features["target"] = engineered["target"].values
    return features, created_flags


def remove_highly_correlated_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.95,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Drop highly correlated numeric features using train data only."""
    numeric_train = X_train.select_dtypes(include=[np.number]).copy()
    if numeric_train.empty:
        return X_train, X_test, []

    correlation_matrix = numeric_train.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    columns_to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    reduced_train = X_train.drop(columns=columns_to_drop, errors="ignore")
    reduced_test = X_test.drop(columns=columns_to_drop, errors="ignore")
    return reduced_train, reduced_test, columns_to_drop


def _find_first_existing(columns: List[str], candidates: List[str]) -> str | None:
    """Return the first candidate column that exists in the dataframe schema."""
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None
