"""Analytics helpers for dashboard and chart data."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from pcos_project.resource_utils import resource_path_obj


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = resource_path_obj("results")
REQUIRED_REFERENCE_COLUMNS = {
    "age_yrs",
    "bmi",
    "cycle_length_days",
    "pcos_y_n",
    "amh_ng_ml",
    "lh_miu_ml",
    "fsh_miu_ml",
    "i_beta_hcg_miu_ml",
    "ii_beta_hcg_miu_ml",
    "follicle_no_l",
    "follicle_no_r",
    "endometrium_mm",
}


def resolve_dataset_path() -> Path:
    """Locate the analytics reference dataset in source or packaged layouts."""
    env_path = os.getenv("PCOS_DATASET_PATH", "").strip()
    candidates = []
    checked_paths: list[str] = []
    invalid_schema_paths: list[str] = []
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            resource_path_obj("dataset/data1.csv"),
            resource_path_obj("dataset/data3.csv"),
            BASE_DIR / "dataset" / "data1.csv",
            BASE_DIR / "dataset" / "data3.csv",
        ]
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        checked_paths.append(str(candidate))
        if not candidate.exists():
            continue
        if _is_reference_dataset(candidate):
            return candidate
        invalid_schema_paths.append(str(candidate))

    details = [f"Checked: {', '.join(checked_paths)}."]
    if invalid_schema_paths:
        details.append(
            "Found incompatible dataset files: "
            + ", ".join(invalid_schema_paths)
            + ". The prediction workflow requires the clinical reference dataset with PCOS feature columns."
        )

    raise FileNotFoundError("Could not locate a compatible packaged dataset. " + " ".join(details))


def _is_reference_dataset(candidate: Path) -> bool:
    """Return True when the CSV has the clinical columns required by analytics and symptom training."""
    try:
        header_frame = pd.read_csv(candidate, nrows=0)
    except Exception:
        return False

    normalized_columns = {normalize_column_name(column) for column in header_frame.columns}
    return REQUIRED_REFERENCE_COLUMNS.issubset(normalized_columns)


def normalize_column_name(value: str) -> str:
    """Convert source column names into project-safe snake_case names."""
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


@lru_cache(maxsize=1)
def load_reference_frame() -> pd.DataFrame:
    """Load and normalize the clinical dataset once for analytics and defaults."""
    df = pd.read_csv(resolve_dataset_path())
    df = df.rename(columns={column: normalize_column_name(column) for column in df.columns})
    df = df.drop(columns=[column for column in ("unnamed_44",) if column in df.columns])

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["beta_hcg_ratio"] = df["i_beta_hcg_miu_ml"] / (df["ii_beta_hcg_miu_ml"] + 1.0)
    df["lh_fsh_ratio"] = df["lh_miu_ml"] / (df["fsh_miu_ml"] + 1.0)
    df["follicle_total"] = df["follicle_no_l"] + df["follicle_no_r"]
    df["waist_hip_ratio"] = df["waist_inch"] / df["hip_inch"].replace(0, pd.NA)

    numeric_medians = df.median(numeric_only=True)
    return df.fillna(numeric_medians)


@lru_cache(maxsize=1)
def get_default_feature_values() -> dict[str, float]:
    """Return median defaults for model/scaler features."""
    df = load_reference_frame()
    return {key: float(value) for key, value in df.median(numeric_only=True).to_dict().items()}


def build_dashboard_summary() -> dict[str, str]:
    """Return top-level dashboard stats for the home page."""
    model_metrics_path = RESULTS_DIR / "model_metrics.csv"
    df = load_reference_frame()
    summary = {
        "patients_analyzed": f"{len(df)}",
        "prediction_accuracy": "Model ready",
        "high_risk_cases": f"{int(df['pcos_y_n'].sum())}",
        "recent_predictions": "Dual-mode ready",
    }
    if model_metrics_path.exists():
        metrics_df = pd.read_csv(model_metrics_path).sort_values("f1", ascending=False)
        if not metrics_df.empty:
            best = metrics_df.iloc[0]
            summary["prediction_accuracy"] = f"{float(best['accuracy']) * 100:.1f}%"
    return summary


def build_analysis_payload(
    prediction_inputs: dict[str, float] | None = None,
    probability: float | None = None,
    mode: str = "basic",
    contribution_chart: dict[str, list[Any]] | None = None,
    ai_explanation: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Construct Chart.js-ready analytics payload."""
    df = load_reference_frame()
    current = prediction_inputs or get_default_feature_values()
    current_probability = float(probability if probability is not None else df["pcos_y_n"].mean())

    age_curve = _risk_curve(df, "age_yrs", [15, 20, 25, 30, 35, 40, 45, 50])
    bmi_curve = _risk_curve(df, "bmi", [15, 20, 25, 30, 35, 40, 50])

    return {
        "has_prediction": True,
        "message": "",
        "mode": mode,
        "mode_label": "Clinical Assessment" if mode == "clinical" else "Basic Screening",
        "risk_probability": {
            "labels": ["Predicted risk", "Remaining"],
            "values": [round(current_probability * 100, 2), round(100 - (current_probability * 100), 2)],
            "probability": round(current_probability, 4),
        },
        "bmi_vs_risk": {
            "points": bmi_curve,
            "patient": {
                "x": round(float(current.get("bmi", get_default_feature_values()["bmi"])), 2),
                "y": round(current_probability * 100, 2),
            },
        },
        "age_vs_risk": {
            "points": age_curve,
            "patient": {
                "x": round(float(current.get("age_yrs", get_default_feature_values()["age_yrs"])), 2),
                "y": round(current_probability * 100, 2),
            },
        },
        "symptom_contribution": contribution_chart
        or {
            "labels": ["Irregular periods", "High BMI", "Acne / pimples", "Hair growth"],
            "values": [24, 18, 13, 12],
        },
        "feature_importance": contribution_chart
        or {
            "labels": ["Irregular periods", "High BMI", "Acne / pimples", "Hair growth"],
            "values": [24, 18, 13, 12],
        },
        "ai_explanation": ai_explanation or {},
    }


def build_empty_analysis_payload() -> dict[str, Any]:
    """Return a chart-safe payload when no prediction exists yet."""
    return {
        "has_prediction": False,
        "message": "Run a prediction to view personalized PCOS analysis charts.",
        "mode": "basic",
        "mode_label": "Basic Screening",
        "risk_probability": {
            "labels": ["Predicted risk", "Remaining"],
            "values": [0, 100],
            "probability": 0,
        },
        "bmi_vs_risk": {
            "points": [],
            "patient": {"x": 0, "y": 0},
        },
        "age_vs_risk": {
            "points": [],
            "patient": {"x": 0, "y": 0},
        },
        "symptom_contribution": {
            "labels": [],
            "values": [],
        },
        "feature_importance": {
            "labels": [],
            "values": [],
        },
        "ai_explanation": {},
    }


def _risk_curve(df: pd.DataFrame, column: str, bins: list[int]) -> list[dict[str, float]]:
    grouped = (
        df.assign(_bin=pd.cut(df[column], bins=bins, include_lowest=True))
        .groupby("_bin", observed=False)["pcos_y_n"]
        .mean()
        .mul(100)
        .reset_index()
    )

    points: list[dict[str, float]] = []
    for _, row in grouped.iterrows():
        interval = row["_bin"]
        value = row["pcos_y_n"]
        if pd.isna(value):
            continue
        midpoint = (float(interval.left) + float(interval.right)) / 2
        points.append({"x": round(midpoint, 2), "y": round(float(value), 2)})
    return points
