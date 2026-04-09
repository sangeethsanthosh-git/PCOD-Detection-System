"""Prediction services for public screening and clinical assessment modes."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from services.analytics_service import get_default_feature_values, load_reference_frame
from services.explainability_service import build_local_explanation, fallback_explanation_from_chart
from pcos_project.resource_utils import resource_path_obj


BASE_DIR = Path(__file__).resolve().parents[1]

MODE_BASIC = "basic"
MODE_CLINICAL = "clinical"

MODE_OPTIONS = [
    {
        "value": MODE_BASIC,
        "label": "Basic Screening",
        "description": "Public-friendly symptom screening using only observable symptoms and lifestyle information.",
    },
    {
        "value": MODE_CLINICAL,
        "label": "Clinical Assessment",
        "description": "Doctor-facing assessment that adds hormonal markers and ultrasound-style findings.",
    },
]

BOOLEAN_CHOICES = [
    {"value": "", "label": "Select"},
    {"value": "0", "label": "No"},
    {"value": "1", "label": "Yes"},
]
STRESS_CHOICES = [
    {"value": "", "label": "Select"},
    {"value": "low", "label": "Low"},
    {"value": "moderate", "label": "Moderate"},
    {"value": "high", "label": "High"},
]
ACTIVITY_CHOICES = [
    {"value": "", "label": "Select"},
    {"value": "low", "label": "Low"},
    {"value": "moderate", "label": "Moderate"},
    {"value": "high", "label": "High"},
]

STEP_DEFINITIONS = [
    {
        "slug": "personal",
        "title": "Step 1 - Personal Info",
        "description": "Baseline demographic and menstrual cycle details used by both prediction modes.",
    },
    {
        "slug": "symptoms",
        "title": "Step 2 - Symptoms",
        "description": "User-observable symptoms that support public screening.",
    },
    {
        "slug": "lifestyle",
        "title": "Step 3 - Lifestyle",
        "description": "Daily context that can change screening confidence and counseling needs.",
    },
    {
        "slug": "clinical",
        "title": "Step 4 - Clinical (Optional)",
        "description": "Hormonal markers and ovarian findings used when clinical data is available.",
        "clinical_only": True,
    },
]

FIELD_DEFINITIONS = [
    {
        "step": "personal",
        "name": "age_yrs",
        "label": "Age",
        "kind": "number",
        "input_type": "number",
        "min": 12,
        "max": 55,
        "step_size": "1",
        "required_modes": [MODE_BASIC, MODE_CLINICAL],
        "placeholder": "e.g. 29",
    },
    {
        "step": "personal",
        "name": "height_cm",
        "label": "Height (cm)",
        "kind": "number",
        "input_type": "number",
        "min": 120,
        "max": 210,
        "step_size": "0.1",
        "required_modes": [MODE_BASIC, MODE_CLINICAL],
        "placeholder": "e.g. 160",
    },
    {
        "step": "personal",
        "name": "weight_kg",
        "label": "Weight (kg)",
        "kind": "number",
        "input_type": "number",
        "min": 25,
        "max": 180,
        "step_size": "0.1",
        "required_modes": [MODE_BASIC, MODE_CLINICAL],
        "placeholder": "e.g. 62",
    },
    {
        "step": "personal",
        "name": "cycle_length_days",
        "label": "Menstrual cycle length (days)",
        "kind": "number",
        "input_type": "number",
        "min": 1,
        "max": 90,
        "step_size": "1",
        "required_modes": [MODE_BASIC, MODE_CLINICAL],
        "placeholder": "e.g. 35",
    },
    {
        "step": "symptoms",
        "name": "irregular_periods",
        "label": "Irregular periods",
        "kind": "boolean",
        "input_type": "select",
        "choices": BOOLEAN_CHOICES,
        "required_modes": [MODE_BASIC, MODE_CLINICAL],
    },
    {
        "step": "symptoms",
        "name": "hair_growth_y_n",
        "label": "Excess hair growth",
        "kind": "boolean",
        "input_type": "select",
        "choices": BOOLEAN_CHOICES,
        "required_modes": [MODE_BASIC, MODE_CLINICAL],
    },
    {
        "step": "symptoms",
        "name": "pimples_y_n",
        "label": "Acne / pimples",
        "kind": "boolean",
        "input_type": "select",
        "choices": BOOLEAN_CHOICES,
        "required_modes": [MODE_BASIC, MODE_CLINICAL],
    },
    {
        "step": "symptoms",
        "name": "skin_darkening_y_n",
        "label": "Skin darkening",
        "kind": "boolean",
        "input_type": "select",
        "choices": BOOLEAN_CHOICES,
        "required_modes": [MODE_BASIC, MODE_CLINICAL],
    },
    {
        "step": "symptoms",
        "name": "hair_loss_y_n",
        "label": "Hair loss",
        "kind": "boolean",
        "input_type": "select",
        "choices": BOOLEAN_CHOICES,
        "required_modes": [MODE_BASIC, MODE_CLINICAL],
    },
    {
        "step": "symptoms",
        "name": "weight_gain_y_n",
        "label": "Recent weight gain",
        "kind": "boolean",
        "input_type": "select",
        "choices": BOOLEAN_CHOICES,
        "required_modes": [MODE_BASIC, MODE_CLINICAL],
    },
    {
        "step": "symptoms",
        "name": "family_history_pcos",
        "label": "Family history of PCOS",
        "kind": "boolean",
        "input_type": "select",
        "choices": BOOLEAN_CHOICES,
        "required_modes": [MODE_BASIC, MODE_CLINICAL],
    },
    {
        "step": "lifestyle",
        "name": "stress_level",
        "label": "Stress level",
        "kind": "choice",
        "input_type": "select",
        "choices": STRESS_CHOICES,
        "required_modes": [MODE_BASIC, MODE_CLINICAL],
    },
    {
        "step": "lifestyle",
        "name": "physical_activity_level",
        "label": "Physical activity level",
        "kind": "choice",
        "input_type": "select",
        "choices": ACTIVITY_CHOICES,
        "required_modes": [MODE_BASIC, MODE_CLINICAL],
    },
    {
        "step": "clinical",
        "name": "amh_ng_ml",
        "label": "AMH level (ng/mL)",
        "kind": "number",
        "input_type": "number",
        "min": 0,
        "max": 30,
        "step_size": "0.01",
        "required_modes": [MODE_CLINICAL],
        "placeholder": "e.g. 4.8",
    },
    {
        "step": "clinical",
        "name": "lh_miu_ml",
        "label": "LH level (mIU/mL)",
        "kind": "number",
        "input_type": "number",
        "min": 0,
        "max": 50,
        "step_size": "0.01",
        "required_modes": [MODE_CLINICAL],
        "placeholder": "e.g. 9.5",
    },
    {
        "step": "clinical",
        "name": "fsh_miu_ml",
        "label": "FSH level (mIU/mL)",
        "kind": "number",
        "input_type": "number",
        "min": 0,
        "max": 50,
        "step_size": "0.01",
        "required_modes": [MODE_CLINICAL],
        "placeholder": "e.g. 5.2",
    },
    {
        "step": "clinical",
        "name": "beta_hcg_miu_ml",
        "label": "Beta-HCG (mIU/mL)",
        "kind": "number",
        "input_type": "number",
        "min": 0,
        "max": 10000,
        "step_size": "0.01",
        "required_modes": [MODE_CLINICAL],
        "placeholder": "e.g. 2.5",
    },
    {
        "step": "clinical",
        "name": "follicle_no_l",
        "label": "Follicle count left",
        "kind": "number",
        "input_type": "number",
        "min": 0,
        "max": 60,
        "step_size": "1",
        "required_modes": [MODE_CLINICAL],
        "placeholder": "e.g. 12",
    },
    {
        "step": "clinical",
        "name": "follicle_no_r",
        "label": "Follicle count right",
        "kind": "number",
        "input_type": "number",
        "min": 0,
        "max": 60,
        "step_size": "1",
        "required_modes": [MODE_CLINICAL],
        "placeholder": "e.g. 14",
    },
    {
        "step": "clinical",
        "name": "endometrium_mm",
        "label": "Endometrium thickness (mm)",
        "kind": "number",
        "input_type": "number",
        "min": 0,
        "max": 30,
        "step_size": "0.1",
        "required_modes": [MODE_CLINICAL],
        "placeholder": "e.g. 8.3",
    },
]

FIELD_LOOKUP = {field["name"]: field for field in FIELD_DEFINITIONS}
CLINICAL_FIELDS = [field["name"] for field in FIELD_DEFINITIONS if MODE_CLINICAL in field["required_modes"] and field["step"] == "clinical"]
CORE_CLINICAL_FIELDS = ["amh_ng_ml", "lh_miu_ml", "fsh_miu_ml"]
SYMPTOM_MODEL_COLUMNS = [
    "age_yrs",
    "height_cm",
    "weight_kg",
    "bmi",
    "cycle_length_days",
    "cycle_r_i",
    "hair_growth_y_n",
    "pimples_y_n",
    "skin_darkening_y_n",
    "hair_loss_y_n",
    "weight_gain_y_n",
    "reg_exercise_y_n",
]
BOOLEAN_FIELD_NAMES = {
    "irregular_periods",
    "hair_growth_y_n",
    "pimples_y_n",
    "skin_darkening_y_n",
    "hair_loss_y_n",
    "weight_gain_y_n",
    "family_history_pcos",
}
FEATURE_LABELS = {
    "age_yrs": "Age",
    "height_cm": "Height",
    "weight_kg": "Weight",
    "bmi": "BMI",
    "cycle_length_days": "Cycle Length",
    "cycle_r_i": "Irregular Periods",
    "hair_growth_y_n": "Hair Growth",
    "pimples_y_n": "Acne",
    "skin_darkening_y_n": "Skin Darkening",
    "hair_loss_y_n": "Hair Loss",
    "weight_gain_y_n": "Weight Gain",
    "family_history_pcos": "Family History",
    "reg_exercise_y_n": "Physical Activity",
    "stress_level": "Stress Level",
    "amh_ng_ml": "AMH",
    "lh_miu_ml": "LH",
    "fsh_miu_ml": "FSH",
    "i_beta_hcg_miu_ml": "Beta-HCG",
    "ii_beta_hcg_miu_ml": "Beta-HCG Repeat",
    "lh_fsh_ratio": "LH/FSH Ratio",
    "follicle_no_l": "Follicle Count Left",
    "follicle_no_r": "Follicle Count Right",
    "follicle_total": "Total Follicles",
    "endometrium_mm": "Endometrium Thickness",
    "waist_inch": "Waist Circumference",
    "hip_inch": "Hip Circumference",
    "waist_hip_ratio": "Waist-Hip Ratio",
    "tsh_miu_l": "TSH",
    "rbs_mg_dl": "Random Blood Sugar",
    "fast_food_y_n": "Fast Food Intake",
}


class PredictionValidationError(Exception):
    """Raised when dual-mode prediction input fails validation."""

    def __init__(self, errors: dict[str, str]) -> None:
        super().__init__("Invalid prediction payload")
        self.errors = errors


def _required_resource_path(relative_path: str) -> Path:
    """Return a bundled resource path or raise a clear file-not-found error."""
    resource = resource_path_obj(relative_path)
    if not resource.exists():
        raise FileNotFoundError(f"Required resource '{relative_path}' was not found at '{resource}'.")
    return resource


def build_prediction_form_context(initial_values: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return page-ready step and field configuration for the dual-mode form."""
    values = initial_values or {}
    steps = []
    for step in STEP_DEFINITIONS:
        step_fields = []
        for field in FIELD_DEFINITIONS:
            if field["step"] != step["slug"]:
                continue
            hydrated = dict(field)
            hydrated["value"] = str(values.get(field["name"], ""))
            hydrated["clinical_only"] = MODE_CLINICAL in field["required_modes"] and MODE_BASIC not in field["required_modes"]
            step_fields.append(hydrated)
        enriched_step = dict(step)
        enriched_step["fields"] = step_fields
        steps.append(enriched_step)

    return {
        "mode_options": MODE_OPTIONS,
        "steps": steps,
        "selected_mode": str(values.get("prediction_mode", MODE_BASIC)),
    }


def run_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate user input, choose a prediction mode, and return structured results."""
    cleaned, submitted_values, display_inputs = _validate_payload(payload)
    feature_row = _build_feature_row(cleaned)

    requested_mode = cleaned["prediction_mode"]
    clinical_ready = _has_complete_clinical_markers(cleaned)
    if requested_mode == MODE_CLINICAL or clinical_ready:
        selected_mode = MODE_CLINICAL
        probability = _predict_clinical(feature_row)
        probability = _apply_contextual_adjustments(probability, cleaned, mode=MODE_CLINICAL)
    else:
        selected_mode = MODE_BASIC
        probability = _predict_symptom(feature_row)
        probability = _apply_contextual_adjustments(probability, cleaned, mode=MODE_BASIC)

    explanation_bundle = _build_ai_explanation(cleaned, feature_row, selected_mode)
    contribution_chart = explanation_bundle["contribution_chart"]
    contributors = explanation_bundle["contributors"] or ["No dominant indicator flagged"]
    result = {
        "mode": selected_mode,
        "mode_label": "Clinical Assessment" if selected_mode == MODE_CLINICAL else "Basic Screening",
        "risk": _risk_level(probability),
        "probability": round(probability, 4),
        "probability_pct": round(probability * 100, 1),
        "contributors": contributors,
        "inputs": display_inputs,
        "explanation": _result_explanation(probability, selected_mode),
        "bmi": round(feature_row["bmi"], 1),
        "contribution_chart": contribution_chart,
        "ai_explanation": explanation_bundle["ai_explanation"],
        "explanation_backend": explanation_bundle["backend"],
    }

    return {
        "mode": selected_mode,
        "mode_label": result["mode_label"],
        "feature_row": feature_row,
        "submitted_values": submitted_values,
        "result": result,
    }


def _validate_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    mode = str(payload.get("prediction_mode", MODE_BASIC)).strip().lower()
    if mode not in {MODE_BASIC, MODE_CLINICAL}:
        raise PredictionValidationError({"prediction_mode": "Select a valid prediction mode."})

    errors: dict[str, str] = {}
    cleaned: dict[str, Any] = {"prediction_mode": mode}
    submitted_values: dict[str, Any] = {"prediction_mode": mode}
    display_inputs: list[dict[str, str]] = [
        {"label": "Prediction Mode", "value": "Clinical Assessment" if mode == MODE_CLINICAL else "Basic Screening"}
    ]

    for field in FIELD_DEFINITIONS:
        raw_value = payload.get(field["name"], "")
        is_required = mode in field["required_modes"]

        if raw_value in ("", None):
            if is_required:
                errors[field["name"]] = "This field is required."
            continue

        try:
            parsed = _coerce_field_value(field, raw_value)
        except ValueError as exc:
            errors[field["name"]] = str(exc)
            continue

        cleaned[field["name"]] = parsed
        submitted_values[field["name"]] = raw_value
        display_inputs.append({"label": field["label"], "value": _display_value(field, parsed)})

    if any(name in cleaned for name in CLINICAL_FIELDS) and not _has_complete_clinical_markers(cleaned):
        for field_name in CORE_CLINICAL_FIELDS:
            if field_name not in cleaned:
                errors[field_name] = "AMH, LH, and FSH are required for clinical assessment."

    if errors:
        raise PredictionValidationError(errors)

    bmi = _calculate_bmi(float(cleaned["weight_kg"]), float(cleaned["height_cm"]))
    cleaned["bmi"] = bmi
    submitted_values["bmi"] = f"{bmi:.1f}"
    display_inputs.insert(4, {"label": "Calculated BMI", "value": f"{bmi:.1f}"})
    return cleaned, submitted_values, display_inputs


def _coerce_field_value(field: dict[str, Any], raw_value: Any) -> Any:
    if field["kind"] == "number":
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Enter a numeric value.") from exc
        if value < float(field["min"]) or value > float(field["max"]):
            raise ValueError(f"Enter a value between {field['min']} and {field['max']}.")
        return value

    normalized = str(raw_value).strip().lower()
    if field["kind"] == "boolean":
        if normalized not in {"0", "1", "yes", "no", "true", "false"}:
            raise ValueError("Select Yes or No.")
        return 1.0 if normalized in {"1", "yes", "true"} else 0.0

    if field["kind"] == "choice":
        valid_choices = {choice["value"] for choice in field["choices"] if choice["value"]}
        if normalized not in valid_choices:
            raise ValueError("Select a valid option.")
        return normalized

    raise ValueError("Unsupported field type.")


def _build_feature_row(cleaned: dict[str, Any]) -> dict[str, float]:
    defaults = get_default_feature_values().copy()
    defaults["age_yrs"] = float(cleaned["age_yrs"])
    defaults["height_cm"] = float(cleaned["height_cm"])
    defaults["weight_kg"] = float(cleaned["weight_kg"])
    defaults["bmi"] = float(cleaned["bmi"])
    defaults["cycle_length_days"] = float(cleaned["cycle_length_days"])
    defaults["cycle_r_i"] = 4.0 if float(cleaned["irregular_periods"]) >= 1 else 2.0
    defaults["hair_growth_y_n"] = float(cleaned["hair_growth_y_n"])
    defaults["pimples_y_n"] = float(cleaned["pimples_y_n"])
    defaults["skin_darkening_y_n"] = float(cleaned["skin_darkening_y_n"])
    defaults["hair_loss_y_n"] = float(cleaned["hair_loss_y_n"])
    defaults["weight_gain_y_n"] = float(cleaned["weight_gain_y_n"])
    defaults["reg_exercise_y_n"] = 1.0 if cleaned["physical_activity_level"] in {"moderate", "high"} else 0.0

    if _has_complete_clinical_markers(cleaned):
        beta_hcg = float(cleaned.get("beta_hcg_miu_ml", defaults.get("i_beta_hcg_miu_ml", 0.0)))
        defaults["amh_ng_ml"] = float(cleaned["amh_ng_ml"])
        defaults["lh_miu_ml"] = float(cleaned["lh_miu_ml"])
        defaults["fsh_miu_ml"] = float(cleaned["fsh_miu_ml"])
        defaults["i_beta_hcg_miu_ml"] = beta_hcg
        defaults["ii_beta_hcg_miu_ml"] = beta_hcg
        defaults["follicle_no_l"] = float(cleaned["follicle_no_l"])
        defaults["follicle_no_r"] = float(cleaned["follicle_no_r"])
        defaults["endometrium_mm"] = float(cleaned["endometrium_mm"])

    defaults["waist_hip_ratio"] = round(defaults["waist_inch"] / max(defaults["hip_inch"], 1.0), 4)
    defaults["beta_hcg_ratio"] = round(defaults["i_beta_hcg_miu_ml"] / (defaults["ii_beta_hcg_miu_ml"] + 1.0), 4)
    defaults["lh_fsh_ratio"] = round(defaults["lh_miu_ml"] / (defaults["fsh_miu_ml"] + 1.0), 4)
    defaults["fsh_lh"] = (
        round(defaults["fsh_miu_ml"] / defaults["lh_miu_ml"], 4)
        if defaults["lh_miu_ml"] > 0
        else float(get_default_feature_values().get("fsh_lh", 2.0))
    )
    defaults["follicle_total"] = float(defaults["follicle_no_l"] + defaults["follicle_no_r"])
    return {key: float(value) for key, value in defaults.items() if isinstance(value, (int, float, np.floating))}


def _predict_clinical(feature_row: dict[str, float]) -> float:
    clinical_model, scaler = _load_saved_artifacts()
    scaled = _scale_feature_row(feature_row, scaler)
    model_columns = _resolve_model_columns(clinical_model, scaler)
    model_input = scaled[model_columns]
    probability = (
        float(clinical_model.predict_proba(model_input)[0, 1])
        if hasattr(clinical_model, "predict_proba")
        else float(clinical_model.predict(model_input)[0])
    )
    return float(np.clip(probability, 0.0, 1.0))


def _predict_symptom(feature_row: dict[str, float]) -> float:
    symptom_model, scaler = _load_symptom_model()
    scaled = _scale_feature_row(feature_row, scaler)
    probability = float(symptom_model.predict_proba(scaled[SYMPTOM_MODEL_COLUMNS])[0, 1])
    return float(np.clip(probability, 0.0, 1.0))


def _apply_contextual_adjustments(probability: float, cleaned: dict[str, Any], mode: str) -> float:
    family_history_delta = 0.07 if float(cleaned.get("family_history_pcos", 0.0)) >= 1 else 0.0
    stress_delta = {"low": 0.0, "moderate": 0.02, "high": 0.05}.get(cleaned.get("stress_level", "low"), 0.0)
    activity_delta = {"low": 0.05, "moderate": 0.01, "high": -0.02}.get(cleaned.get("physical_activity_level", "moderate"), 0.0)
    mode_scale = 1.0 if mode == MODE_BASIC else 0.6
    adjusted = probability + ((family_history_delta + stress_delta + activity_delta) * mode_scale)
    return float(np.clip(adjusted, 0.0, 1.0))


def _build_contribution_chart(cleaned: dict[str, Any], feature_row: dict[str, float], mode: str) -> dict[str, list[Any]]:
    bmi_score = max(0.0, min(1.0, (feature_row["bmi"] - 22.0) / 10.0))
    scores: list[tuple[str, float]] = []

    if mode == MODE_BASIC:
        scores = [
            ("Irregular periods", 0.24 if float(cleaned["irregular_periods"]) >= 1 else 0.03),
            ("High BMI", 0.20 * bmi_score),
            ("Acne / pimples", 0.13 if float(cleaned["pimples_y_n"]) >= 1 else 0.02),
            ("Excess hair growth", 0.12 if float(cleaned["hair_growth_y_n"]) >= 1 else 0.01),
            ("Skin darkening", 0.12 if float(cleaned["skin_darkening_y_n"]) >= 1 else 0.01),
            ("Hair loss", 0.08 if float(cleaned["hair_loss_y_n"]) >= 1 else 0.01),
            ("Recent weight gain", 0.10 if float(cleaned["weight_gain_y_n"]) >= 1 else 0.02),
            ("Family history", 0.09 if float(cleaned["family_history_pcos"]) >= 1 else 0.01),
            ("Stress level", {"low": 0.02, "moderate": 0.05, "high": 0.09}[cleaned["stress_level"]]),
            ("Low activity", {"low": 0.10, "moderate": 0.05, "high": 0.01}[cleaned["physical_activity_level"]]),
        ]
    else:
        scores = [
            ("Irregular periods", 0.18 if float(cleaned["irregular_periods"]) >= 1 else 0.02),
            ("High BMI", 0.12 * bmi_score),
            ("Hormonal imbalance", 0.18 * max(0.1, min(1.0, feature_row["lh_fsh_ratio"]))),
            ("High AMH", 0.20 * max(0.0, min(1.0, feature_row["amh_ng_ml"] / 8.0))),
            ("Follicle count", 0.18 * max(0.0, min(1.0, feature_row["follicle_total"] / 24.0))),
            ("Endometrium finding", 0.09 * max(0.1, min(1.0, feature_row["endometrium_mm"] / 12.0))),
            ("Skin darkening", 0.07 if float(cleaned["skin_darkening_y_n"]) >= 1 else 0.01),
            ("Recent weight gain", 0.07 if float(cleaned["weight_gain_y_n"]) >= 1 else 0.01),
        ]

    ranked = sorted(scores, key=lambda item: item[1], reverse=True)
    return {
        "labels": [item[0] for item in ranked[:6]],
        "values": [round(item[1] * 100, 1) for item in ranked[:6]],
    }


def _build_ai_explanation(
    cleaned: dict[str, Any],
    feature_row: dict[str, float],
    mode: str,
) -> dict[str, Any]:
    fallback_chart = _build_contribution_chart(cleaned, feature_row, mode)
    fallback_bundle = fallback_explanation_from_chart(fallback_chart)

    try:
        if mode == MODE_BASIC:
            symptom_model, scaler = _load_symptom_model()
            scaled = _scale_feature_row(feature_row, scaler)
            shap_frame = scaled[SYMPTOM_MODEL_COLUMNS].copy()
            return build_local_explanation(
                model=symptom_model,
                X_row=shap_frame,
                feature_labels=FEATURE_LABELS,
            )

        clinical_model, scaler = _load_saved_artifacts()
        scaled = _scale_feature_row(feature_row, scaler)
        model_for_explanation = clinical_model
        if clinical_model.__class__.__name__ == "StackingClassifier":
            model_for_explanation = _load_explainer_model()

        shap_frame = scaled[_resolve_model_columns(model_for_explanation, scaler)].copy()
        return build_local_explanation(
            model=model_for_explanation,
            X_row=shap_frame,
            feature_labels=FEATURE_LABELS,
        )
    except Exception:
        return fallback_bundle


def _result_explanation(probability: float, mode: str) -> str:
    if probability < 0.33:
        return (
            "Current inputs suggest a low immediate PCOS risk. Persistent symptoms should still be reviewed clinically."
            if mode == MODE_BASIC
            else "Clinical markers currently suggest a lower PCOS risk profile, though physician review remains important."
        )
    if probability < 0.66:
        return (
            "Several indicators are mixed, so a moderate-risk result is appropriate. Clinical follow-up is recommended."
            if mode == MODE_BASIC
            else "Clinical markers show a moderate-risk pattern that deserves endocrine or gynecology review."
        )
    return (
        "Multiple screening indicators align with a high-risk PCOS profile. A clinical evaluation is strongly recommended."
        if mode == MODE_BASIC
        else "Hormonal and ovarian indicators align with a high-risk PCOS profile. Clinical follow-up is strongly recommended."
    )


def _display_value(field: dict[str, Any], value: Any) -> str:
    if field["kind"] == "number":
        return f"{float(value):.2f}".rstrip("0").rstrip(".")
    if field["kind"] == "boolean":
        return "Yes" if float(value) >= 1 else "No"
    return str(value).replace("_", " ").title()


def _calculate_bmi(weight_kg: float, height_cm: float) -> float:
    if height_cm <= 0:
        raise PredictionValidationError({"height_cm": "Height must be greater than zero."})
    return round(weight_kg / ((height_cm / 100.0) ** 2), 2)


def _has_complete_clinical_markers(cleaned: dict[str, Any]) -> bool:
    return all(name in cleaned for name in CORE_CLINICAL_FIELDS)


def _risk_level(probability: float) -> str:
    if probability < 0.33:
        return "Low"
    if probability < 0.66:
        return "Moderate"
    return "High"


@lru_cache(maxsize=1)
def _load_saved_artifacts() -> tuple[Any, Any]:
    clinical_model = joblib.load(_required_resource_path("models/pcos_model.pkl"))
    scaler = joblib.load(_required_resource_path("models/scaler.pkl"))
    return clinical_model, scaler


@lru_cache(maxsize=1)
def _load_explainer_model() -> Any:
    explainer_path = resource_path_obj("models/explainer_model.pkl")
    if not explainer_path.exists():
        clinical_model, _ = _load_saved_artifacts()
        return clinical_model
    return joblib.load(explainer_path)


@lru_cache(maxsize=1)
def _load_model_metadata() -> dict[str, Any]:
    metadata_path = resource_path_obj("models/model_metadata.pkl")
    if not metadata_path.exists():
        return {}
    metadata = joblib.load(metadata_path)
    return metadata if isinstance(metadata, dict) else {}


@lru_cache(maxsize=1)
def _load_symptom_model() -> tuple[RandomForestClassifier, Any]:
    _, scaler = _load_saved_artifacts()
    df = load_reference_frame()
    scaler_columns = list(getattr(scaler, "feature_names_in_", []))
    missing_columns = [column for column in scaler_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            "The packaged reference dataset is missing required model columns: "
            + ", ".join(sorted(missing_columns))
        )

    full_features = df[scaler_columns].copy()
    scaled = pd.DataFrame(scaler.transform(full_features), columns=scaler.feature_names_in_)
    target = df["pcos_y_n"].astype(int)

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=4,
        random_state=42,
    )
    model.fit(scaled[SYMPTOM_MODEL_COLUMNS], target)
    return model, scaler


def _scale_feature_row(feature_row: dict[str, float], scaler: Any) -> pd.DataFrame:
    scaler_columns = list(getattr(scaler, "feature_names_in_", []))
    frame = pd.DataFrame([{column: feature_row.get(column, 0.0) for column in scaler_columns}])
    return pd.DataFrame(scaler.transform(frame), columns=scaler_columns)


def _resolve_model_columns(model: Any, scaler: Any) -> list[str]:
    model_columns = list(getattr(model, "feature_names_in_", []))
    if model_columns:
        return model_columns

    metadata = _load_model_metadata()
    selected_features = list(metadata.get("selected_features", []))
    if selected_features:
        return selected_features

    return list(getattr(scaler, "feature_names_in_", []))
