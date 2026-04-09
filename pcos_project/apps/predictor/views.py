"""Views for prediction pages."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from django.contrib import messages
from django.shortcuts import redirect, render
from pcos_project.resource_utils import resource_path_obj

from .forms import PredictionForm

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None  # type: ignore[assignment]

MODEL_PATH = resource_path_obj("models/pcos_model.pkl")
SCALER_PATH = resource_path_obj("models/scaler.pkl")


@lru_cache(maxsize=1)
def _load_artifacts():
    """Lazy-load model artifacts from disk."""
    if joblib is None:
        return None, None
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        model = None
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    return model, scaler


def predict_view(request):
    """Render prediction form and handle risk inference requests."""
    form = PredictionForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        cleaned = form.cleaned_data
        probability, contributors = _run_prediction(cleaned)
        risk_level = _risk_level(probability)
        request.session["prediction_result"] = {
            "probability": probability,
            "risk_level": risk_level,
            "contributors": contributors,
            "submitted_data": _serialize_form_payload(cleaned),
        }
        return redirect("predictor:result")

    context = {"active_nav": "predict", "form": form}
    return render(request, "predictor/predict.html", context)


def result_view(request):
    """Render risk result page from session context."""
    result = request.session.get("prediction_result")
    if not result:
        messages.info(request, "Submit patient details first to generate a risk score.")
        return redirect("predictor:predict")

    context = {
        "active_nav": "predict",
        "risk_level": result["risk_level"],
        "probability": result["probability"],
        "contributors": result["contributors"],
        "submitted_data": result["submitted_data"],
    }
    return render(request, "predictor/result.html", context)


def _run_prediction(form_data: Dict[str, object]) -> Tuple[float, List[str]]:
    """Predict probability from user inputs with robust fallback."""
    model, scaler = _load_artifacts()
    features = _prepare_feature_frame(form_data, model)

    if model is None:
        probability = _heuristic_probability(form_data)
        return probability, _top_contributors(form_data)

    try:
        model_input = features.copy()
        if scaler is not None and hasattr(scaler, "n_features_in_"):
            if int(scaler.n_features_in_) == model_input.shape[1]:
                model_input = pd.DataFrame(
                    scaler.transform(model_input),
                    columns=model_input.columns,
                )

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(model_input)[0, 1])
        else:
            probability = float(model.predict(model_input)[0])
    except Exception:
        probability = _heuristic_probability(form_data)

    contributors = _top_contributors(form_data)
    return float(np.clip(probability, 0.0, 1.0)), contributors


def _prepare_feature_frame(form_data: Dict[str, object], model) -> pd.DataFrame:
    """Map form inputs to model-compatible feature frame."""
    feature_map = {
        "age": "age_yrs",
        "bmi": "bmi",
        "amh_level": "amh_ng_ml",
        "lh_level": "lh_miu_ml",
        "fsh_level": "fsh_miu_ml",
        "follicle_left": "follicle_no_l",
        "follicle_right": "follicle_no_r",
        "weight_gain": "weight_gain_y_n",
        "skin_darkening": "skin_darkening_y_n",
        "hair_growth": "hair_growth_y_n",
        "pimples": "pimples_y_n",
        "cycle_regularity": "cycle_r_i",
    }

    row = {}
    for key, value in form_data.items():
        mapped_key = feature_map.get(key, key)
        if key in {"weight_gain", "skin_darkening", "hair_growth", "pimples"}:
            row[mapped_key] = int(value)
        elif key == "cycle_regularity":
            row[mapped_key] = 1 if str(value).lower() == "irregular" else 0
        else:
            row[mapped_key] = float(value)

    row["follicle_total"] = row.get("follicle_no_l", 0.0) + row.get("follicle_no_r", 0.0)
    row["lh_fsh_ratio"] = row.get("lh_miu_ml", 0.0) / (row.get("fsh_miu_ml", 0.0) + 1.0)

    frame = pd.DataFrame([row])
    if model is not None and hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        for col in expected:
            if col not in frame.columns:
                frame[col] = 0.0
        frame = frame[expected]
    return frame


def _heuristic_probability(form_data: Dict[str, object]) -> float:
    """Simple medical-risk heuristic fallback if model inference is unavailable."""
    score = 0.12
    score += 0.18 if float(form_data["bmi"]) >= 28 else 0.0
    score += 0.20 if float(form_data["amh_level"]) >= 4.5 else 0.0
    score += 0.15 if str(form_data["cycle_regularity"]).lower() == "irregular" else 0.0
    score += 0.12 if int(form_data["weight_gain"]) == 1 else 0.0
    score += 0.10 if int(form_data["skin_darkening"]) == 1 else 0.0
    score += 0.08 if int(form_data["hair_growth"]) == 1 else 0.0
    score += 0.05 if int(form_data["pimples"]) == 1 else 0.0
    return float(np.clip(score, 0.05, 0.95))


def _top_contributors(form_data: Dict[str, object]) -> List[str]:
    """Return visible feature contributions to explain prediction."""
    contributors = []
    if float(form_data["amh_level"]) >= 4.5:
        contributors.append("High AMH level")
    if str(form_data["cycle_regularity"]).lower() == "irregular":
        contributors.append("Irregular menstrual cycle")
    if float(form_data["bmi"]) >= 27:
        contributors.append("Elevated BMI")
    if int(form_data["weight_gain"]) == 1:
        contributors.append("Recent weight gain")
    if int(form_data["hair_growth"]) == 1:
        contributors.append("Clinical hair growth symptom")
    return contributors[:4] if contributors else ["No dominant risk contributor flagged"]


def _risk_level(probability: float) -> str:
    """Map probability to low/medium/high risk tiers."""
    if probability < 0.33:
        return "Low"
    if probability < 0.66:
        return "Medium"
    return "High"


def _serialize_form_payload(form_data: Dict[str, object]) -> Dict[str, str]:
    """Prepare form data for result-page summary table."""
    return {str(k): str(v) for k, v in form_data.items()}
