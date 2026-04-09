"""Project views for pages and JSON APIs."""

from __future__ import annotations

import json
import os
import re
import traceback
from datetime import datetime, timezone
from typing import Any

from django.conf import settings
from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from pcos_project.resource_utils import log_path_obj
from services.analytics_service import build_analysis_payload, build_dashboard_summary, build_empty_analysis_payload
from services.doctor_locator import autocomplete_locations, find_doctors
from services.google_search import get_suggestions
from services.predictor_service import (
    MODE_BASIC,
    PredictionValidationError,
    build_prediction_form_context,
    run_prediction,
)
from services.youtube_service import search_videos


SAFE_QUERY_RE = re.compile(r"^[A-Za-z0-9\s,.'()/+\-:&?]{2,120}$")
QUERY_MAX_LENGTH = 120

ARTICLES = [
    {
        "title": "Updated Clinical Perspective on PCOS Diagnosis",
        "source": "Endocrine Practice",
        "summary": "AI screening works best when paired with metabolic assessment and clinician review.",
    },
    {
        "title": "Lifestyle Intervention and Ovulatory Health",
        "source": "Women's Health Review",
        "summary": "Structured movement and nutrition plans remain first-line support for many patients.",
    },
    {
        "title": "Insulin Resistance in PCOS: Practical Monitoring",
        "source": "Clinical Metabolism Journal",
        "summary": "Continuous monitoring of BMI, glucose, and symptoms helps reduce long-term risk.",
    },
]

FAQ_ITEMS = [
    {
        "question": "Can PCOS be cured permanently?",
        "answer": "PCOS is usually managed over time. Treatment focuses on symptom control, fertility support, and metabolic health.",
    },
    {
        "question": "Does weight loss help PCOS?",
        "answer": "For many patients, even modest weight reduction can help improve cycle regularity and insulin sensitivity.",
    },
    {
        "question": "Can PCOS affect fertility?",
        "answer": "Yes. PCOS can interfere with ovulation, but many patients conceive with the right medical support.",
    },
]


def home(request):
    """Render the landing dashboard."""
    return render(
        request,
        "home.html",
        {
            "active_nav": "home",
            "stats": build_dashboard_summary(),
            "articles": ARTICLES,
        },
    )


def predict_page(request):
    """Render the dual-mode prediction workflow."""
    last_prediction = _session_prediction(request)
    form_context = build_prediction_form_context(last_prediction.get("submitted_values"))
    return render(
        request,
        "predict.html",
        {
            "active_nav": "predict",
            "prediction_modes": form_context["mode_options"],
            "prediction_steps": form_context["steps"],
            "selected_mode": form_context["selected_mode"],
            "last_prediction": last_prediction,
        },
    )


def result_page(request):
    """Render the latest prediction from the session."""
    last_prediction = _session_prediction(request)
    result = last_prediction.get("result", {})
    probability = float(result.get("probability_pct", 0) or 0)
    return render(
        request,
        "result.html",
        {
            "active_nav": "predict",
            "last_prediction": last_prediction,
            "probability": round(probability, 2),
            "gauge_risk_label": _risk_gauge_label(probability),
        },
    )


def analysis_page(request):
    """Render the analytics dashboard page."""
    return render(
        request,
        "analysis.html",
        {
            "active_nav": "analysis",
            "last_prediction": _session_prediction(request),
        },
    )


def doctors_page(request):
    """Render the provider search page."""
    return render(
        request,
        "doctors.html",
        {
            "active_nav": "doctors",
            "sample_locations": ["Mumbai", "Delhi", "Bangalore", "Chennai"],
            "google_maps_browser_key": os.getenv("GOOGLE_MAPS_BROWSER_KEY", "").strip(),
        },
    )


def education_page(request):
    """Render the patient education page."""
    return render(
        request,
        "education.html",
        {
            "active_nav": "education",
            "video_topics": ["PCOS treatment", "PCOS diet", "PCOS exercise", "PCOS hormonal imbalance"],
        },
    )


def search_page(request):
    """Render the smart search page."""
    return render(
        request,
        "search.html",
        {
            "active_nav": "search",
            "faqs": FAQ_ITEMS,
            "initial_query": (request.GET.get("q") or "").strip(),
        },
    )


@require_POST
@csrf_protect
def predict_api(request):
    """Return a prediction result for dual-mode user input."""
    try:
        payload = _read_json_body(request)
        prediction = _json_safe(run_prediction(payload))
        analytics = _json_safe(
            build_analysis_payload(
                prediction_inputs=prediction["feature_row"],
                probability=prediction["result"]["probability"],
                mode=prediction["mode"],
                contribution_chart=prediction["result"]["contribution_chart"],
                ai_explanation=prediction["result"].get("ai_explanation"),
            )
        )
    except PredictionValidationError as exc:
        return JsonResponse(
            {
                "ok": False,
                "message": "Please review the highlighted values before prediction can run.",
                "errors": exc.errors,
            },
            status=400,
        )
    except Exception as exc:
        log_file = _write_prediction_error(exc)
        message = str(exc) if settings.DEBUG else "Prediction service is temporarily unavailable. Please try again."
        return JsonResponse(
            {
                "ok": False,
                "message": message,
                "error": str(exc),
                "log_file": str(log_file),
            },
            status=500,
        )

    try:
        request.session["last_prediction"] = prediction
        request.session["prediction_result"] = prediction["result"]
        request.session["risk_probability"] = float(prediction["result"]["probability"])
        request.session["ai_explanation"] = prediction["result"].get("ai_explanation", {})
        request.session["symptom_contribution"] = {
            label: value
            for label, value in zip(
                prediction["result"].get("contribution_chart", {}).get("labels", []),
                prediction["result"].get("contribution_chart", {}).get("values", []),
            )
        }
        request.session["analysis_payload"] = analytics
        request.session.modified = True

        return JsonResponse(
            {
                "ok": True,
                "mode": prediction["mode"],
                "risk_probability": float(prediction["result"]["probability"]),
                "risk_level": prediction["result"]["risk"],
                "result": prediction["result"],
                "analysis": analytics,
            }
        )
    except Exception as exc:
        log_file = _write_prediction_error(exc)
        message = str(exc) if settings.DEBUG else "Prediction service is temporarily unavailable. Please try again."
        return JsonResponse(
            {
                "ok": False,
                "message": message,
                "error": str(exc),
                "log_file": str(log_file),
            },
            status=500,
        )


@require_http_methods(["GET", "POST"])
def analytics_api(request):
    """Return chart data for the analysis page or the latest prediction."""
    if request.method == "POST":
        try:
            prediction = run_prediction(_read_json_body(request))
        except PredictionValidationError as exc:
            return JsonResponse({"ok": False, "errors": exc.errors}, status=400)
    else:
        prediction = _session_prediction(request)

    if prediction:
        analysis = build_analysis_payload(
            prediction_inputs=prediction.get("feature_row"),
            probability=prediction.get("result", {}).get("probability"),
            mode=prediction.get("mode", MODE_BASIC),
            contribution_chart=prediction.get("result", {}).get("contribution_chart"),
            ai_explanation=prediction.get("result", {}).get("ai_explanation"),
        )
    else:
        analysis = build_empty_analysis_payload()

    return JsonResponse({"ok": True, **analysis, "analysis": analysis})


@require_GET
def analysis_data(request):
    """Return a flat, chart-safe analysis payload for frontend consumers."""
    prediction = _session_prediction(request)
    stored_analysis = request.session.get("analysis_payload")
    if prediction and not stored_analysis:
        stored_analysis = build_analysis_payload(
            prediction_inputs=prediction.get("feature_row"),
            probability=prediction.get("result", {}).get("probability"),
            mode=prediction.get("mode", MODE_BASIC),
            contribution_chart=prediction.get("result", {}).get("contribution_chart"),
            ai_explanation=prediction.get("result", {}).get("ai_explanation"),
        )

    analysis = stored_analysis or build_empty_analysis_payload()
    risk_probability = request.session.get("risk_probability")
    if risk_probability is None:
        risk_probability = float(analysis.get("risk_probability", {}).get("probability", 0) or 0)

    symptoms = request.session.get("symptom_contribution", {})
    if not symptoms:
        contribution = analysis.get("symptom_contribution", {})
        symptoms = {
            label: value
            for label, value in zip(contribution.get("labels", []), contribution.get("values", []))
        }

    has_prediction = bool(prediction)
    default_message = "Run a prediction to view personalized PCOS analysis charts."
    ai_explanation = request.session.get("ai_explanation")
    if not isinstance(ai_explanation, dict):
        ai_explanation = analysis.get("ai_explanation", {})

    return JsonResponse(
        {
            "has_prediction": has_prediction,
            "message": analysis.get("message", "") if has_prediction else default_message,
            "risk_probability": float(risk_probability or 0),
            "symptoms": symptoms if isinstance(symptoms, dict) else {},
            "symptom_contribution": symptoms if isinstance(symptoms, dict) else {},
            "feature_importance": analysis.get("feature_importance", {"labels": [], "values": []}),
            "ai_explanation": ai_explanation if isinstance(ai_explanation, dict) else {},
            "bmi_vs_risk": analysis.get("bmi_vs_risk", {"points": [], "patient": {"x": 0, "y": 0}}),
            "age_vs_risk": analysis.get("age_vs_risk", {"points": [], "patient": {"x": 0, "y": 0}}),
        }
    )


@require_GET
def feature_importance_api(request):
    """Return session-backed AI explanation details for charts or panels."""
    analysis = request.session.get("analysis_payload") or build_empty_analysis_payload()
    ai_explanation = request.session.get("ai_explanation", {})
    feature_importance = analysis.get("feature_importance", {"labels": [], "values": []})
    return JsonResponse(
        {
            "ok": True,
            "has_prediction": "last_prediction" in request.session,
            "ai_explanation": ai_explanation if isinstance(ai_explanation, dict) else {},
            "feature_importance": feature_importance,
        }
    )


@require_GET
def doctors_api(request):
    """Search for nearby gynecology providers."""
    rate_limited = _rate_limit_response(request, "doctor-search", limit=30, window_seconds=60 * 60)
    if rate_limited:
        return rate_limited

    location = (request.GET.get("location") or "").strip()
    latitude = _parse_optional_float(request.GET.get("lat"))
    longitude = _parse_optional_float(request.GET.get("lon"))

    if location and not _is_safe_query(location, max_length=80):
        return JsonResponse({"ok": False, "message": "Location contains unsupported characters."}, status=400)
    if not location and (latitude is None or longitude is None):
        return JsonResponse({"ok": False, "message": "Enter a location or allow automatic location access."}, status=400)

    try:
        results = find_doctors(location=location or None, latitude=latitude, longitude=longitude)
    except Exception:
        return JsonResponse(
            {
                "ok": False,
                "message": "We could not reach the doctor location service right now. Please try again shortly.",
            },
            status=502,
        )

    message = "" if results else "No gynecology hospitals or clinics were found nearby."
    return JsonResponse({"ok": True, "results": results, "message": message})


@require_GET
def location_autocomplete_api(request):
    """Return location autocomplete suggestions."""
    rate_limited = _rate_limit_response(request, "location-autocomplete", limit=60, window_seconds=60 * 60)
    if rate_limited:
        return rate_limited

    query = (request.GET.get("q") or "").strip()
    if len(query) < 2:
        return JsonResponse({"ok": True, "results": []})
    if not _is_safe_query(query, max_length=80):
        return JsonResponse({"ok": False, "message": "Location text contains unsupported characters."}, status=400)

    try:
        suggestions = autocomplete_locations(query)
    except Exception:
        return JsonResponse({"ok": False, "message": "Location autocomplete is unavailable right now."}, status=502)

    return JsonResponse({"ok": True, "results": suggestions})


@require_GET
def videos_api(request):
    """Return educational video search results."""
    rate_limited = _rate_limit_response(request, "videos", limit=40, window_seconds=60 * 60)
    if rate_limited:
        return rate_limited

    query = (request.GET.get("q") or "PCOS treatment").strip()
    if not _is_safe_query(query):
        return JsonResponse({"ok": False, "message": "Search text contains unsupported characters."}, status=400)

    try:
        payload = search_videos(query)
    except Exception:
        return JsonResponse({"ok": False, "message": "Video search is unavailable at the moment."}, status=502)

    items = payload.get("items", [])
    message = payload.get("message", "") or ("No PCOS-related videos were returned for that topic." if not items else "")
    return JsonResponse({"ok": True, "results": items, "message": message})


@require_GET
def suggestions_api(request):
    """Return Google query suggestions for PCOS questions."""
    rate_limited = _rate_limit_response(request, "suggestions", limit=100, window_seconds=60 * 60)
    if rate_limited:
        return rate_limited

    query = (request.GET.get("q") or "").strip()
    if len(query) < 2:
        return JsonResponse({"ok": True, "suggestions": [], "faqs": FAQ_ITEMS[:2]})
    if not _is_safe_query(query):
        return JsonResponse({"ok": False, "message": "Search text contains unsupported characters."}, status=400)

    try:
        suggestions = get_suggestions(query)
    except Exception:
        suggestions = _fallback_suggestions(query)

    matching_faqs = [item for item in FAQ_ITEMS if query.lower() in item["question"].lower()] or FAQ_ITEMS[:2]
    message = "" if suggestions else "No PCOS-related suggestion phrases were found for that query."
    return JsonResponse({"ok": True, "suggestions": suggestions, "faqs": matching_faqs, "message": message})


def _read_json_body(request) -> dict[str, Any]:
    if not request.body:
        return {}
    return json.loads(request.body.decode("utf-8"))


def _session_prediction(request) -> dict[str, Any]:
    last_prediction = request.session.get("last_prediction", {})
    if isinstance(last_prediction, dict) and last_prediction:
        return last_prediction

    legacy_prediction = request.session.get("prediction_result", {})
    if isinstance(legacy_prediction, dict) and "result" not in legacy_prediction and "risk" in legacy_prediction:
        return {"result": legacy_prediction}
    return legacy_prediction if isinstance(legacy_prediction, dict) else {}


def _is_safe_query(query: str, max_length: int = QUERY_MAX_LENGTH) -> bool:
    return len(query) <= max_length and bool(SAFE_QUERY_RE.fullmatch(query))


def _risk_gauge_label(probability_percent: float) -> str:
    """Return the display label used by the result-page confidence gauge."""
    if probability_percent > 60:
        return "High Risk"
    if probability_percent > 30:
        return "Moderate Risk"
    return "Low Risk"


def _fallback_suggestions(query: str) -> list[str]:
    base = [
        f"PCOS {query} treatment",
        f"PCOS {query} diet",
        f"PCOS {query} causes",
        f"PCOS {query} vitamins",
    ]
    return [item for item in base if "pcos" in item.lower()][:6]


def _parse_optional_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rate_limit_response(request, scope: str, *, limit: int, window_seconds: int) -> JsonResponse | None:
    client_id = (
        request.META.get("HTTP_X_FORWARDED_FOR", "").split(",")[0].strip()
        or request.META.get("REMOTE_ADDR", "anonymous")
    )
    cache_key = f"ratelimit:{scope}:{client_id}"
    current = cache.get(cache_key, 0)
    if current >= limit:
        return JsonResponse(
            {
                "ok": False,
                "message": "Too many requests. Please wait a moment before trying again.",
            },
            status=429,
        )
    cache.set(cache_key, current + 1, window_seconds)
    return None


def _json_safe(value: Any) -> Any:
    """Convert runtime objects into JSON/session-safe Python primitives."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except Exception:
            pass

    tolist_method = getattr(value, "tolist", None)
    if callable(tolist_method):
        try:
            return _json_safe(tolist_method())
        except Exception:
            pass

    return str(value)


def _write_prediction_error(exc: Exception):
    """Persist packaged prediction failures to a local log file."""
    log_file = log_path_obj("prediction_errors.log")
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {exc.__class__.__name__}: {exc}\n")
            handle.write(traceback.format_exc())
            handle.write("\n")
    except OSError:
        pass
    return log_file
