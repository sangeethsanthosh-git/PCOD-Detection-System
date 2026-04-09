"""Views for model analysis dashboards."""

from __future__ import annotations

import json

from django.shortcuts import render
from pcos_project.resource_utils import resource_path_obj


def charts_view(request):
    """Render analytics charts page."""
    feature_labels = [
        "AMH",
        "Cycle Irregularity",
        "BMI",
        "LH/FSH Ratio",
        "Follicle Count",
        "Skin Darkening",
    ]
    feature_values = [0.18, 0.16, 0.15, 0.12, 0.10, 0.09]

    feature_file = resource_path_obj("results/feature_scores.csv")
    if feature_file.exists():
        try:
            import pandas as pd

            df = pd.read_csv(feature_file).head(8)
            if "feature" in df.columns:
                feature_labels = df["feature"].astype(str).tolist()
            if "importance" in df.columns:
                feature_values = df["importance"].astype(float).tolist()
            elif "combined_score" in df.columns:
                feature_values = df["combined_score"].astype(float).tolist()
        except Exception:
            pass

    context = {
        "active_nav": "analysis",
        "feature_labels_json": json.dumps(feature_labels),
        "feature_values_json": json.dumps(feature_values),
        "risk_distribution_json": json.dumps([58, 27, 15]),
        "age_vs_pcos_json": json.dumps(
            {"x": [18, 22, 25, 28, 32, 35, 38, 42], "y": [0.12, 0.18, 0.22, 0.35, 0.47, 0.52, 0.56, 0.63]}
        ),
        "bmi_vs_pcos_json": json.dumps(
            {"x": [18, 21, 24, 27, 30, 33, 36], "y": [0.09, 0.14, 0.20, 0.33, 0.45, 0.57, 0.66]}
        ),
    }
    return render(request, "analysis/charts.html", context)
