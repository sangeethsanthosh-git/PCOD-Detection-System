"""URL routes for predictor pages."""

from __future__ import annotations

from django.urls import path

from . import views


app_name = "predictor"

urlpatterns = [
    path("", views.predict_view, name="predict"),
    path("result/", views.result_view, name="result"),
]
