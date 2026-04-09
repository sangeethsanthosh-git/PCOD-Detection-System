"""URL routes for analytics pages."""

from __future__ import annotations

from django.urls import path

from . import views


app_name = "analysis"

urlpatterns = [path("", views.charts_view, name="charts")]
