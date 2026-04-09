"""URL routes for educational pages."""

from __future__ import annotations

from django.urls import path

from . import views


app_name = "education"

urlpatterns = [path("", views.resources_view, name="resources")]
