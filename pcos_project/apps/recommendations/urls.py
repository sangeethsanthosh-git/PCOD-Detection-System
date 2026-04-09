"""URL routes for recommendation pages."""

from __future__ import annotations

from django.urls import path

from . import views


app_name = "recommendations"

urlpatterns = [path("", views.doctors_view, name="doctors")]
