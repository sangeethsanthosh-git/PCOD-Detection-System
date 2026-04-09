"""URL routes for search assistant pages."""

from __future__ import annotations

from django.urls import path

from . import views


app_name = "search"

urlpatterns = [path("", views.assistant_view, name="assistant")]
