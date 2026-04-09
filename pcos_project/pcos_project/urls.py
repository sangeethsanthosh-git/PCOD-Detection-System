"""URL routing for the PCOS AI Clinical Support Platform."""

from __future__ import annotations

from django.contrib import admin
from django.urls import path

from . import views


urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.home, name="home"),
    path("predict/", views.predict_page, name="predict"),
    path("result/", views.result_page, name="result"),
    path("analysis/", views.analysis_page, name="analysis"),
    path("doctors/", views.doctors_page, name="doctors"),
    path("education/", views.education_page, name="education"),
    path("search/", views.search_page, name="search"),
    path("api/predict/", views.predict_api, name="api_predict"),
    path("api/analytics/", views.analytics_api, name="api_analytics"),
    path("api/analysis/", views.analysis_data, name="api_analysis"),
    path("api/analysis-data/", views.analysis_data, name="api_analysis_data"),
    path("api/feature-importance/", views.feature_importance_api, name="api_feature_importance"),
    path("api/doctors/", views.doctors_api, name="api_doctors"),
    path("api/location-autocomplete/", views.location_autocomplete_api, name="api_location_autocomplete"),
    path("api/videos/", views.videos_api, name="api_videos"),
    path("api/suggestions/", views.suggestions_api, name="api_suggestions"),
]
