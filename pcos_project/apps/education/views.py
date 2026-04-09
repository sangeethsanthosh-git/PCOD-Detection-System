"""Views for educational resources."""

from __future__ import annotations

from django.shortcuts import render


def resources_view(request):
    """Render educational videos and article links."""
    videos = [
        {"title": "What is PCOS?", "embed_url": "https://www.youtube.com/embed/dRjQnYvJXwA"},
        {"title": "PCOS Diet Basics", "embed_url": "https://www.youtube.com/embed/LfQW0z4mYBs"},
        {"title": "PCOS Exercise Guidance", "embed_url": "https://www.youtube.com/embed/o2lK5M9C4hA"},
        {"title": "Hormonal Balance Explained", "embed_url": "https://www.youtube.com/embed/1H2P6z-88G8"},
    ]

    articles = [
        {
            "title": "Diagnostic Criteria in Clinical Practice",
            "summary": "Review of current criteria and metabolic screening pathways.",
        },
        {
            "title": "Nutrition Planning for Insulin Resistance",
            "summary": "Evidence-based meal timing and macro composition considerations.",
        },
        {
            "title": "Mental Health and PCOS",
            "summary": "Recognizing emotional burden and integrating supportive care.",
        },
    ]

    context = {"active_nav": "education", "videos": videos, "articles": articles}
    return render(request, "education/resources.html", context)
