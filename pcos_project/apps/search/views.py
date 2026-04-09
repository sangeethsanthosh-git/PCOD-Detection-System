"""Views for the smart search assistant."""

from __future__ import annotations

from django.shortcuts import render


FAQ_ITEMS = [
    {"q": "Can PCOS be cured permanently?", "a": "PCOS is managed, not cured; treatment focuses on symptoms and long-term risk reduction."},
    {"q": "Does weight loss help PCOS?", "a": "For many patients, modest weight reduction can improve cycle regularity and metabolic markers."},
    {"q": "Is irregular period always PCOS?", "a": "No. Irregular cycles have multiple causes and require medical evaluation."},
    {"q": "Can PCOS affect fertility?", "a": "It can affect ovulation, but many individuals conceive with clinical support and treatment."},
]


def assistant_view(request):
    """Render assistant page with FAQ and simple keyword response."""
    query = request.GET.get("q", "").strip()
    answer = ""
    if query:
        lowered = query.lower()
        if "diet" in lowered:
            answer = "Focus on balanced meals, high fiber, lean protein, and controlled refined sugar intake."
        elif "exercise" in lowered:
            answer = "Aim for regular aerobic activity plus strength training for insulin sensitivity support."
        elif "fertility" in lowered:
            answer = "Early consultation with a gynecologist/endocrinologist can help optimize ovulation planning."
        else:
            answer = "This assistant provides general guidance. Please consult a clinician for personalized diagnosis."

    context = {"active_nav": "search", "faqs": FAQ_ITEMS, "query": query, "answer": answer}
    return render(request, "search/assistant.html", context)
