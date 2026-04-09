"""Views for doctor recommendation pages."""

from __future__ import annotations

from django.shortcuts import render


def doctors_view(request):
    """Render doctor/hospital recommendation cards."""
    location = request.GET.get("location", "").strip()
    cards = [
        {
            "hospital": "City Women's Specialty Center",
            "specialization": "Gynecology & Reproductive Endocrinology",
            "location": "Downtown Medical District",
        },
        {
            "hospital": "Metro Hormonal Health Clinic",
            "specialization": "PCOS & Metabolic Care",
            "location": "North Avenue",
        },
        {
            "hospital": "Harmony Multispecialty Hospital",
            "specialization": "Obstetrics, Gynecology, Lifestyle Medicine",
            "location": "Westcare Campus",
        },
    ]

    context = {
        "active_nav": "doctors",
        "location": location,
        "cards": cards,
    }
    return render(request, "recommendations/doctors.html", context)
