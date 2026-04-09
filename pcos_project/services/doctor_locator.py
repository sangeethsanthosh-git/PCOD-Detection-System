"""Doctor and hospital location search helpers."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

from django.core.cache import cache


GOOGLE_TEXTSEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
GOOGLE_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
GOOGLE_AUTOCOMPLETE_URL = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "PCOS-AI-Clinical-Support-Platform/1.0"
CACHE_SECONDS = 60 * 60
RESULT_LIMIT = 8
SPECIALTY_TERMS = (
    "gyne",
    "gynae",
    "gynae",
    "obstetric",
    "obstet",
    "maternity",
    "women",
    "fertility",
    "ivf",
    "reproductive",
)
EXCLUDE_TERMS = (
    "ent",
    "eye",
    "dental",
    "dentist",
    "ortho",
    "orthopedic",
    "cardio",
    "skin",
    "dermat",
    "neuro",
    "ear",
)


def find_doctors(
    *,
    location: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    limit: int = RESULT_LIMIT,
) -> list[dict[str, Any]]:
    """Return nearby gynecology-related providers for text or coordinate input."""
    lat, lon, display_location = _resolve_search_origin(location=location, latitude=latitude, longitude=longitude)
    cache_key = f"doctors:{_cache_fragment(display_location)}:{round(lat, 3)}:{round(lon, 3)}:{limit}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    providers: list[dict[str, Any]] = []
    google_api_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()

    if google_api_key:
        try:
            providers = _google_places_search(lat, lon, google_api_key, limit)
        except Exception:
            providers = []

    if not providers:
        providers = _osm_provider_search(lat, lon, display_location, limit)

    cache.set(cache_key, providers, CACHE_SECONDS)
    return providers


def autocomplete_locations(query: str, limit: int = 6) -> list[dict[str, str]]:
    """Return manual location suggestions for doctor search."""
    cache_key = f"location-autocomplete:{_cache_fragment(query)}:{limit}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    google_api_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if google_api_key:
        try:
            suggestions = _google_autocomplete(query, google_api_key, limit)
            if suggestions:
                cache.set(cache_key, suggestions, CACHE_SECONDS)
                return suggestions
        except Exception:
            pass

    suggestions = _nominatim_autocomplete(query, limit)
    cache.set(cache_key, suggestions, CACHE_SECONDS)
    return suggestions


def _resolve_search_origin(
    *,
    location: str | None,
    latitude: float | None,
    longitude: float | None,
) -> tuple[float, float, str]:
    if latitude is not None and longitude is not None:
        display = _reverse_geocode(latitude, longitude)
        return latitude, longitude, display
    if location:
        lat, lon = _geocode_with_nominatim(location)
        return lat, lon, location
    raise ValueError("A location or coordinates are required.")


def _google_places_search(lat: float, lon: float, api_key: str, limit: int) -> list[dict[str, Any]]:
    providers: list[dict[str, Any]] = []
    queries = [
        {"keyword": "gynecologist", "type": "hospital"},
        {"keyword": "gynecology clinic", "type": "hospital"},
        {"keyword": "women clinic", "type": "hospital"},
    ]
    for query in queries:
        params = {
            "location": f"{lat},{lon}",
            "radius": 15000,
            "key": api_key,
            **query,
        }
        payload = _fetch_json(f"{GOOGLE_NEARBY_URL}?{urlencode(params)}")
        for item in payload.get("results", []):
            structured = _format_google_provider(item, lat, lon)
            if structured:
                providers.append(structured)
            if len(providers) >= limit * 2:
                break
        if len(providers) >= limit * 2:
            break

    deduped = _dedupe_and_rank(providers)
    return deduped[:limit]


def _format_google_provider(item: dict[str, Any], origin_lat: float, origin_lon: float) -> dict[str, Any] | None:
    name = item.get("name") or ""
    address = item.get("vicinity") or item.get("formatted_address") or "Address unavailable"
    types = [value.lower() for value in item.get("types", [])]
    text_blob = " ".join([name.lower(), address.lower(), " ".join(types)])
    if not _is_relevant_specialist(text_blob):
        return None

    location = item.get("geometry", {}).get("location", {})
    lat = location.get("lat")
    lon = location.get("lng")
    if lat is None or lon is None:
        return None

    place_id = item.get("place_id", "")
    maps_link = (
        f"https://www.google.com/maps/search/?api=1&query_place_id={place_id}"
        if place_id
        else f"https://www.google.com/maps/search/?api=1&query={quote_plus(name)}"
    )
    distance_km = _haversine_km(origin_lat, origin_lon, float(lat), float(lon))
    speciality = _speciality_label(text_blob)
    return {
        "name": name,
        "hospital_name": name,
        "speciality": speciality,
        "address": address,
        "distance_km": round(distance_km, 1),
        "rating": item.get("rating"),
        "maps_link": maps_link,
        "source": "google",
    }


def _osm_provider_search(lat: float, lon: float, location_hint: str, limit: int) -> list[dict[str, Any]]:
    structured = _nominatim_provider_search(lat, lon, location_hint, limit)
    if structured:
        return structured

    try:
        radius_options = [8000, 12000]
        for radius in radius_options:
            elements = _overpass_search(lat, lon, radius=radius)
            structured = [_format_osm_provider(element, lat, lon) for element in elements]
            structured = [item for item in structured if item]
            structured = _dedupe_and_rank(structured)
            if structured:
                return structured[:limit]
    except Exception:
        return []

    return []


def _geocode_with_nominatim(location: str) -> tuple[float, float]:
    query = urlencode({"q": location, "format": "jsonv2", "limit": 1})
    payload = _fetch_json(f"{NOMINATIM_SEARCH_URL}?{query}")
    if not payload:
        raise ValueError("Invalid location")
    return float(payload[0]["lat"]), float(payload[0]["lon"])


def _reverse_geocode(lat: float, lon: float) -> str:
    query = urlencode({"lat": lat, "lon": lon, "format": "jsonv2"})
    payload = _fetch_json(f"{NOMINATIM_REVERSE_URL}?{query}")
    address = payload.get("address", {})
    parts = [
        address.get("city"),
        address.get("town"),
        address.get("municipality"),
        address.get("state"),
    ]
    compact = ", ".join(part for part in parts if part)
    return compact or payload.get("display_name") or f"{lat:.4f},{lon:.4f}"


def _google_autocomplete(query: str, api_key: str, limit: int) -> list[dict[str, str]]:
    params = {
        "input": query,
        "types": "(cities)",
        "components": "country:in",
        "key": api_key,
    }
    payload = _fetch_json(f"{GOOGLE_AUTOCOMPLETE_URL}?{urlencode(params)}")
    suggestions = []
    for item in payload.get("predictions", [])[:limit]:
        suggestions.append(
            {
                "description": item.get("description", ""),
                "place_id": item.get("place_id", ""),
            }
        )
    return suggestions


def _nominatim_autocomplete(query: str, limit: int) -> list[dict[str, str]]:
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": limit,
        "addressdetails": 1,
    }
    payload = _fetch_json(f"{NOMINATIM_SEARCH_URL}?{urlencode(params)}")
    return [
        {
            "description": item.get("display_name", ""),
            "place_id": str(item.get("place_id", "")),
        }
        for item in payload[:limit]
    ]


def _nominatim_provider_search(lat: float, lon: float, location_hint: str, limit: int) -> list[dict[str, Any]]:
    searches = [
        "women clinic",
        "maternity hospital",
        "fertility clinic",
        "gynecology clinic",
    ]
    providers: list[dict[str, Any]] = []
    delta = 0.3
    viewbox = f"{lon - delta},{lat + delta},{lon + delta},{lat - delta}"
    for search in searches:
        params = {
            "q": f"{search} in {location_hint}",
            "format": "jsonv2",
            "limit": limit,
            "viewbox": viewbox,
            "bounded": 1,
        }
        payload = _fetch_json(f"{NOMINATIM_SEARCH_URL}?{urlencode(params)}")
        for item in payload:
            display_name = item.get("display_name", "")
            if not _is_relevant_specialist(display_name):
                continue
            item_lat = float(item.get("lat"))
            item_lon = float(item.get("lon"))
            providers.append(
                {
                    "name": item.get("name") or display_name.split(",")[0],
                    "hospital_name": item.get("name") or display_name.split(",")[0],
                    "speciality": _speciality_label(display_name),
                    "address": display_name,
                    "distance_km": round(_haversine_km(lat, lon, item_lat, item_lon), 1),
                    "rating": None,
                    "maps_link": f"https://www.google.com/maps/search/?api=1&query={item_lat},{item_lon}",
                    "source": "nominatim",
                }
            )
        if providers:
            break
    return _dedupe_and_rank(providers)[:limit]


def _overpass_search(lat: float, lon: float, radius: int) -> list[dict[str, Any]]:
    query = f"""
[out:json][timeout:25];
(
  node["amenity"~"hospital|clinic|doctors"]["name"~"gyne|gynae|obstet|women|maternity|fertility|ivf", i](around:{radius},{lat},{lon});
  node["healthcare"~"doctor|clinic|hospital"]["name"~"gyne|gynae|obstet|women|maternity|fertility|ivf", i](around:{radius},{lat},{lon});
  node["healthcare:speciality"~"gyne|gynae|obstet|fertility", i](around:{radius},{lat},{lon});
  way["amenity"~"hospital|clinic|doctors"]["name"~"gyne|gynae|obstet|women|maternity|fertility|ivf", i](around:{radius},{lat},{lon});
  way["healthcare"~"doctor|clinic|hospital"]["name"~"gyne|gynae|obstet|women|maternity|fertility|ivf", i](around:{radius},{lat},{lon});
  way["healthcare:speciality"~"gyne|gynae|obstet|fertility", i](around:{radius},{lat},{lon});
  relation["amenity"~"hospital|clinic|doctors"]["name"~"gyne|gynae|obstet|women|maternity|fertility|ivf", i](around:{radius},{lat},{lon});
  relation["healthcare"~"doctor|clinic|hospital"]["name"~"gyne|gynae|obstet|women|maternity|fertility|ivf", i](around:{radius},{lat},{lon});
  relation["healthcare:speciality"~"gyne|gynae|obstet|fertility", i](around:{radius},{lat},{lon});
);
out center tags 80;
"""
    payload = _fetch_json(OVERPASS_URL, data=urlencode({"data": query}).encode("utf-8"))
    return payload.get("elements", [])


def _format_osm_provider(element: dict[str, Any], origin_lat: float, origin_lon: float) -> dict[str, Any] | None:
    tags = element.get("tags") or {}
    name = tags.get("name") or tags.get("operator") or "Gynecology provider"
    lat = element.get("lat") or element.get("center", {}).get("lat")
    lon = element.get("lon") or element.get("center", {}).get("lon")
    if lat is None or lon is None:
        return None

    text_blob = " ".join(str(value).lower() for value in tags.values())
    if not _is_relevant_specialist(f"{name.lower()} {text_blob}"):
        return None

    address_parts = [
        tags.get("addr:housenumber"),
        tags.get("addr:street"),
        tags.get("addr:suburb"),
        tags.get("addr:city"),
        tags.get("addr:district"),
        tags.get("addr:state"),
        tags.get("addr:postcode"),
        tags.get("addr:full"),
    ]
    address = ", ".join(part for part in address_parts if part) or "Address unavailable"
    distance_km = _haversine_km(origin_lat, origin_lon, float(lat), float(lon))

    return {
        "name": name,
        "hospital_name": tags.get("operator") or name,
        "speciality": _speciality_label(text_blob),
        "address": address,
        "distance_km": round(distance_km, 1),
        "rating": None,
        "maps_link": f"https://www.google.com/maps/search/?api=1&query={lat},{lon}",
        "source": "osm",
    }


def _dedupe_and_rank(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        key = (item["name"].strip().lower(), item["address"].strip().lower())
        existing = unique.get(key)
        if existing is None:
            unique[key] = item
            continue
        existing_rating = existing.get("rating") or 0
        current_rating = item.get("rating") or 0
        if current_rating > existing_rating:
            unique[key] = item

    def score(item: dict[str, Any]) -> tuple[float, float]:
        rating = float(item.get("rating") or 0)
        distance = float(item.get("distance_km") or 99)
        return (rating, -distance)

    return sorted(unique.values(), key=score, reverse=True)


def _is_relevant_specialist(text_blob: str) -> bool:
    lowered = text_blob.lower()
    if any(term in lowered for term in EXCLUDE_TERMS):
        return False
    return any(term in lowered for term in SPECIALTY_TERMS)


def _speciality_label(text_blob: str) -> str:
    lowered = text_blob.lower()
    if "fertility" in lowered or "ivf" in lowered:
        return "Gynecology and Fertility"
    if "obstet" in lowered:
        return "Gynecology and Obstetrics"
    return "Gynecology"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    )
    return 2 * radius_km * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _fetch_json(url: str, data: bytes | None = None) -> Any:
    request = Request(
        url,
        data=data,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise ValueError("Doctor search service unavailable") from exc


def _cache_fragment(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
