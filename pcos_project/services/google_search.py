"""Google suggestion search helpers."""

from __future__ import annotations

import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from django.core.cache import cache


SUGGEST_URL = "https://suggestqueries.google.com/complete/search"
USER_AGENT = "Mozilla/5.0"
CACHE_SECONDS = 60 * 60
ALLOWED_TERMS = ("pcos", "polycystic ovary", "hormonal imbalance")


def get_suggestions(query: str, limit: int = 8) -> list[str]:
    """Return Google suggestion phrases filtered to PCOS-related topics."""
    normalized_query = _normalize_query(query)
    cache_key = f"suggest:{_cache_fragment(normalized_query)}:{limit}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    params = {
        "client": "firefox",
        "hl": "en",
        "q": normalized_query,
    }
    request = Request(f"{SUGGEST_URL}?{urlencode(params)}", headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise ValueError("Suggestion service unavailable") from exc

    suggestions = [
        item.strip()
        for item in payload[1]
        if item.strip() and _is_allowed_suggestion(item)
    ][:limit]
    cache.set(cache_key, suggestions, CACHE_SECONDS)
    return suggestions


def _normalize_query(query: str) -> str:
    cleaned = query.strip()
    lowered = cleaned.lower()
    if any(term in lowered for term in ALLOWED_TERMS):
        return cleaned
    return f"PCOS {cleaned}".strip()


def _is_allowed_suggestion(value: str) -> bool:
    lowered = value.lower()
    return any(term in lowered for term in ALLOWED_TERMS)


def _cache_fragment(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
