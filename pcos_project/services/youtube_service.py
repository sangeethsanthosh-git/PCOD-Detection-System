"""YouTube educational content retrieval."""

from __future__ import annotations

import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from django.core.cache import cache


YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
USER_AGENT = "Mozilla/5.0"
CACHE_SECONDS = 60 * 60
ALLOWED_TERMS = ("pcos", "polycystic ovary", "hormonal imbalance")

CURATED_VIDEOS = [
    {
        "title": "Foods to Eat vs. Foods to Avoid: PCOS Edition",
        "channel": "CLS Health",
        "video_id": "N37YB3UY5bw",
        "topics": ("pcos", "diet", "treatment"),
    },
    {
        "title": "Polycystic Ovary Syndrome Made Easy",
        "channel": "Rhesus Medicine",
        "video_id": "YVQzolMgNp0",
        "topics": ("pcos", "overview", "education"),
    },
    {
        "title": "Understanding PCOS Symptoms and Treatment",
        "channel": "Natalie Crawford, MD",
        "video_id": "NcGeMWaF4ac",
        "topics": ("pcos", "treatment", "symptoms"),
    },
    {
        "title": "The PCOS Diet Plan Explained",
        "channel": "Boston IVF",
        "video_id": "TCFmxcWVDFw",
        "topics": ("pcos", "diet", "nutrition"),
    },
    {
        "title": "Exercise for Women with PCOS",
        "channel": "growwithjo",
        "video_id": "3F7fYGrCgVY",
        "topics": ("pcos", "exercise", "workout"),
    },
    {
        "title": "PCOS Weight Loss Workout",
        "channel": "Akshaya Agnes",
        "video_id": "YukpAFgNJM8",
        "topics": ("pcos", "exercise", "weight loss"),
    },
]


def search_videos(query: str, limit: int = 6) -> dict[str, object]:
    """Return YouTube results or curated fallback content."""
    normalized_query = _normalize_query(query)
    cache_key = f"videos:{_cache_fragment(normalized_query)}:{limit}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if api_key:
        try:
            payload = _search_via_api(normalized_query, api_key, limit)
            cache.set(cache_key, payload, CACHE_SECONDS)
            return payload
        except ValueError as exc:
            payload = {
                "items": _fallback_videos(normalized_query, limit),
                "message": str(exc),
            }
            cache.set(cache_key, payload, CACHE_SECONDS)
            return payload

    payload = {
        "items": _fallback_videos(normalized_query, limit),
        "message": "Showing curated educational videos because a YouTube API key is not configured.",
    }
    cache.set(cache_key, payload, CACHE_SECONDS)
    return payload


def _search_via_api(query: str, api_key: str, limit: int) -> dict[str, object]:
    params = {
        "part": "snippet",
        "type": "video",
        "safeSearch": "strict",
        "maxResults": min(limit * 2, 12),
        "q": query,
        "key": api_key,
    }
    request = Request(f"{YOUTUBE_SEARCH_URL}?{urlencode(params)}", headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code in {403, 429}:
            raise ValueError("YouTube API limit reached. Showing curated PCOS education videos instead.") from exc
        raise ValueError("Video search is temporarily unavailable.") from exc
    except (URLError, TimeoutError) as exc:
        raise ValueError("Video search is temporarily unavailable.") from exc

    items = []
    for item in data.get("items", []):
        video_id = item.get("id", {}).get("videoId")
        snippet = item.get("snippet", {})
        if not video_id:
            continue
        if not _is_pcos_video(snippet):
            continue
        items.append(
            {
                "title": snippet.get("title", "PCOS video"),
                "channel": snippet.get("channelTitle", "YouTube"),
                "thumbnail": (
                    snippet.get("thumbnails", {}).get("high", {}).get("url")
                    or snippet.get("thumbnails", {}).get("medium", {}).get("url")
                    or snippet.get("thumbnails", {}).get("default", {}).get("url")
                ),
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "embed_url": f"https://www.youtube.com/embed/{video_id}",
            }
        )
        if len(items) >= limit:
            break

    if not items:
        return {
            "items": _fallback_videos(query, limit),
            "message": "No strongly PCOS-related videos were returned, so curated education videos are shown instead.",
        }

    return {"items": items, "message": ""}


def _fallback_videos(query: str, limit: int) -> list[dict[str, str]]:
    tokens = {token.lower() for token in query.split() if token}
    ranked = sorted(
        CURATED_VIDEOS,
        key=lambda item: len(tokens.intersection(item["topics"])),
        reverse=True,
    )
    chosen = ranked[:limit]
    return [
        {
            "title": item["title"],
            "channel": item["channel"],
            "thumbnail": f"https://i.ytimg.com/vi/{item['video_id']}/hqdefault.jpg",
            "url": f"https://www.youtube.com/watch?v={item['video_id']}",
            "embed_url": f"https://www.youtube.com/embed/{item['video_id']}",
        }
        for item in chosen
    ]


def _normalize_query(query: str) -> str:
    lowered = query.strip().lower()
    if any(term in lowered for term in ALLOWED_TERMS):
        return query.strip()
    return f"PCOS {query.strip()}".strip()


def _is_pcos_video(snippet: dict[str, object]) -> bool:
    haystack = " ".join(
        [
            str(snippet.get("title", "")),
            str(snippet.get("description", "")),
            str(snippet.get("channelTitle", "")),
        ]
    ).lower()
    return any(term in haystack for term in ALLOWED_TERMS)


def _cache_fragment(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
