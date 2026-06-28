from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import requests

from app.agents.query_planning_agent import PlaceQuery
from app.models.activity import Activity
from app.services.destination_normalizer import destination_matches_text, normalize_destination


GOOGLE_PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
CACHE_DIR = Path("data/api_cache/google_places")


def search_places(
    destination: str,
    queries: list[str] | list[PlaceQuery],
    limit: int = 20,
    avoid: list[str] | None = None,
) -> list[Activity]:
    activities, _metadata = search_places_with_metadata(destination, queries, limit=limit, avoid=avoid)
    return activities


def search_places_with_metadata(
    destination: str,
    queries: list[str] | list[PlaceQuery],
    limit: int = 20,
    avoid: list[str] | None = None,
) -> tuple[list[Activity], dict]:
    """Search Google Places with concrete text queries created by the Query Planning Agent."""
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_PLACES_API_KEY fehlt in der .env-Datei.")

    destination = normalize_destination(destination)
    planned_queries = _normalize_queries(destination, queries, avoid or [])
    if not planned_queries:
        planned_queries = [PlaceQuery(query=f"best things to do {destination}", reason="Generic fallback.", source="places_tool")]

    per_query_limit = max(5, min(20, math.ceil(limit / max(1, len(planned_queries))) + 4))
    activities: list[Activity] = []
    cache_hits = 0

    for planned_query in planned_queries:
        if _cache_path(planned_query.query, per_query_limit).exists():
            cache_hits += 1
        for place in _cached_google_text_search(api_key, planned_query.query, per_query_limit):
            activity = _place_to_activity(place, planned_query)
            if activity and _activity_matches_destination(activity, destination):
                activities.append(activity)

    activities = _deduplicate(activities)
    activities = _filter_low_quality(activities)
    activities = _sort_by_quality(activities)
    return activities[:limit], {
        "query_count": len(planned_queries),
        "cache_hits": cache_hits,
        "queries": [
            {
                "query": query.query,
                "reason": query.reason,
                "source": query.source,
                "must_have": query.must_have,
            }
            for query in planned_queries
        ],
    }


def search_indoor_places(destination: str, limit: int = 8) -> list[Activity]:
    return search_places(
        destination=destination,
        queries=[
            f"indoor activities {destination}",
            f"museums galleries indoor attractions {destination}",
            f"shopping centers indoor experiences {destination}",
        ],
        limit=limit,
    )


def _normalize_queries(destination: str, queries: list[str] | list[PlaceQuery], avoid: list[str]) -> list[PlaceQuery]:
    result: list[PlaceQuery] = []
    seen: set[str] = set()
    for item in queries:
        if isinstance(item, PlaceQuery):
            query = item
        else:
            query = PlaceQuery(query=str(item), source="raw")
        text = " ".join(query.query.strip().split())
        if not text:
            continue
        if destination.lower() not in text.lower():
            text = f"{text} {destination}"
        if _query_conflicts_with_avoid(text, avoid):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(PlaceQuery(query=text, reason=query.reason, source=query.source, must_have=query.must_have))
    return result


def _query_conflicts_with_avoid(query: str, avoid: list[str]) -> bool:
    text = query.lower()
    return any(term.strip().lower() and term.strip().lower() in text for term in avoid)


def _cached_google_text_search(api_key: str, query: str, limit: int) -> list[dict]:
    cache_path = _cache_path(query, limit)
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    places = _google_text_search(api_key, query, limit)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(places, file, ensure_ascii=True, indent=2)
    return places


def _google_text_search(api_key: str, query: str, limit: int) -> list[dict]:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "places.displayName,"
            "places.formattedAddress,"
            "places.rating,"
            "places.userRatingCount,"
            "places.types,"
            "places.primaryType,"
            "places.location,"
            "places.websiteUri,"
            "places.googleMapsUri,"
            "places.regularOpeningHours"
        ),
    }
    payload = {
        "textQuery": query,
        "languageCode": "en",
        "maxResultCount": limit,
    }

    response = requests.post(GOOGLE_PLACES_URL, headers=headers, json=payload, timeout=20)
    response.raise_for_status()
    return response.json().get("places", [])


def _place_to_activity(place: dict, planned_query: PlaceQuery) -> Activity | None:
    name = (place.get("displayName") or {}).get("text")
    if not name:
        return None

    types = _place_types(place)
    category = _normalize_category(types, planned_query.query, name)
    location = place.get("location") or {}

    return Activity(
        name=name,
        category=category,
        description=_build_description(place, category, planned_query),
        cost=_estimate_cost(category),
        duration_hours=_estimate_duration(category),
        indoor=_estimate_indoor(category, types),
        latitude=location.get("latitude"),
        longitude=location.get("longitude"),
        distance_m=None,
        source="google_places",
    )


def _place_types(place: dict) -> list[str]:
    types = place.get("types") or []
    primary_type = place.get("primaryType")
    if primary_type and primary_type not in types:
        return [primary_type, *types]
    return types


def _build_description(place: dict, category: str, planned_query: PlaceQuery) -> str:
    address = place.get("formattedAddress", "")
    rating = place.get("rating")
    review_count = place.get("userRatingCount")
    maps_url = place.get("googleMapsUri")
    website = place.get("websiteUri")
    types = _place_types(place)

    parts = [
        f"Category: {category}",
        f"Matched query: {planned_query.query}",
        f"Matched must-have: {', '.join(planned_query.must_have)}" if planned_query.must_have else "",
        f"Address: {address}" if address else "",
        f"Rating: {rating}/5" if rating else "",
        f"Reviews: {review_count}" if review_count else "",
        f"Types: {', '.join(types[:6])}" if types else "",
        f"Website: {website}" if website else "",
        f"Google Maps: {maps_url}" if maps_url else "",
    ]
    return " | ".join(part for part in parts if part)


def _normalize_category(types: list[str], query: str, name: str) -> str:
    joined = " ".join(types).lower()
    query_text = f"{query} {name}".lower()
    if any(value in joined for value in ["restaurant", "cafe", "bakery", "meal", "food"]):
        return "food"
    if any(value in joined for value in ["bar", "night_club"]):
        return "nightlife"
    if any(value in joined for value in ["park", "garden", "natural_feature", "hiking"]):
        return "nature"
    if any(value in joined for value in ["shopping_mall", "market", "store", "book_store"]):
        return "shopping"
    if any(value in joined for value in ["stadium", "gym", "sports", "fitness"]):
        return "sport"
    if any(value in joined for value in ["movie_theater", "amusement_center", "video_game", "casino"]):
        return "entertainment"
    if any(value in joined for value in ["museum", "art_gallery", "tourist_attraction", "historical_landmark", "church", "castle"]):
        return "culture"
    if any(term in query_text for term in ["shop", "store", "market", "merchandise", "figure", "collectible"]):
        return "shopping"
    if any(term in query_text for term in ["food", "restaurant", "cafe", "ramen", "dining"]):
        return "food"
    return "activity"


def _estimate_cost(category: str) -> float:
    if category == "food":
        return 25.0
    if category in {"culture", "entertainment"}:
        return 18.0
    if category == "nature":
        return 0.0
    if category == "nightlife":
        return 25.0
    if category in {"shopping", "sport"}:
        return 20.0
    return 12.0


def _estimate_duration(category: str) -> float:
    if category in {"culture", "entertainment", "shopping", "sport"}:
        return 2.0
    if category == "nightlife":
        return 2.5
    return 1.5


def _estimate_indoor(category: str, types: list[str]) -> bool:
    joined = " ".join(types).lower()
    if category in {"food", "nightlife", "shopping", "entertainment"}:
        return True
    if any(value in joined for value in ["museum", "art_gallery", "store", "restaurant", "cafe", "theater"]):
        return True
    if category == "nature" or "park" in joined:
        return False
    return False


def _filter_low_quality(activities: list[Activity]) -> list[Activity]:
    result: list[Activity] = []
    for activity in activities:
        if len(activity.name.strip()) < 3:
            continue
        description = activity.description.lower()
        if any(value in description for value in ["lodging", "hotel"]):
            continue
        result.append(activity)
    return result


def _activity_matches_destination(activity: Activity, destination: str) -> bool:
    address = _description_field(activity.description, "Address")
    if address:
        return destination_matches_text(destination, address)
    return destination_matches_text(destination, activity.description)


def _sort_by_quality(activities: list[Activity]) -> list[Activity]:
    return sorted(activities, key=_score_activity, reverse=True)


def _score_activity(activity: Activity) -> float:
    text = activity.description.lower()
    rating = _extract_float(text, "rating: ", "/5")
    reviews = _extract_int(text, "reviews: ")
    score = 0.0
    if rating:
        score += rating * 10
    if reviews:
        score += min(math.log10(reviews + 1) * 8, 40)
    if "google maps:" in text:
        score += 3
    if "website:" in text:
        score += 2
    return score


def _extract_float(text: str, start: str, end: str) -> float:
    if start not in text:
        return 0.0
    try:
        return float(text.split(start, 1)[1].split(end, 1)[0])
    except ValueError:
        return 0.0


def _extract_int(text: str, start: str) -> int:
    if start not in text:
        return 0
    try:
        return int(text.split(start, 1)[1].split("|", 1)[0].strip())
    except ValueError:
        return 0


def _deduplicate(activities: list[Activity]) -> list[Activity]:
    seen: set[str] = set()
    result: list[Activity] = []
    for activity in activities:
        key = activity.name.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        result.append(activity)
    return result


def _description_field(description: str, label: str) -> str:
    marker = f"{label}:"
    for part in str(description or "").split("|"):
        cleaned = part.strip()
        if cleaned.lower().startswith(marker.lower()):
            return cleaned.split(":", 1)[1].strip()
    return ""


def _cache_path(query: str, limit: int) -> Path:
    safe_query = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")
    return CACHE_DIR / f"{safe_query}_{limit}.json"
