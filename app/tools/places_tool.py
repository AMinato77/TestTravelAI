from __future__ import annotations

import math
import os
import re
from pathlib import Path

import requests

from app.models.activity import Activity
from app.services.destination_normalizer import normalize_destination
from app.tools.openai_runtime import demo_fallback_enabled, generate_json


GOOGLE_PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
CACHE_DIR = Path("data/api_cache/google_places")

INTEREST_QUERIES = {
    "culture": [
        "best museums in {destination}",
        "best cultural attractions in {destination}",
        "art galleries and cultural sites in {destination}",
    ],
    "history": [
        "historical landmarks in {destination}",
        "historic attractions in {destination}",
        "historic architecture in {destination}",
    ],
    "food": [
        "best local restaurants in {destination}",
        "best local food experiences in {destination}",
        "famous food spots in {destination}",
    ],
    "street food": [
        "best street food markets in {destination}",
        "street food vendors in {destination}",
        "food markets in {destination}",
    ],
    "local spots": [
        "hidden gems in {destination}",
        "local neighborhoods in {destination}",
        "local experiences in {destination}",
    ],
    "hidden gems": [
        "hidden gems in {destination}",
        "local experiences in {destination}",
    ],
    "nature": [
        "best parks in {destination}",
        "best viewpoints in {destination}",
        "best outdoor attractions in {destination}",
    ],
    "nightlife": [
        "best nightlife in {destination}",
        "best bars in {destination}",
        "live music venues in {destination}",
    ],
    "shopping": [
        "best shopping streets in {destination}",
        "local markets in {destination}",
        "shopping districts in {destination}",
    ],
    "sport": [
        "sports activities in {destination}",
        "stadiums in {destination}",
        "sports venues in {destination}",
    ],
    "gaming": [
        "gaming arcades in {destination}",
        "video game stores in {destination}",
        "esports venues in {destination}",
    ],
    "anime": [
        "anime shops in {destination}",
        "manga stores in {destination}",
        "otaku attractions in {destination}",
    ],
    "technology": [
        "technology stores in {destination}",
        "electronics districts in {destination}",
        "computer stores in {destination}",
    ],
    "photography": [
        "best photo spots in {destination}",
        "scenic viewpoints in {destination}",
        "instagrammable places in {destination}",
    ],
    "architecture": [
        "architectural landmarks in {destination}",
        "famous buildings in {destination}",
        "historic architecture in {destination}",
    ],
}


def search_places(
    destination: str,
    interests: list[str],
    limit: int = 20,
    avoid: list[str] | None = None,
) -> list[Activity]:
    """
    Search real activities with Google Places.

    This is the only active activity API now. There is no Geoapify fallback and
    no local demo fallback. If Google returns nothing, the app receives an empty
    list and the issue becomes visible instead of hidden.
    """
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_PLACES_API_KEY fehlt in der .env-Datei.")

    destination = normalize_destination(destination)
    avoid = avoid or []
    activities: list[Activity] = []
    queries = _queries_for_interests(destination, interests, avoid=avoid)
    per_query_limit = max(5, min(20, math.ceil(limit / max(1, len(queries))) + 4))

    for interest, query in queries:
        places = _cached_google_text_search(api_key, query, per_query_limit)
        for place in places:
            activity = _place_to_activity(place, interest)
            if activity:
                activities.append(activity)

    activities = _deduplicate(activities)
    activities = _filter_low_quality(activities)
    activities = _sort_by_quality(activities)
    activities = _diversify(activities, limit)
    return activities[:limit]


def search_indoor_places(destination: str, limit: int = 8) -> list[Activity]:
    """Search real indoor alternatives for rainy days."""
    return search_places(
        destination=destination,
        interests=["culture", "food", "shopping"],
        limit=limit,
    )


def _queries_for_interests(
    destination: str,
    interests: list[str],
    avoid: list[str] | None = None,
) -> list[tuple[str, str]]:
    destination = normalize_destination(destination)
    ai_queries = _ai_queries_for_interests(destination, interests, avoid or [])
    if ai_queries:
        return ai_queries

    return _template_queries_for_interests(destination, interests)


def _ai_queries_for_interests(
    destination: str,
    interests: list[str],
    avoid: list[str],
) -> list[tuple[str, str]]:
    if demo_fallback_enabled():
        return []
    try:
        data = generate_json(
            system_prompt=(
                "You are a Google Places query planning agent for a travel app. "
                "Create precise text search queries for real activities in the destination. "
                "Use only the provided destination and user interests. "
                "Respect avoid preferences by not creating queries for avoided topics. "
                "Return strict JSON with key queries. queries is a list of objects with "
                "keys interest and query. Keep 1-3 queries per important interest, max 8 total. "
                "Queries must be useful for Google Places Text Search and include the destination."
            ),
            payload={
                "destination": destination,
                "interests": interests,
                "avoid": avoid,
                "examples": [
                    {"interest": "gaming", "query": f"gaming arcades in {destination}"},
                    {"interest": "food", "query": f"local restaurants in {destination}"},
                ],
            },
            model_env="OPENAI_PLACES_QUERY_MODEL",
        )
    except Exception:
        return []

    queries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in data.get("queries", []):
        if not isinstance(item, dict):
            continue
        interest = str(item.get("interest") or "general").strip().lower()
        query = str(item.get("query") or "").strip()
        if not query or destination.lower() not in query.lower():
            continue
        if _query_conflicts_with_avoid(query, avoid):
            continue
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append((interest, query))
        if len(queries) >= 8:
            break
    return queries


def _template_queries_for_interests(destination: str, interests: list[str]) -> list[tuple[str, str]]:
    queries: list[tuple[str, str]] = []
    seen: set[str] = set()

    for interest in interests:
        normalized = interest.strip().lower()
        templates = INTEREST_QUERIES.get(normalized)
        if not templates:
            templates = ["best things to do related to " + normalized + " in {destination}"]
        for template in templates:
            query = template.format(destination=destination)
            if query not in seen:
                seen.add(query)
                queries.append((normalized, query))

    if not queries:
        queries.append(("general", f"best things to do in {destination}"))
    return queries


def _query_conflicts_with_avoid(query: str, avoid: list[str]) -> bool:
    text = query.lower()
    avoid_text = " ".join(avoid).lower()
    if any(term in avoid_text for term in ["culture", "museum", "museums"]):
        if any(term in text for term in ["museum", "museums", "cultural", "culture", "gallery", "historic"]):
            return True
    if any(term in avoid_text for term in ["food", "restaurant", "cafe"]):
        if any(term in text for term in ["food", "restaurant", "cafe", "dining"]):
            return True
    if any(term in avoid_text for term in ["nightlife", "club"]):
        if any(term in text for term in ["nightlife", "club", "bar"]):
            return True
    return False


def _cached_google_text_search(api_key: str, query: str, limit: int) -> list[dict]:
    cache_path = _cache_path(query, limit)
    if cache_path.exists():
        import json

        with cache_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    places = _google_text_search(api_key, query, limit)

    import json

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


def _place_to_activity(place: dict, requested_interest: str) -> Activity | None:
    name = (place.get("displayName") or {}).get("text")
    if not name:
        return None

    types = _place_types(place)
    category = _normalize_category(types, requested_interest, name)
    location = place.get("location") or {}

    return Activity(
        name=name,
        category=category,
        description=_build_description(place, category),
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


def _build_description(place: dict, category: str) -> str:
    address = place.get("formattedAddress", "")
    rating = place.get("rating")
    review_count = place.get("userRatingCount")
    maps_url = place.get("googleMapsUri")
    website = place.get("websiteUri")
    types = _place_types(place)

    parts = [
        f"Category: {category}",
        f"Address: {address}" if address else "",
        f"Rating: {rating}/5" if rating else "",
        f"Reviews: {review_count}" if review_count else "",
        f"Types: {', '.join(types[:6])}" if types else "",
        f"Website: {website}" if website else "",
        f"Google Maps: {maps_url}" if maps_url else "",
    ]
    return " | ".join(part for part in parts if part)


def _normalize_category(types: list[str], requested_interest: str, name: str) -> str:
    joined = " ".join(types).lower()
    requested_interest = requested_interest.strip().lower()
    name_lower = name.lower()

    if requested_interest in {"anime", "gaming", "technology", "photography", "architecture", "local spots"}:
        if _types_are_reasonable_for_interest(joined, requested_interest, name_lower):
            return requested_interest

    if any(value in joined for value in ["restaurant", "cafe", "bakery", "meal", "food"]):
        if requested_interest == "gaming" and any(value in name_lower for value in ["game", "gaming", "esport", "e-sport"]):
            return "gaming"
        return "street_food" if requested_interest == "street food" else "food"
    if any(value in joined for value in ["bar", "night_club"]):
        return "nightlife"
    if any(value in joined for value in ["park", "garden", "natural_feature", "hiking"]):
        return "nature"
    if any(
        value in joined
        for value in [
            "museum",
            "art_gallery",
            "tourist_attraction",
            "historical_landmark",
            "church",
            "castle",
            "performing_arts_theater",
        ]
    ):
        if requested_interest == "history":
            return "history"
        if requested_interest == "architecture":
            return "architecture"
        if requested_interest == "photography":
            return "photography"
        if requested_interest in {"anime", "technology"} and any(
            value in name_lower for value in ["anime", "manga", "gundam", "akihabara", "electric"]
        ):
            return requested_interest
        return "culture"
    if any(value in joined for value in ["shopping_mall", "market", "store", "book_store"]):
        if requested_interest == "anime" and any(value in joined for value in ["book_store", "comic", "toy_store"]):
            return "anime"
        if requested_interest == "technology" and any(value in joined for value in ["electronics", "computer", "store"]):
            return "technology"
        return "shopping"
    if any(value in joined for value in ["stadium", "gym", "sports", "fitness"]):
        return "sport"
    if any(value in joined for value in ["movie_theater", "amusement_center", "video_game", "casino"]):
        return "gaming"
    return "place"


def _types_are_reasonable_for_interest(joined_types: str, requested_interest: str, name: str) -> bool:
    if requested_interest == "anime":
        return any(value in joined_types for value in ["book_store", "store", "shopping", "tourist_attraction"]) or any(
            value in name for value in ["anime", "manga", "gundam", "otaku"]
        )
    if requested_interest == "gaming":
        return any(value in joined_types for value in ["amusement", "store", "casino", "tourist_attraction"]) or any(
            value in name for value in ["game", "gaming", "esport", "e-sport", "nintendo"]
        )
    if requested_interest == "technology":
        return any(value in joined_types for value in ["electronics", "store", "shopping"]) or any(
            value in name for value in ["tech", "camera", "computer", "akihabara", "electric"]
        )
    if requested_interest == "photography":
        return any(value in joined_types for value in ["tourist_attraction", "park", "point_of_interest"])
    if requested_interest == "architecture":
        return any(value in joined_types for value in ["tourist_attraction", "church", "museum", "point_of_interest"])
    if requested_interest == "local spots":
        return any(value in joined_types for value in ["tourist_attraction", "restaurant", "cafe", "park", "point_of_interest"])
    return False


def _estimate_cost(category: str) -> float:
    if category in {"food", "street_food"}:
        return 25.0
    if category in {"culture", "history", "architecture", "photography"}:
        return 15.0
    if category == "nature":
        return 0.0
    if category == "nightlife":
        return 25.0
    if category == "shopping":
        return 20.0
    if category == "sport":
        return 20.0
    if category in {"gaming", "anime", "technology", "local spots"}:
        return 18.0
    return 10.0


def _estimate_duration(category: str) -> float:
    if category in {"culture", "history", "architecture", "photography"}:
        return 2.0
    if category in {"food", "street_food"}:
        return 1.5
    if category == "nature":
        return 1.5
    if category == "nightlife":
        return 2.5
    if category in {"shopping", "sport", "gaming", "anime", "technology", "local spots"}:
        return 2.0
    return 1.5


def _estimate_indoor(category: str, types: list[str]) -> bool:
    joined = " ".join(types).lower()
    if category in {"food", "street_food", "nightlife", "shopping", "gaming", "anime", "technology"}:
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
        if activity.category == "place":
            continue
        result.append(activity)
    return result


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
    if activity.category in {"culture", "history", "food", "street_food", "nature", "architecture", "photography"}:
        score += 5
    if "google maps:" in text:
        score += 3
    if "website:" in text:
        score += 2
    return score


def _extract_float(text: str, start: str, end: str) -> float:
    if start not in text:
        return 0.0
    try:
        value = text.split(start, 1)[1].split(end, 1)[0]
        return float(value)
    except ValueError:
        return 0.0


def _extract_int(text: str, start: str) -> int:
    if start not in text:
        return 0
    try:
        value = text.split(start, 1)[1].split("|", 1)[0].strip()
        return int(value)
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


def _diversify(activities: list[Activity], limit: int) -> list[Activity]:
    max_per_category = max(3, limit // 2)
    counts: dict[str, int] = {}
    selected: list[Activity] = []

    for activity in activities:
        count = counts.get(activity.category, 0)
        if count >= max_per_category:
            continue
        selected.append(activity)
        counts[activity.category] = count + 1
        if len(selected) >= limit:
            break
    return selected


def _cache_path(query: str, limit: int) -> Path:
    safe_query = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")
    return CACHE_DIR / f"{safe_query}_{limit}.json"
