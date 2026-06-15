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
    activities, _metadata = search_places_with_metadata(destination, interests, limit=limit, avoid=avoid)
    return activities


def search_places_with_metadata(
    destination: str,
    interests: list[str],
    limit: int = 20,
    avoid: list[str] | None = None,
) -> tuple[list[Activity], dict]:
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
    cache_hits = 0

    for interest, query in queries:
        if _cache_path(query, per_query_limit).exists():
            cache_hits += 1
        places = _cached_google_text_search(api_key, query, per_query_limit)
        for place in places:
            activity = _place_to_activity(place, interest)
            if activity and _activity_matches_destination(activity, destination):
                activities.append(activity)

    activities = _deduplicate(activities)
    activities = _filter_low_quality(activities)
    activities = _sort_by_quality(activities)
    activities = _diversify(activities, limit)
    return activities[:limit], {
        "query_count": len(queries),
        "cache_hits": cache_hits,
        "queries": [{"interest": interest, "query": query} for interest, query in queries],
    }


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
    priority_queries = _priority_template_queries(destination, interests, avoid or [])
    ai_queries = _ai_queries_for_interests(destination, interests, avoid or [])
    if ai_queries:
        return _merge_queries(priority_queries, ai_queries, max_count=8)

    return _merge_queries(priority_queries, _template_queries_for_interests(destination, interests), max_count=8)


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
                "Use only the provided destination and user interests. Interests are broad categories "
                "such as anime, sport, food, gaming, shopping, local spots, or nature. "
                "Respect avoid preferences by not creating queries for avoided topics. "
                "Return strict JSON with key queries. queries is a list of objects with "
                "keys interest and query. Keep 1-3 queries per important interest, max 8 total. "
                "Create category-level Google Places text-search queries. Queries must be useful for Google Places "
                "Text Search and include the destination."
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


def _priority_template_queries(destination: str, interests: list[str], avoid: list[str]) -> list[tuple[str, str]]:
    queries: list[tuple[str, str]] = []
    normalized = {interest.strip().lower() for interest in interests}
    templates: dict[str, list[str]] = {
        "anime": [
            "anime shops in {destination}",
            "manga stores in {destination}",
        ],
        "gaming": [
            "gaming arcades in {destination}",
            "video game stores in {destination}",
        ],
        "sport": [
            "stadiums in {destination}",
        ],
        "shopping": [
            "shopping districts in {destination}",
        ],
    }
    for interest, interest_templates in templates.items():
        if interest not in normalized:
            continue
        for template in interest_templates:
            query = template.format(destination=destination)
            if not _query_conflicts_with_avoid(query, avoid):
                queries.append((interest, query))
    return queries


def _merge_queries(*groups: list[tuple[str, str]], max_count: int) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for group in groups:
        for interest, query in group:
            key = query.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            result.append((interest, query))
            if len(result) >= max_count:
                return result
    return result


def _query_conflicts_with_avoid(query: str, avoid: list[str]) -> bool:
    text = query.lower()
    avoid_text = " ".join(avoid).lower()
    for term in avoid:
        normalized = str(term).strip().lower()
        if normalized and normalized in text:
            return True
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
        description=_build_description(place, category, requested_interest),
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


def _build_description(place: dict, category: str, requested_interest: str) -> str:
    address = place.get("formattedAddress", "")
    rating = place.get("rating")
    review_count = place.get("userRatingCount")
    maps_url = place.get("googleMapsUri")
    website = place.get("websiteUri")
    types = _place_types(place)

    parts = [
        f"Category: {category}",
        f"Matched query interest: {requested_interest}" if requested_interest else "",
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

    if _is_sport_interest(requested_interest, name_lower, joined):
        return "sport"

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


def _is_sport_interest(requested_interest: str, name: str, joined_types: str) -> bool:
    interest = requested_interest.lower()
    sport_markers = [
        "sport",
        "real madrid",
        "bernab",
        "football",
        "fussball",
        "fußball",
        "stadium",
        "stadion",
        "bullfighting",
        "stierkampf",
        "toros",
        "las ventas",
    ]
    if any(marker in interest for marker in sport_markers):
        return True
    if any(marker in name for marker in ["bernab", "real madrid", "las ventas", "bullfighting"]):
        return True
    return any(value in joined_types for value in ["stadium", "sports", "sports_complex", "sports_activity_location"])


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


def _activity_matches_destination(activity: Activity, destination: str) -> bool:
    destination_lower = normalize_destination(destination).lower()
    description = activity.description.lower()
    if _has_foreign_address(description, destination_lower):
        return False
    if destination_lower and destination_lower in description:
        return True

    country_fallbacks = {
        "tokyo": ["japan"],
        "osaka": ["japan"],
        "kyoto": ["japan"],
        "barcelona": ["spain"],
        "madrid": ["spain"],
        "paris": ["france"],
        "rome": ["italy"],
        "milan": ["italy"],
        "vienna": ["austria"],
        "brussels": ["belgium"],
    }
    allowed_countries = country_fallbacks.get(destination_lower, [])
    if allowed_countries and any(country in description for country in allowed_countries):
        foreign_city_markers = [
            "bogot",
            "paris",
            "costa rica",
            "oyama",
            "madrid",
            "barcelona",
            "tokyo",
            "osaka",
            "kyoto",
        ]
        return not any(marker in description for marker in foreign_city_markers if marker != destination_lower)

    return not any(marker in description for marker in _foreign_markers(destination_lower))


def _has_foreign_address(description: str, destination_lower: str) -> bool:
    address_match = re.search(r"address:\s*([^|]+)", description)
    if not address_match:
        return False
    address = address_match.group(1).strip().lower()
    if not address:
        return False
    if destination_lower and destination_lower in address:
        return False
    allowed_country = {
        "tokyo": "japan",
        "osaka": "japan",
        "kyoto": "japan",
        "madrid": "spain",
        "barcelona": "spain",
        "paris": "france",
        "rome": "italy",
        "milan": "italy",
        "vienna": "austria",
    }.get(destination_lower)
    if allowed_country and allowed_country in address and not any(marker in address for marker in _foreign_markers(destination_lower)):
        return False
    return any(marker in address for marker in _foreign_markers(destination_lower))


def _foreign_markers(destination_lower: str) -> list[str]:
    markers = [
        "usa",
        "united states",
        "los angeles",
        "california",
        "yokohama",
        "kawasaki",
        "saitama",
        "chiba",
        "colombia",
        "bogot",
        "costa rica",
        "france",
        "spain",
        "italy",
        "japan",
        "tokyo",
        "osaka",
        "kyoto",
        "madrid",
        "barcelona",
        "paris",
        "rome",
        "milan",
    ]
    destination_aliases = {
        "tokyo": {"tokyo", "japan"},
        "osaka": {"osaka", "japan"},
        "kyoto": {"kyoto", "japan"},
        "madrid": {"madrid", "spain"},
        "barcelona": {"barcelona", "spain"},
        "paris": {"paris", "france"},
        "rome": {"rome", "italy"},
        "milan": {"milan", "italy"},
    }.get(destination_lower, {destination_lower})
    return [marker for marker in markers if marker not in destination_aliases]


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
    if activity.category in {"anime", "gaming", "technology", "sport", "shopping", "local spots"}:
        score += 8
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
