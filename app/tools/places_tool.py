from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path

import requests

from app.models.activity import Activity
from app.services.activity_ranker import (
    diversify_activities,
    estimate_cost,
    estimate_duration,
    normalize_category,
    rank_activities,
)


FALLBACK_ACTIVITIES = [
    Activity("Local food market", "food", "Authentic street food and local snacks.", 18, 2, False),
    Activity("Historic old town walk", "culture", "Self-guided walk through local architecture.", 0, 2, False),
    Activity("Interactive game cafe", "gaming", "Board games, arcade machines, and casual food.", 25, 2, True),
    Activity("Anime and comic shop area", "anime", "Specialty shops and pop culture stores.", 15, 1.5, True),
    Activity("City museum", "culture", "Indoor museum option for bad weather.", 16, 2, True),
    Activity("Local history museum", "culture", "Indoor exhibition focused on city history.", 14, 2, True),
    Activity("Public art gallery", "culture", "Indoor gallery with accessible culture and art.", 12, 1.5, True),
    Activity("Historic exhibition hall", "culture", "Indoor cultural stop with historical context.", 10, 1.5, True),
    Activity("Night market district", "nightlife", "Evening food stalls and casual nightlife.", 30, 2.5, False),
    Activity("Urban park viewpoint", "nature", "Relaxed outdoor break with city views.", 0, 1.5, False),
]

CACHE_DIR = Path("data/api_cache/geoapify")


def search_places(destination: str, interests: list[str], limit: int = 12) -> list[Activity]:
    """
    Search real places via Geoapify (when GEOAPIFY_API_KEY is configured).
    Falls back to curated local activities without a key.
    """
    api_key = os.getenv("GEOAPIFY_API_KEY")
    if not api_key:
        return _filter_fallback(interests, limit)

    lat_lon = _geocode_city(destination, api_key)
    if not lat_lon:
        return _filter_fallback(interests, limit)

    lat, lon = lat_lon
    places = _balanced_geoapify_places(
        destination=destination,
        lat=lat,
        lon=lon,
        interests=interests,
        limit=limit,
        api_key=api_key,
    )
    activities = [_activity_from_geoapify_place(place) for place in places]
    activities = [activity for activity in activities if activity.name]
    activities = _deduplicate_activities(activities)
    ranked = rank_activities(activities, interests)
    diversified = diversify_activities(ranked, limit)
    return diversified or _filter_fallback(interests, limit)


def search_indoor_places(destination: str, limit: int = 8) -> list[Activity]:
    api_key = os.getenv("GEOAPIFY_API_KEY")
    if not api_key:
        return [activity for activity in FALLBACK_ACTIVITIES if activity.indoor][:limit]

    lat_lon = _geocode_city(destination, api_key)
    if not lat_lon:
        return [activity for activity in FALLBACK_ACTIVITIES if activity.indoor][:limit]

    lat, lon = lat_lon
    categories = "tourism.sights,tourism.attraction,commercial.shopping_mall,catering.cafe"
    places = _cached_geoapify_places(
        destination=f"{destination}_indoor",
        lat=lat,
        lon=lon,
        categories=categories,
        limit=max(limit * 3, 20),
        api_key=api_key,
    )
    activities = [_activity_from_geoapify_place(place) for place in places]
    activities = [activity for activity in activities if activity.name]
    activities.extend(activity for activity in FALLBACK_ACTIVITIES if activity.indoor)
    activities = _deduplicate_activities(activities)
    indoor_activities = [activity for activity in activities if activity.indoor]
    ranked = rank_activities(indoor_activities or activities, ["culture", "local spots"])
    return ranked[:limit]


def _filter_fallback(interests: list[str], limit: int) -> list[Activity]:
    matches = [activity for activity in FALLBACK_ACTIVITIES if activity.matches_any_interest(interests)]
    return (matches or FALLBACK_ACTIVITIES)[:limit]


def _looks_indoor(kinds: str) -> bool:
    indoor_words = ["museums", "shops", "theatres", "galleries", "cinemas"]
    return any(word in kinds.lower() for word in indoor_words)


def _geocode_city(destination: str, api_key: str) -> tuple[float, float] | None:
    response = requests.get(
        "https://api.geoapify.com/v1/geocode/search",
        params={"text": destination, "format": "json", "apiKey": api_key, "limit": 1},
        timeout=20,
    )
    if response.status_code != 200:
        return None
    data = response.json()
    results = data.get("results") or []
    if not results:
        return None
    top = results[0]
    lat = top.get("lat")
    lon = top.get("lon")
    if lat is None or lon is None:
        return None
    return float(lat), float(lon)


def _geoapify_places(
    *,
    lat: float,
    lon: float,
    categories: str,
    radius_m: int = 5000,
    limit: int = 20,
    api_key: str,
) -> list[dict]:
    response = requests.get(
        "https://api.geoapify.com/v2/places",
        params={
            "categories": categories,
            "filter": f"circle:{lon},{lat},{radius_m}",
            "bias": f"proximity:{lon},{lat}",
            "limit": limit,
            "apiKey": api_key,
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    places: list[dict] = []
    for feature in data.get("features", []):
        props = feature.get("properties", {}) or {}
        coords = feature.get("geometry", {}).get("coordinates", [None, None])
        places.append(
            {
                "name": props.get("name"),
                "categories": props.get("categories", []),
                "address": props.get("formatted"),
                "city": props.get("city"),
                "country": props.get("country"),
                "distance_m": props.get("distance"),
                "lat": coords[1],
                "lon": coords[0],
                "opening_hours": props.get("opening_hours"),
                "website": props.get("website"),
                "place_id": props.get("place_id"),
            }
        )
    return places


def _cached_geoapify_places(
    *,
    destination: str,
    lat: float,
    lon: float,
    categories: str,
    limit: int,
    api_key: str,
) -> list[dict]:
    cache_path = _cache_path(destination=destination, categories=categories, lat=lat, lon=lon, limit=limit)
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    places = _geoapify_places(lat=lat, lon=lon, categories=categories, limit=limit, api_key=api_key)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(places, file, ensure_ascii=True, indent=2)
    return places


def _balanced_geoapify_places(
    *,
    destination: str,
    lat: float,
    lon: float,
    interests: list[str],
    limit: int,
    api_key: str,
) -> list[dict]:
    grouped_categories = _category_groups_for_interests(interests)
    per_group_limit = max(5, min(12, limit))
    places: list[dict] = []
    for label, categories in grouped_categories.items():
        group_places = _cached_geoapify_places(
            destination=f"{destination}_{label}",
            lat=lat,
            lon=lon,
            categories=categories,
            limit=per_group_limit,
            api_key=api_key,
        )
        places.extend(group_places)
    return places


def _cache_path(*, destination: str, categories: str, lat: float, lon: float, limit: int) -> Path:
    key = f"{destination}_{round(lat, 3)}_{round(lon, 3)}_{categories}_{limit}".lower()
    safe_key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return CACHE_DIR / f"{safe_key}.json"


def _categories_for_interests(interests: list[str]) -> str:
    # Keep categories conservative; invalid category strings cause Geoapify to return 400.
    categories: set[str] = {"tourism.sights", "tourism.attraction", "catering.restaurant"}
    normalized = {interest.strip().lower() for interest in interests}
    if "food" in normalized or "street food" in normalized:
        categories.update({"catering.fast_food", "catering.cafe", "catering.bar"})
    # For now, keep culture/history under general tourism categories to avoid invalid strings.
    if "nightlife" in normalized:
        categories.update({"catering.bar"})
    if "nature" in normalized:
        categories.update({"leisure.park"})
    return ",".join(sorted(categories))


def _category_groups_for_interests(interests: list[str]) -> dict[str, str]:
    normalized = {interest.strip().lower() for interest in interests}
    groups = {
        "culture": "tourism.sights,tourism.attraction",
        "food": "catering.restaurant,catering.cafe,catering.fast_food,catering.bar",
    }
    if "nature" in normalized:
        groups["nature"] = "leisure.park"
    if "nightlife" in normalized:
        groups["nightlife"] = "catering.bar"
    if any(value in normalized for value in ["local", "local spots", "hidden gems"]):
        groups["local_mix"] = "tourism.sights,tourism.attraction,catering.restaurant,catering.cafe"
    return groups


def _activity_from_geoapify_place(place: dict) -> Activity:
    name = place.get("name") or ""
    categories = place.get("categories") or []
    category = normalize_category(categories)
    description_parts = []
    if place.get("address"):
        description_parts.append(place["address"])
    if place.get("opening_hours"):
        description_parts.append(f"Hours: {place['opening_hours']}")
    description = " | ".join(description_parts)
    indoor = any(
        key in str(categories).lower()
        for key in ["museum", "gallery", "theatre", "cinema", "shopping_mall", "cafe", "restaurant"]
    )
    return Activity(
        name=name,
        category=category,
        description=description,
        cost=estimate_cost(category, categories),
        duration_hours=estimate_duration(category),
        indoor=indoor,
        latitude=place.get("lat"),
        longitude=place.get("lon"),
        distance_m=place.get("distance_m"),
        source="geoapify",
    )


def _deduplicate_activities(activities: list[Activity]) -> list[Activity]:
    unique: list[Activity] = []
    seen_names: set[str] = set()
    seen_nearby: set[tuple[str, float | None, float | None]] = set()
    for activity in activities:
        normalized_name = _normalize_name(activity.name)
        nearby_key = (
            normalized_name,
            round(activity.latitude, 4) if activity.latitude is not None else None,
            round(activity.longitude, 4) if activity.longitude is not None else None,
        )
        if normalized_name in seen_names or nearby_key in seen_nearby:
            continue
        seen_names.add(normalized_name)
        seen_nearby.add(nearby_key)
        unique.append(activity)
    return unique


def _normalize_name(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_name.lower()).strip()


def _search_places_raw(
    lat: float,
    lon: float,
    categories: str = "tourism.sights,tourism.attraction,catering.restaurant",
    radius: int = 5000,
    limit: int = 20,
) -> list[dict]:
    api_key = os.getenv("GEOAPIFY_API_KEY")
    if not api_key:
        raise ValueError("GEOAPIFY_API_KEY fehlt in der .env-Datei.")
    return _geoapify_places(lat=lat, lon=lon, categories=categories, radius_m=radius, limit=limit, api_key=api_key)


if __name__ == "__main__":
    places = _search_places_raw(
        lat=41.3874,
        lon=2.1686,
        categories="tourism.sights,tourism.attraction,catering.restaurant",
        radius=5000,
        limit=10,
    )
    for place in places:
        print(place)
