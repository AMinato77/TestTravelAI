from __future__ import annotations

import os

import requests

from app.models.activity import Activity


FALLBACK_ACTIVITIES = [
    Activity("Local food market", "food", "Authentic street food and local snacks.", 18, 2, False),
    Activity("Historic old town walk", "culture", "Self-guided walk through local architecture.", 0, 2, False),
    Activity("Interactive game cafe", "gaming", "Board games, arcade machines, and casual food.", 25, 2, True),
    Activity("Anime and comic shop area", "anime", "Specialty shops and pop culture stores.", 15, 1.5, True),
    Activity("City museum", "culture", "Indoor museum option for bad weather.", 16, 2, True),
    Activity("Night market district", "nightlife", "Evening food stalls and casual nightlife.", 30, 2.5, False),
    Activity("Urban park viewpoint", "nature", "Relaxed outdoor break with city views.", 0, 1.5, False),
]


def search_places(destination: str, interests: list[str], limit: int = 12) -> list[Activity]:
    """Search OpenTripMap when configured; otherwise return curated fallback activities."""
    api_key = os.getenv("OPENTRIPMAP_API_KEY")
    if not api_key:
        return _filter_fallback(interests, limit)

    geo = requests.get(
        "https://api.opentripmap.com/0.1/en/places/geoname",
        params={"name": destination, "apikey": api_key},
        timeout=20,
    )
    geo.raise_for_status()
    location = geo.json()

    places = requests.get(
        "https://api.opentripmap.com/0.1/en/places/radius",
        params={
            "radius": 5000,
            "lon": location["lon"],
            "lat": location["lat"],
            "limit": limit,
            "apikey": api_key,
            "format": "json",
        },
        timeout=20,
    )
    places.raise_for_status()

    activities: list[Activity] = []
    for item in places.json():
        name = item.get("name")
        if not name:
            continue
        kinds = item.get("kinds", "sightseeing")
        point = item.get("point", {})
        activities.append(
            Activity(
                name=name,
                category=kinds.split(",")[0],
                description=kinds.replace(",", ", "),
                cost=0,
                duration_hours=1.5,
                indoor=_looks_indoor(kinds),
                latitude=point.get("lat"),
                longitude=point.get("lon"),
                source="opentripmap",
            )
        )
    return activities or _filter_fallback(interests, limit)


def _filter_fallback(interests: list[str], limit: int) -> list[Activity]:
    matches = [activity for activity in FALLBACK_ACTIVITIES if activity.matches_any_interest(interests)]
    return (matches or FALLBACK_ACTIVITIES)[:limit]


def _looks_indoor(kinds: str) -> bool:
    indoor_words = ["museums", "shops", "theatres", "galleries", "cinemas"]
    return any(word in kinds.lower() for word in indoor_words)

