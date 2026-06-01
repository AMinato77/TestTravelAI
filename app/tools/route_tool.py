from __future__ import annotations

import os

import requests


def get_route_summary(start: tuple[float, float], end: tuple[float, float]) -> dict:
    """Return route distance and duration when OpenRouteService is configured."""
    api_key = os.getenv("OPENROUTESERVICE_API_KEY")
    if not api_key:
        return {
            "distance_km": None,
            "duration_minutes": None,
            "summary": "No OpenRouteService API key configured.",
        }

    response = requests.post(
        "https://api.openrouteservice.org/v2/directions/foot-walking",
        headers={"Authorization": api_key, "Content-Type": "application/json"},
        json={"coordinates": [[start[1], start[0]], [end[1], end[0]]]},
        timeout=20,
    )
    response.raise_for_status()
    segment = response.json()["features"][0]["properties"]["segments"][0]
    return {
        "distance_km": round(segment["distance"] / 1000, 2),
        "duration_minutes": round(segment["duration"] / 60),
        "summary": "walking",
    }
