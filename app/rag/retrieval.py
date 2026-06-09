from __future__ import annotations

from app.models.activity import Activity


def retrieve_local_fallback_activities(
    interests: list[str],
    destination: str,
    limit: int = 8,
) -> list[Activity]:
    """
    Legacy stub.

    We no longer use local demo activities. Real activity data now comes from
    Google Places in app/tools/places_tool.py.
    """
    return []


def retrieve_activities(interests: list[str], destination: str, limit: int = 8) -> list[Activity]:
    """Backward-compatible alias for older scripts."""
    return retrieve_local_fallback_activities(interests, destination, limit)
