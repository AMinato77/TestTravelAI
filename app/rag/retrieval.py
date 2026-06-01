from __future__ import annotations

from app.models.activity import Activity
from app.tools.places_tool import FALLBACK_ACTIVITIES


def retrieve_activities(interests: list[str], destination: str, limit: int = 8) -> list[Activity]:
    """Local retrieval placeholder. Chroma ingestion will replace this in the next phase."""
    scored: list[tuple[int, Activity]] = []
    for activity in FALLBACK_ACTIVITIES:
        score = sum(
            1
            for interest in interests
            if interest.lower() in f"{activity.name} {activity.category} {activity.description}".lower()
        )
        if destination:
            score += 0
        scored.append((score, activity))

    ranked = [activity for score, activity in sorted(scored, key=lambda item: item[0], reverse=True) if score > 0]
    return (ranked or FALLBACK_ACTIVITIES)[:limit]

