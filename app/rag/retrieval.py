from __future__ import annotations

from app.models.activity import Activity
from app.tools.places_tool import FALLBACK_ACTIVITIES


def retrieve_activities(interests: list[str], destination: str, limit: int = 8) -> list[Activity]:
    """Local retrieval placeholder. Chroma ingestion will replace this in the next phase."""
    scored: list[tuple[int, Activity]] = []
    for activity in FALLBACK_ACTIVITIES:
        score = sum(_score_activity_interest(activity, interest) for interest in interests)
        if destination:
            score += 0
        scored.append((score, activity))

    ranked = [activity for score, activity in sorted(scored, key=lambda item: item[0], reverse=True) if score > 0]
    return ranked[:limit]


def _score_activity_interest(activity: Activity, interest: str) -> int:
    normalized = interest.strip().lower()
    if not normalized:
        return 0
    name = activity.name.lower()
    category = activity.category.lower()
    description = activity.description.lower()

    if normalized == category:
        return 3
    if normalized in name:
        return 2
    if normalized == "history" and category == "culture":
        return 2
    if normalized == "local spots" and "local" in description:
        return 1
    if normalized not in {"culture", "history"} and normalized in description:
        return 1
    return 0
