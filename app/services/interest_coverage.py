from __future__ import annotations

from collections import Counter

from app.models.activity import Activity
from app.models.itinerary import Itinerary


INTEREST_TERMS = {
    "anime": ["anime", "manga", "otaku", "comic", "comics", "gundam", "japan", "japanese"],
    "gaming": ["gaming", "game", "games", "arcade", "esport", "e-sport", "nintendo", "playstation", "xbox"],
    "food": ["food", "restaurant", "cafe", "tapas", "dining", "ramen", "sushi"],
    "street food": ["street food", "market", "food truck", "tapas", "ramen"],
    "shopping": ["shopping", "shop", "store", "mall", "market", "boutique"],
    "sport": ["sport", "stadium", "arena", "football", "soccer", "fussball", "fußball"],
    "local spots": ["local", "neighborhood", "neighbourhood", "park", "plaza", "square", "market"],
    "hidden gems": ["hidden", "local", "neighborhood", "neighbourhood"],
    "nature": ["nature", "park", "garden", "beach", "viewpoint"],
    "nightlife": ["nightlife", "bar", "club", "music"],
    "technology": ["technology", "electronics", "computer", "camera"],
    "culture": ["culture", "museum", "gallery", "art"],
    "history": ["history", "historic", "monument", "museum"],
}


def activity_matches_interest(activity: Activity, interest: str) -> bool:
    normalized = interest.strip().lower()
    if not normalized:
        return False
    category = activity.category.strip().lower()
    text = f"{activity.name} {activity.category} {activity.description}".lower()
    if category == normalized:
        return True
    if normalized == "street food" and category == "food":
        return True
    if normalized == "local spots" and category in {"local spots", "nature"}:
        return True
    terms = INTEREST_TERMS.get(normalized, [normalized])
    return any(term in text for term in terms)


def coverage_for_activities(activities: list[Activity], interests: list[str]) -> dict:
    cleaned = _clean_interests(interests)
    covered = {
        interest: [activity.name for activity in activities if activity_matches_interest(activity, interest)]
        for interest in cleaned
    }
    return {
        "covered": {interest: names for interest, names in covered.items() if names},
        "missing": [interest for interest, names in covered.items() if not names],
        "counts": {interest: len(names) for interest, names in covered.items()},
    }


def coverage_for_itinerary(itinerary: Itinerary, interests: list[str]) -> dict:
    activities = [activity for day in itinerary.days for activity in day.activities]
    return coverage_for_activities(activities, interests)


def rebalance_for_interest_coverage(
    selected: list[Activity],
    candidates: list[Activity],
    interests: list[str],
    limit: int,
) -> list[Activity]:
    """Prefer at least one candidate per requested interest when available."""
    cleaned = _clean_interests(interests)
    if not cleaned or limit <= 0:
        return selected[:limit]

    selected_by_name = {activity.name.strip().lower(): activity for activity in selected}
    ordered: list[Activity] = []
    used: set[str] = set()

    for interest in _coverage_priority(cleaned):
        if any(activity_matches_interest(activity, interest) for activity in ordered):
            continue
        existing = next((activity for activity in selected if activity_matches_interest(activity, interest)), None)
        candidate = existing or next(
            (activity for activity in candidates if activity_matches_interest(activity, interest)),
            None,
        )
        if not candidate:
            continue
        key = candidate.name.strip().lower()
        if key in used:
            continue
        ordered.append(candidate)
        used.add(key)

    for activity in selected:
        key = activity.name.strip().lower()
        if key in used:
            continue
        ordered.append(activity)
        used.add(key)
        if len(ordered) >= limit:
            return ordered[:limit]

    for activity in candidates:
        key = activity.name.strip().lower()
        if key in used:
            continue
        ordered.append(activity)
        used.add(key)
        if len(ordered) >= limit:
            break

    return ordered[:limit]


def interest_coverage_notes(itinerary: Itinerary, interests: list[str]) -> list[str]:
    coverage = coverage_for_itinerary(itinerary, interests)
    notes: list[str] = []
    missing = coverage.get("missing", [])
    if missing:
        notes.append(f"Soft interest coverage gap: {', '.join(missing)}.")
    counts = coverage.get("counts", {})
    if counts:
        notes.append(
            "Interest coverage: "
            + ", ".join(f"{interest}={count}" for interest, count in counts.items())
            + "."
        )
    return notes


def _coverage_priority(interests: list[str]) -> list[str]:
    counter = Counter(interests)
    specific = ["anime", "gaming", "sport", "technology", "nightlife", "shopping", "food", "street food"]
    ordered = [interest for interest in specific if interest in counter]
    ordered.extend(interest for interest in interests if interest not in ordered)
    return ordered


def _clean_interests(interests: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for interest in interests:
        normalized = str(interest).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result
