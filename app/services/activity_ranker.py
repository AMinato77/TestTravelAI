from __future__ import annotations

from app.models.activity import Activity


CHAIN_BRANDS = [
    "starbucks",
    "mcdonald",
    "burger king",
    "kfc",
    "subway",
    "pans & company",
]


def normalize_category(categories: list[str]) -> str:
    joined = " ".join(categories).lower()
    if any(value in joined for value in ["restaurant", "cafe", "fast_food"]):
        return "food"
    if "bar" in joined or "nightclub" in joined:
        return "nightlife"
    if any(value in joined for value in ["museum", "gallery", "artwork", "sights", "attraction", "tourism"]):
        return "culture"
    if "park" in joined or "natural" in joined:
        return "nature"
    if "shopping" in joined:
        return "shopping"
    return "place"


def estimate_cost(category: str, categories: list[str]) -> float:
    joined = " ".join(categories).lower()
    if "restaurant" in joined:
        return 25.0
    if "cafe" in joined:
        return 12.0
    if "bar" in joined or category == "nightlife":
        return 20.0
    if any(value in joined for value in ["museum", "gallery"]):
        return 15.0
    if "attraction" in joined or "sights" in joined:
        return 10.0
    if "park" in joined or "natural" in joined:
        return 0.0
    return 8.0


def estimate_duration(category: str) -> float:
    if category == "culture":
        return 2.0
    if category == "nightlife":
        return 2.5
    return 1.5


def score_activity(activity: Activity, interests: list[str]) -> int:
    score = 0
    text = f"{activity.name} {activity.category} {activity.description}".lower()
    normalized_interests = {interest.strip().lower() for interest in interests}

    for interest in normalized_interests:
        if interest and interest in text:
            score += 3

    if activity.category == "culture":
        score += 2
    if activity.category == "food" and ("food" in normalized_interests or "street food" in normalized_interests):
        score += 3
    if activity.category == "nature" and "nature" in normalized_interests:
        score += 2
    if any(value in normalized_interests for value in ["local", "local spots", "hidden gems"]):
        score += 1

    name = activity.name.lower()
    if any(chain in name for chain in CHAIN_BRANDS):
        score -= 6
    if any(value in normalized_interests for value in ["local", "local spots", "hidden gems"]):
        if any(chain in name for chain in CHAIN_BRANDS):
            score -= 4

    return score


def rank_activities(activities: list[Activity], interests: list[str]) -> list[Activity]:
    return sorted(
        activities,
        key=lambda activity: (
            -score_activity(activity, interests),
            activity.distance_m if activity.distance_m is not None else float("inf"),
            activity.name.lower(),
        ),
    )


def diversify_activities(activities: list[Activity], limit: int) -> list[Activity]:
    if limit <= 0:
        return []

    max_per_category = max(2, limit // 2)
    selected: list[Activity] = []
    category_counts: dict[str, int] = {}

    for activity in activities:
        count = category_counts.get(activity.category, 0)
        if count >= max_per_category:
            continue
        selected.append(activity)
        category_counts[activity.category] = count + 1
        if len(selected) >= limit:
            return selected

    for activity in activities:
        if activity in selected:
            continue
        selected.append(activity)
        if len(selected) >= limit:
            break

    return selected
