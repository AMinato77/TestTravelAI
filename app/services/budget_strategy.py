from __future__ import annotations

from app.models.itinerary import Itinerary
from app.models.user_profile import UserProfile


def target_budget_range(budget: float, profile: UserProfile | None = None) -> tuple[float, float]:
    """Return the intended activity/experience spend range for a trip budget."""
    style = (profile.travel_style if profile else "balanced").lower()
    preference = (profile.budget_preference if profile else "medium").lower()

    if style == "budget" or preference == "low":
        return budget * 0.35, budget * 0.75
    if style == "luxury" or preference == "high":
        return budget * 0.65, budget * 0.95
    if style == "relaxed":
        return budget * 0.45, budget * 0.85
    return budget * 0.55, budget * 0.9


def budget_utilization(itinerary: Itinerary, budget: float) -> float:
    if budget <= 0:
        return 0.0
    return itinerary.total_cost / budget


def under_budget_gap(itinerary: Itinerary, budget: float, profile: UserProfile | None = None) -> float:
    target_min, _ = target_budget_range(budget, profile)
    return max(0.0, target_min - itinerary.total_cost)


def add_budget_upgrades(
    itinerary: Itinerary,
    budget: float,
    profile: UserProfile | None = None,
) -> Itinerary:
    """Add explicit experience-budget allocations when a valid plan is too thin."""
    gap = under_budget_gap(itinerary, budget, profile)
    if gap <= 0 or not itinerary.days:
        return itinerary

    max_total_upgrade = max(0.0, budget - itinerary.total_cost)
    gap = min(gap, max_total_upgrade)
    if gap < 20:
        return itinerary

    days = [day for day in itinerary.days if day.activities]
    if not days:
        days = itinerary.days

    per_day = round(gap / len(days), 2)
    for day in days:
        category_hint = _dominant_day_category(day.activities)
        day.notes.append(
            f"Optional {itinerary.currency} {per_day:g} reserve suggested for stronger {category_hint} experiences; not counted as a booked activity."
        )
    return itinerary


def _dominant_day_category(activities: list[Activity]) -> str:
    counts: dict[str, int] = {}
    for activity in activities:
        counts[activity.category] = counts.get(activity.category, 0) + 1
    if not counts:
        return "local"
    return max(counts, key=counts.get)


def _upgrade_name(category: str) -> str:
    if category in {"food", "street_food"}:
        return "Premium local dining reservation budget"
    if category == "gaming":
        return "Premium gaming or arcade experience budget"
    if category == "sport":
        return "Premium sport ticket or activity budget"
    if category in {"culture", "history"}:
        return "Premium guided visit or ticket budget"
    return "Curated local experience budget"
