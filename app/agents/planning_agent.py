from __future__ import annotations

from app.models.activity import Activity
from app.models.itinerary import Itinerary, ItineraryDay
from app.models.user_profile import UserProfile
from app.services.budget_strategy import target_budget_range
from app.services.interest_coverage import activity_matches_interest, coverage_for_activities, coverage_for_itinerary
from app.services.itinerary_builder import build_rule_based_itinerary
from app.tools.openai_runtime import demo_fallback_enabled, generate_json


def create_initial_itinerary(destination: str, days: int, activities: list[Activity]) -> Itinerary:
    plan_days: list[ItineraryDay] = []
    for day_number in range(1, max(1, days) + 1):
        start = (day_number - 1) * 3
        plan_days.append(ItineraryDay(day=day_number, activities=activities[start : start + 3]))
    return Itinerary(destination=destination, days=plan_days)


def plan_itinerary(
    destination: str,
    days: int,
    budget: float,
    activities: list[Activity],
    weather: dict,
    profile: UserProfile,
    constraints: dict | None = None,
) -> Itinerary:
    if demo_fallback_enabled():
        return build_rule_based_itinerary(destination, days, budget, activities, weather, profile)

    target_min, target_max = target_budget_range(budget, profile)
    constraints = constraints or {}
    candidate_interest_coverage = coverage_for_activities(activities, profile.interests)
    activity_payload = [
        {
            "name": activity.name,
            "category": activity.category,
            "cost": activity.cost,
            "duration_hours": activity.duration_hours,
            "indoor": activity.indoor,
            "description": activity.description,
        }
        for activity in activities
    ]
    data = generate_json(
        system_prompt=(
            "You are the Planning Agent for an adaptive travel planner. "
            "Return strict JSON with key days. days is a list of objects "
            "with day, activity_names, and notes. Choose only activity_names "
            "from the provided activity list. Respect budget, weather, "
            "travel_style, avoid list, user interests, and relaxed pacing. "
            "Do not repeat the same activity across days. "
            "Cover as many distinct user interests as possible. "
            "If the candidate list contains activities for specific interests such as anime, gaming, sport, "
            "technology, or shopping, include at least one matching activity for each of those interests when feasible. "
            "Do not create a bare-minimum "
            "plan when the budget is generous. Aim to use the target budget range with "
            "higher-quality matching activities, paid experiences, and explicit notes "
            "about why the spend level is appropriate. "
            "Write all user-facing notes in concise German."
        ),
        payload={
            "destination": destination,
            "duration_days": days,
            "budget": budget,
            "target_budget_range": {
                "min": round(target_min, 2),
                "max": round(target_max, 2),
                "currency": "EUR",
            },
            "weather": weather,
            "profile": profile.to_dict(),
            "constraints": constraints,
            "candidate_interest_coverage": candidate_interest_coverage,
            "activities": activity_payload,
        },
        model_env="OPENAI_PLANNING_MODEL",
    )
    return _itinerary_from_plan(destination, data, activities, days, profile.interests)


def _itinerary_from_plan(
    destination: str,
    data: dict,
    activities: list[Activity],
    requested_days: int,
    interests: list[str],
) -> Itinerary:
    by_name = {activity.name.strip().lower(): activity for activity in activities}
    used_names: set[str] = set()
    plan_days: list[ItineraryDay] = []
    for day_data in data.get("days", []):
        selected: list[Activity] = []
        for name in day_data.get("activity_names", []):
            key = str(name).strip().lower()
            if key not in by_name or key in used_names:
                continue
            selected.append(by_name[key])
            used_names.add(key)
        plan_days.append(
            ItineraryDay(
                day=_parse_day_number(day_data.get("day"), len(plan_days) + 1),
                activities=selected,
                notes=_parse_notes(day_data.get("notes")),
            )
        )

    if not plan_days:
        itinerary = create_initial_itinerary(destination, requested_days, activities)
        _repair_interest_coverage(itinerary.days, activities, interests, {activity.name.strip().lower() for day in itinerary.days for activity in day.activities})
        return itinerary

    _repair_day_count_and_empty_days(plan_days, activities, requested_days, used_names)
    _repair_interest_coverage(plan_days, activities, interests, used_names)
    return Itinerary(destination=destination, days=plan_days)


def _repair_day_count_and_empty_days(
    plan_days: list[ItineraryDay],
    activities: list[Activity],
    requested_days: int,
    used_names: set[str],
) -> None:
    while len(plan_days) < requested_days:
        plan_days.append(
            ItineraryDay(
                day=len(plan_days) + 1,
                activities=[],
                notes=["Added by planner repair because the model returned too few days."],
            )
        )

    del plan_days[requested_days:]

    for index, day in enumerate(plan_days, start=1):
        day.day = index
        if day.activities:
            continue
        replacements = _next_unused_activities(activities, used_names, count=2)
        if not replacements:
            day.notes.append("No unused API activity candidate was available for this day.")
            continue
        day.activities.extend(replacements)
        day.notes.append("Added by planner repair because the model returned an empty day.")
        used_names.update(activity.name.strip().lower() for activity in replacements)


def _next_unused_activities(activities: list[Activity], used_names: set[str], count: int) -> list[Activity]:
    selected: list[Activity] = []
    used_categories: set[str] = set()
    for activity in activities:
        key = activity.name.strip().lower()
        if key in used_names or activity.source == "budget_strategy":
            continue
        if activity.category in used_categories and len(used_categories) < count:
            continue
        selected.append(activity)
        used_categories.add(activity.category)
        if len(selected) >= count:
            return selected

    for activity in activities:
        key = activity.name.strip().lower()
        if key in used_names or activity in selected or activity.source == "budget_strategy":
            continue
        selected.append(activity)
        if len(selected) >= count:
            break
    return selected


def _repair_interest_coverage(
    plan_days: list[ItineraryDay],
    activities: list[Activity],
    interests: list[str],
    used_names: set[str],
) -> None:
    if not plan_days or not activities or not interests:
        return
    itinerary = Itinerary(destination="", days=plan_days)
    itinerary_coverage = coverage_for_itinerary(itinerary, interests)
    candidate_coverage = coverage_for_activities(activities, interests)

    missing = [
        interest
        for interest in _coverage_priority(interests)
        if interest in candidate_coverage.get("covered", {}) and interest in itinerary_coverage.get("missing", [])
    ]
    for interest in missing:
        candidate = next(
            (
                activity
                for activity in activities
                if activity.name.strip().lower() not in used_names and activity_matches_interest(activity, interest)
            ),
            None,
        )
        if not candidate:
            continue
        day = min(plan_days, key=lambda item: (item.total_duration_hours, len(item.activities)))
        if day.total_duration_hours + candidate.duration_hours > 8 or len(day.activities) >= 4:
            continue
        day.activities.append(candidate)
        used_names.add(candidate.name.strip().lower())
        day.notes.append(f"Added {candidate.name} to improve soft interest coverage for {interest}.")


def _coverage_priority(interests: list[str]) -> list[str]:
    priority = ["anime", "gaming", "sport", "technology", "nightlife", "shopping", "food", "street food"]
    cleaned = []
    seen = set()
    for interest in interests:
        normalized = str(interest).strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            cleaned.append(normalized)
    ordered = [interest for interest in priority if interest in seen]
    ordered.extend(interest for interest in cleaned if interest not in ordered)
    return ordered


def _parse_day_number(value, fallback: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        digits = "".join(char for char in value if char.isdigit())
        if digits:
            return int(digits)
    return fallback


def _parse_notes(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]
