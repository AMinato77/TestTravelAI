from __future__ import annotations

from app.models.activity import Activity
from app.models.itinerary import Itinerary, ItineraryDay
from app.models.user_profile import UserProfile
from app.services.budget_strategy import target_budget_range
from app.services.itinerary_builder import build_rule_based_itinerary
from app.tools.openai_runtime import demo_fallback_enabled, generate_json


def create_initial_itinerary(destination: str, days: int, activities: list[Activity]) -> Itinerary:
    plan_days: list[ItineraryDay] = []
    for day_number in range(1, max(1, days) + 1):
        start = (day_number - 1) * 2
        plan_days.append(ItineraryDay(day=day_number, activities=activities[start : start + 2]))
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
    constraints = constraints or {}
    if demo_fallback_enabled():
        return build_rule_based_itinerary(destination, days, budget, activities, weather, profile)

    target_min, target_max = target_budget_range(budget, profile)
    data = generate_json(
        system_prompt=(
            "You are the Planning Agent for an adaptive travel planner. "
            "Create a realistic itinerary using only provided activity names. "
            "Prioritize concrete user wishes, preference notes, avoid constraints, weather, "
            "budget, and relaxed pacing. Do not repeat activities. "
            "Return strict JSON with key days. Each day has day, activity_names, notes. "
            "Write concise German user-facing notes."
        ),
        payload={
            "destination": destination,
            "duration_days": days,
            "budget": budget,
            "target_budget_range": {"min": round(target_min, 2), "max": round(target_max, 2), "currency": "EUR"},
            "weather": weather,
            "profile": profile.to_dict(),
            "constraints": constraints,
            "activities": [
                {
                    "name": activity.name,
                    "category": activity.category,
                    "cost": activity.cost,
                    "duration_hours": activity.duration_hours,
                    "indoor": activity.indoor,
                    "description": activity.description,
                }
                for activity in activities
            ],
        },
        model_env="OPENAI_PLANNING_MODEL",
    )
    return _itinerary_from_plan(destination, data, activities, days)


def _itinerary_from_plan(destination: str, data: dict, activities: list[Activity], requested_days: int) -> Itinerary:
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
        return create_initial_itinerary(destination, requested_days, activities)

    _repair_day_count_and_empty_days(plan_days, activities, requested_days, used_names)
    return Itinerary(destination=destination, days=plan_days)


def _repair_day_count_and_empty_days(
    plan_days: list[ItineraryDay],
    activities: list[Activity],
    requested_days: int,
    used_names: set[str],
) -> None:
    while len(plan_days) < requested_days:
        plan_days.append(ItineraryDay(day=len(plan_days) + 1, activities=[], notes=[]))
    del plan_days[requested_days:]
    for index, day in enumerate(plan_days, start=1):
        day.day = index
        if day.activities:
            continue
        replacements = _next_unused_activities(activities, used_names, count=2)
        day.activities.extend(replacements)
        used_names.update(activity.name.strip().lower() for activity in replacements)
        if replacements:
            day.notes.append("Ergaenzt, weil der Agent fuer diesen Tag keine Aktivitaeten ausgewaehlt hatte.")
        else:
            day.notes.append("Keine ungenutzten API-Kandidaten fuer diesen Tag verfuegbar.")


def _next_unused_activities(activities: list[Activity], used_names: set[str], count: int) -> list[Activity]:
    selected: list[Activity] = []
    for activity in activities:
        key = activity.name.strip().lower()
        if key in used_names or activity.source == "budget_strategy":
            continue
        selected.append(activity)
        if len(selected) >= count:
            break
    return selected


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
