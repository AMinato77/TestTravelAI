from __future__ import annotations

from app.models.activity import Activity
from app.models.itinerary import Itinerary, ItineraryDay
from app.models.user_profile import UserProfile
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
) -> Itinerary:
    if demo_fallback_enabled():
        return build_rule_based_itinerary(destination, days, budget, activities, weather, profile)

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
            "travel_style, avoid list, and relaxed pacing."
        ),
        payload={
            "destination": destination,
            "duration_days": days,
            "budget": budget,
            "weather": weather,
            "profile": profile.to_dict(),
            "activities": activity_payload,
        },
        model_env="OPENAI_PLANNING_MODEL",
    )
    return _itinerary_from_plan(destination, data, activities, days)


def _itinerary_from_plan(
    destination: str,
    data: dict,
    activities: list[Activity],
    requested_days: int,
) -> Itinerary:
    by_name = {activity.name.strip().lower(): activity for activity in activities}
    plan_days: list[ItineraryDay] = []
    for day_data in data.get("days", []):
        selected = [
            by_name[name.strip().lower()]
            for name in day_data.get("activity_names", [])
            if name.strip().lower() in by_name
        ]
        plan_days.append(
            ItineraryDay(
                day=_parse_day_number(day_data.get("day"), len(plan_days) + 1),
                activities=selected,
                notes=_parse_notes(day_data.get("notes")),
            )
        )

    if not plan_days:
        return create_initial_itinerary(destination, requested_days, activities)

    return Itinerary(destination=destination, days=plan_days)


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
