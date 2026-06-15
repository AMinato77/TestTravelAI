from __future__ import annotations

from app.models.activity import Activity
from app.models.itinerary import Itinerary, ItineraryDay
from app.models.user_profile import UserProfile
from app.services.budget_strategy import add_budget_upgrades


def build_rule_based_itinerary(
    destination: str,
    days: int,
    budget: float,
    activities: list[Activity],
    weather: dict,
    profile: UserProfile,
) -> Itinerary:
    days = max(1, min(days, 14))
    max_activities = _max_activities_per_day(profile)
    remaining = activities[:]
    plan_days: list[ItineraryDay] = []
    daily_budget = budget / days if days else budget

    for day_number in range(1, days + 1):
        day_weather = _weather_for_day(weather, day_number)
        days_left = days - day_number + 1
        activity_cap = min(max_activities, _fair_activity_cap(len(remaining), days_left, max_activities))
        selected = _select_day_activities(
            candidates=remaining,
            max_activities=activity_cap,
            daily_budget=daily_budget,
            rainy=bool(day_weather and day_weather.get("is_rainy")),
        )
        remaining = [activity for activity in remaining if activity not in selected]
        notes = _day_notes(day_weather, selected, profile)
        plan_days.append(ItineraryDay(day=day_number, activities=selected, notes=notes))

    itinerary = Itinerary(destination=destination, days=plan_days)
    return add_budget_upgrades(itinerary, budget, profile)


def _select_day_activities(
    candidates: list[Activity],
    max_activities: int,
    daily_budget: float,
    rainy: bool,
) -> list[Activity]:
    pool = _prioritize_for_weather(candidates, rainy)
    selected: list[Activity] = []
    used_categories: set[str] = set()
    current_cost = 0.0

    for activity in pool:
        if len(selected) >= max_activities:
            break
        if current_cost + activity.cost > daily_budget and selected:
            continue
        if activity.category in used_categories and len(used_categories) < 2:
            continue
        selected.append(activity)
        used_categories.add(activity.category)
        current_cost += activity.cost

    if len(selected) < max_activities:
        for activity in pool:
            if activity in selected:
                continue
            if current_cost + activity.cost > daily_budget and selected:
                continue
            selected.append(activity)
            current_cost += activity.cost
            if len(selected) >= max_activities:
                break

    return selected


def _prioritize_for_weather(candidates: list[Activity], rainy: bool) -> list[Activity]:
    if not rainy:
        return candidates
    return sorted(
        candidates,
        key=lambda activity: (
            not activity.indoor,
            activity.distance_m if activity.distance_m is not None else float("inf"),
            activity.name.lower(),
        ),
    )


def _max_activities_per_day(profile: UserProfile) -> int:
    if profile.travel_style == "relaxed":
        return 2
    if profile.travel_style == "adventure":
        return 4
    return 3


def _fair_activity_cap(remaining_count: int, days_left: int, max_activities: int) -> int:
    if remaining_count <= 0 or days_left <= 0:
        return max_activities
    return max(1, min(max_activities, -(-remaining_count // days_left)))


def _weather_for_day(weather: dict, day_number: int) -> dict | None:
    forecast = weather.get("forecast") or []
    index = day_number - 1
    if 0 <= index < len(forecast):
        return forecast[index]
    return None


def _day_notes(day_weather: dict | None, selected: list[Activity], profile: UserProfile) -> list[str]:
    notes: list[str] = []
    if profile.travel_style == "relaxed":
        notes.append("Relaxed pacing: limited number of main activities.")
    if day_weather and day_weather.get("is_rainy"):
        notes.append(f"Rain-aware planning: rain chance is {day_weather['rain_chance']}%.")
        if all(activity.indoor for activity in selected):
            notes.append("Indoor activities were prioritized.")
    return notes
