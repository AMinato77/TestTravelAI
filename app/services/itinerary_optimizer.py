from __future__ import annotations

import re

from app.models.activity import Activity
from app.models.itinerary import Itinerary, ValidationResult
from app.models.user_profile import UserProfile
from app.services.budget_strategy import add_budget_upgrades


def optimize_itinerary_rules(
    itinerary: Itinerary,
    validation: ValidationResult,
    alternatives: list[Activity],
    budget: float,
    weather: dict,
    profile: UserProfile | None = None,
    constraints: dict | None = None,
) -> Itinerary:
    avoid = profile.avoid if profile else []
    safe_alternatives = [
        activity for activity in alternatives if not _activity_conflicts_with_avoid(activity, avoid)
    ]
    indoor_alternatives = [activity for activity in safe_alternatives if activity.indoor]
    cheap_alternatives = sorted(safe_alternatives, key=lambda activity: activity.cost)

    for issue in validation.issues:
        if issue.issue_type == "rain_conflict":
            _replace_rain_conflict(itinerary, issue.day, issue.activity, indoor_alternatives, weather)
        elif issue.issue_type in {"schedule_overload", "day_overload"}:
            _trim_day(itinerary, issue.day, profile)
        elif issue.issue_type == "budget_exceeded":
            _reduce_budget(itinerary, budget, cheap_alternatives)
        elif issue.issue_type == "budget_underused":
            add_budget_upgrades(itinerary, budget, profile)
        elif issue.issue_type == "preference_conflict":
            _replace_preference_conflict(itinerary, issue.day, issue.activity, safe_alternatives, profile)
        elif issue.issue_type == "empty_day":
            _fill_empty_day(itinerary, issue.day, safe_alternatives)

    if itinerary.total_cost > budget:
        _reduce_budget(itinerary, budget, cheap_alternatives)
    add_budget_upgrades(itinerary, budget, profile)

    return itinerary


def _replace_rain_conflict(
    itinerary: Itinerary,
    day_number: int | None,
    activity_name: str | None,
    indoor_alternatives: list[Activity],
    weather: dict,
) -> None:
    if day_number is None or not activity_name or not indoor_alternatives:
        return
    day = _find_day(itinerary, day_number)
    if not day:
        return

    for index, activity in enumerate(day.activities):
        if activity.name != activity_name:
            continue
        replacement = _best_indoor_replacement(indoor_alternatives, itinerary, preferred_category=activity.category)
        if not replacement:
            return
        day.activities[index] = replacement
        day_weather = _weather_for_day(weather, day_number)
        rain_chance = day_weather.get("rain_chance", "?") if day_weather else "?"
        if replacement.category == activity.category:
            reason = f"same category ({activity.category}) and rain chance is {rain_chance}%"
        else:
            reason = f"rain chance is {rain_chance}%"
        _add_note(day, f"Replaced {activity.name} with {replacement.name} because {reason}.")
        return


def _trim_day(itinerary: Itinerary, day_number: int | None, profile: UserProfile | None) -> None:
    day = _find_day(itinerary, day_number) if day_number is not None else None
    if not day or not day.activities:
        return

    max_activities = 2 if profile and profile.travel_style == "relaxed" else 3
    while len(day.activities) > max_activities:
        removed = day.activities.pop()
        _add_note(day, f"Removed {removed.name} to simplify the day.")

    while day.total_duration_hours > 8 and day.activities:
        removed = day.activities.pop()
        _add_note(day, f"Removed {removed.name} to reduce total duration.")


def _reduce_budget(itinerary: Itinerary, budget: float, cheap_alternatives: list[Activity]) -> None:
    attempts = 0
    while itinerary.total_cost > budget and attempts < 20:
        attempts += 1
        day = max(itinerary.days, key=lambda candidate: candidate.total_cost)
        if not day.activities:
            break
        expensive = max(day.activities, key=lambda activity: activity.cost)
        replacement = _cheaper_unused(expensive, cheap_alternatives, itinerary)
        if replacement:
            index = day.activities.index(expensive)
            day.activities[index] = replacement
            _add_note(day, f"Replaced {expensive.name} with {replacement.name} to reduce budget.")
            continue
        day.activities.remove(expensive)
        _add_note(day, f"Removed {expensive.name} to reduce budget.")


def _fill_empty_day(itinerary: Itinerary, day_number: int | None, alternatives: list[Activity]) -> None:
    day = _find_day(itinerary, day_number) if day_number is not None else None
    if not day:
        return
    for replacement in _unused_activities(alternatives, itinerary, count=2):
        day.activities.append(replacement)
        _add_note(day, f"Added {replacement.name} because the day was empty.")


def _replace_preference_conflict(
    itinerary: Itinerary,
    day_number: int | None,
    activity_name: str | None,
    alternatives: list[Activity],
    profile: UserProfile | None,
) -> None:
    if day_number is None or not activity_name:
        return
    day = _find_day(itinerary, day_number)
    if not day:
        return
    safe_alternatives = [
        candidate
        for candidate in alternatives
        if not _activity_conflicts_with_avoid(candidate, profile.avoid if profile else [])
    ]
    replacement = _first_unused(safe_alternatives, itinerary)
    for index, activity in enumerate(day.activities):
        if activity.name != activity_name:
            continue
        if replacement:
            day.activities[index] = replacement
            _add_note(day, f"Replaced {activity.name} with {replacement.name} because it conflicted with preferences.")
        else:
            day.activities.pop(index)
            _add_note(day, f"Removed {activity.name} because it conflicted with preferences.")
        return


def _content_tokens(text: str) -> list[str]:
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "und",
        "oder",
        "mit",
        "von",
        "für",
        "fuer",
        "eine",
        "einen",
        "der",
        "die",
        "das",
        "zu",
        "in",
        "im",
        "am",
        "an",
        "visit",
        "besuch",
        "besuchen",
        "sehen",
        "erleben",
        "shops",
        "restaurants",
    }
    tokens = re.findall(r"[a-z0-9äöüß]+", text.lower())
    return [token for token in tokens if len(token) > 2 and token not in stop_words]


def _find_day(itinerary: Itinerary, day_number: int) -> object | None:
    for day in itinerary.days:
        if day.day == day_number:
            return day
    return None


def _first_unused(candidates: list[Activity], itinerary: Itinerary) -> Activity | None:
    used = {activity.name for day in itinerary.days for activity in day.activities}
    for candidate in candidates:
        if candidate.name not in used:
            return candidate
    return None


def _unused_activities(candidates: list[Activity], itinerary: Itinerary, count: int) -> list[Activity]:
    used = {activity.name for day in itinerary.days for activity in day.activities}
    selected: list[Activity] = []
    used_categories: set[str] = set()
    for candidate in candidates:
        if candidate.name in used:
            continue
        if candidate.category in used_categories and len(used_categories) < count:
            continue
        selected.append(candidate)
        used.add(candidate.name)
        used_categories.add(candidate.category)
        if len(selected) >= count:
            return selected
    for candidate in candidates:
        if candidate.name in used:
            continue
        selected.append(candidate)
        used.add(candidate.name)
        if len(selected) >= count:
            break
    return selected


def _best_indoor_replacement(
    candidates: list[Activity],
    itinerary: Itinerary,
    preferred_category: str,
) -> Activity | None:
    used = {activity.name for day in itinerary.days for activity in day.activities}
    same_category = [
        candidate
        for candidate in candidates
        if candidate.category == preferred_category and candidate.name not in used
    ]
    if same_category:
        return same_category[0]

    unused = [candidate for candidate in candidates if candidate.name not in used]
    if unused:
        return unused[0]

    return None


def _cheaper_unused(expensive: Activity, candidates: list[Activity], itinerary: Itinerary) -> Activity | None:
    used = {activity.name for day in itinerary.days for activity in day.activities}
    for candidate in candidates:
        if candidate.name in used:
            continue
        if candidate.cost < expensive.cost:
            return candidate
    return None


def _weather_for_day(weather: dict, day_number: int) -> dict | None:
    forecast = weather.get("forecast") or []
    index = day_number - 1
    if 0 <= index < len(forecast):
        return forecast[index]
    if weather.get("rain_expected"):
        return {"is_rainy": True, "rain_chance": weather.get("max_rain_chance", 100)}
    return None


def _activity_conflicts_with_avoid(activity: Activity, avoid: list[str]) -> bool:
    haystack = f"{activity.name} {activity.category} {activity.description}".lower()
    avoid_text = " ".join(avoid).lower()
    if any(term in avoid_text for term in ["food", "restaurant", "cafe"]):
        return activity.category == "food" or any(
            term in haystack for term in ["restaurant", "food", "cafe", "café", "creperie", "crêperie"]
        )
    if any(term in avoid_text for term in ["museum", "museums"]):
        return "museum" in haystack or activity.category == "museum"
    if any(term in avoid_text for term in ["nightlife", "club"]):
        return activity.category == "nightlife" or any(term in haystack for term in ["club", "nightlife", "bar"])
    return any(term and term in haystack for term in avoid)


def _add_note(day, note: str) -> None:
    if day.notes is None:
        day.notes = []
    elif isinstance(day.notes, str):
        day.notes = [day.notes] if day.notes.strip() else []
    elif not isinstance(day.notes, list):
        day.notes = [str(day.notes)]
    day.notes.append(note)
