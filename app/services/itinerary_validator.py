from __future__ import annotations

import re

from app.models.itinerary import Itinerary, ValidationIssue, ValidationResult
from app.models.user_profile import UserProfile
from app.services.budget_strategy import budget_utilization, target_budget_range


def validate_itinerary_rules(
    itinerary: Itinerary,
    budget: float,
    weather: dict,
    profile: UserProfile | None = None,
    constraints: dict | None = None,
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    constraints = constraints or {}
    avoid_terms = _clean_terms((constraints.get("avoid") or []) + (profile.avoid if profile else []))

    if itinerary.total_cost > budget:
        issues.append(
            ValidationIssue(
                severity="error",
                issue_type="budget_exceeded",
                message=f"Budget exceeded: {itinerary.total_cost:.0f} > {budget:.0f} {itinerary.currency}.",
            )
        )
    elif budget >= 150:
        target_min, target_max = target_budget_range(budget, profile)
        if itinerary.total_cost < target_min - 5:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    issue_type="budget_underused",
                    message=(
                        f"Budget is underused: {itinerary.total_cost:.0f} of {budget:.0f} "
                        f"{itinerary.currency} planned ({budget_utilization(itinerary, budget):.0%}). "
                        f"Target range is {target_min:.0f}-{target_max:.0f} {itinerary.currency}."
                    ),
                )
            )

    for day in itinerary.days:
        if not day.activities:
            issues.append(
                ValidationIssue(
                    severity="error",
                    issue_type="empty_day",
                    message="No activities planned for this day.",
                    day=day.day,
                )
            )

        if day.total_duration_hours > 8:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    issue_type="day_overload",
                    message="This day is packed with more than 8 hours of activities.",
                    day=day.day,
                )
            )

        if profile and profile.travel_style == "relaxed" and len(day.activities) > 2:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    issue_type="schedule_overload",
                    message="Relaxed travel style detected; this day has more than 2 major activities.",
                    day=day.day,
                )
            )

        if avoid_terms:
            for activity in day.activities:
                if _activity_conflicts_with_avoid(activity, avoid_terms):
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            issue_type="preference_conflict",
                            activity=activity.name,
                            message="Activity conflicts with user avoid preferences.",
                            day=day.day,
                        )
                    )

        seen_names: set[str] = set()
        for activity in day.activities:
            if activity.source == "budget_strategy":
                continue
            key = activity.name.strip().lower()
            if key in seen_names:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        issue_type="duplicate_activity_same_day",
                        activity=activity.name,
                        message="The same activity is repeated within one day.",
                        day=day.day,
                    )
                )
            seen_names.add(key)

        day_weather = _weather_for_day(weather, day.day)
        if day_weather and day_weather.get("is_rainy"):
            for activity in day.activities:
                if activity.indoor:
                    continue
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        issue_type="rain_conflict",
                        activity=activity.name,
                        message=(
                            f"Outdoor activity planned on rainy day "
                            f"({day_weather['rain_chance']}% rain chance)."
                        ),
                        day=day.day,
                    )
                )

    used_days_by_name: dict[str, list[int]] = {}
    for day in itinerary.days:
        for activity in day.activities:
            if activity.source == "budget_strategy":
                continue
            used_days_by_name.setdefault(activity.name.strip().lower(), []).append(day.day)
    for activity_name, used_days in used_days_by_name.items():
        if len(used_days) > 1:
            issues.append(
                ValidationIssue(
                    severity="error",
                    issue_type="duplicate_activity_across_days",
                    activity=activity_name,
                    message=f"Activity is repeated across days: {used_days}.",
                )
            )

    expected_destination = itinerary.destination.strip().lower()
    if expected_destination:
        for day in itinerary.days:
            for activity in day.activities:
                if _activity_has_foreign_location(activity, expected_destination):
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            issue_type="destination_mismatch",
                            activity=activity.name,
                            day=day.day,
                            message=f"Activity appears to be outside the destination {itinerary.destination}.",
                        )
                    )

    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    return ValidationResult(
        ok=error_count == 0,
        issues=issues,
        error_count=error_count,
        warning_count=warning_count,
    )


def _weather_for_day(weather: dict, day_number: int) -> dict | None:
    forecast = weather.get("forecast") or []
    index = day_number - 1
    if 0 <= index < len(forecast):
        return forecast[index]
    if weather.get("rain_expected"):
        return {"is_rainy": True, "rain_chance": weather.get("max_rain_chance", 100)}
    return None


def _activity_conflicts_with_avoid(activity, avoid: list[str]) -> bool:
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


def _clean_terms(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value).strip().lower().split())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


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
        "besuch",
        "besuchen",
        "sehen",
        "erleben",
        "locations",
        "orte",
        "shops",
        "restaurants",
        "event",
        "events",
        "experience",
        "experiences",
    }
    tokens = re.findall(r"[a-z0-9äöüß]+", text.lower())
    return [token for token in tokens if len(token) > 2 and token not in stop_words]


def _activity_has_foreign_location(activity, expected_destination: str) -> bool:
    description = activity.description.lower()
    address_match = re.search(r"address:\s*([^|]+)", description)
    if address_match:
        address = address_match.group(1).strip()
        return bool(address and expected_destination not in address)
    return False
