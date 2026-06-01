from __future__ import annotations

from app.models.itinerary import Itinerary, ValidationIssue, ValidationResult
from app.models.user_profile import UserProfile


def validate_itinerary(
    itinerary: Itinerary,
    budget: float,
    weather: dict,
    profile: UserProfile | None = None,
) -> ValidationResult:
    issues: list[ValidationIssue] = []

    if itinerary.total_cost > budget:
        issues.append(
            ValidationIssue(
                severity="error",
                message=f"Budget exceeded: {itinerary.total_cost:.0f} > {budget:.0f} {itinerary.currency}.",
            )
        )

    for day in itinerary.days:
        if day.total_duration_hours > 8:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message="This day is packed with more than 8 hours of activities.",
                    day=day.day,
                )
            )
        if profile and profile.travel_style == "relaxed" and len(day.activities) > 2:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message="Relaxed travel style detected; this day has more than 2 major activities.",
                    day=day.day,
                )
            )
        day_weather = _weather_for_day(weather, day.day)
        if day_weather and day_weather.get("is_rainy"):
            outdoor = [activity.name for activity in day.activities if not activity.indoor]
            if outdoor:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        message=(
                            f"Rain chance {day_weather['rain_chance']}%; outdoor activities may need replacing: "
                            f"{', '.join(outdoor)}."
                        ),
                        day=day.day,
                    )
                )

    return ValidationResult(ok=not issues, issues=issues)


def _weather_for_day(weather: dict, day_number: int) -> dict | None:
    forecast = weather.get("forecast") or []
    index = day_number - 1
    if 0 <= index < len(forecast):
        return forecast[index]
    if weather.get("rain_expected"):
        return {"is_rainy": True, "rain_chance": weather.get("max_rain_chance", 100)}
    return None
