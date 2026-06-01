from __future__ import annotations

from app.models.activity import Activity
from app.models.itinerary import Itinerary
from app.models.user_profile import UserProfile


def optimize_itinerary(
    itinerary: Itinerary,
    alternatives: list[Activity],
    budget: float,
    weather: dict,
    profile: UserProfile | None = None,
) -> Itinerary:
    indoor_alternatives = [activity for activity in alternatives if activity.indoor]
    cheap_alternatives = sorted(alternatives, key=lambda activity: activity.cost)

    for day in itinerary.days:
        if profile and profile.travel_style == "relaxed" and len(day.activities) > 2:
            day.activities = day.activities[:2]
            day.notes.append("Day simplified because the saved travel style is relaxed.")
        day_weather = _weather_for_day(weather, day.day)
        if day_weather and day_weather.get("is_rainy") and indoor_alternatives:
            day.activities = [
                indoor_alternatives[0] if not activity.indoor else activity
                for activity in day.activities
            ]
            day.notes.append(f"Outdoor activities were replaced because rain chance is {day_weather['rain_chance']}%.")

    while itinerary.total_cost > budget and cheap_alternatives:
        most_expensive_day = max(itinerary.days, key=lambda day: day.total_cost)
        if not most_expensive_day.activities:
            break
        expensive = max(most_expensive_day.activities, key=lambda activity: activity.cost)
        replacement = cheap_alternatives[0]
        index = most_expensive_day.activities.index(expensive)
        most_expensive_day.activities[index] = replacement
        most_expensive_day.notes.append(f"Replaced {expensive.name} to reduce cost.")
        if replacement.cost >= expensive.cost:
            break

    return itinerary


def _weather_for_day(weather: dict, day_number: int) -> dict | None:
    forecast = weather.get("forecast") or []
    index = day_number - 1
    if 0 <= index < len(forecast):
        return forecast[index]
    if weather.get("rain_expected"):
        return {"is_rainy": True, "rain_chance": weather.get("max_rain_chance", 100)}
    return None
