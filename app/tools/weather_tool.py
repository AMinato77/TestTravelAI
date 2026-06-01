from __future__ import annotations

import os

import requests


BASE_URL = "https://api.weatherapi.com/v1/forecast.json"
RAIN_THRESHOLD = 60


def get_weather(destination: str, days: int = 3) -> dict:
    """Return an agent-friendly forecast summary for validation and optimization."""
    forecast = get_weather_forecast(destination, days=days)
    rainy_days = [day for day in forecast if day["is_rainy"]]
    return {
        "provider": "weatherapi" if _weather_api_key() else "fallback",
        "summary": _build_summary(forecast),
        "rain_expected": bool(rainy_days),
        "max_rain_chance": max((day["rain_chance"] for day in forecast), default=0),
        "temperature_c": forecast[0]["avg_temp_c"] if forecast else None,
        "forecast": forecast,
    }


def get_weather_forecast(city: str, days: int = 3) -> list[dict]:
    api_key = _weather_api_key()
    days = max(1, min(int(days), 14))
    if not api_key:
        return _fallback_forecast(city, days)

    response = requests.get(
        BASE_URL,
        params={
            "key": api_key,
            "q": city,
            "days": days,
            "aqi": "no",
            "alerts": "no",
            "lang": "de",
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()

    forecast: list[dict] = []
    for day in data["forecast"]["forecastday"]:
        rain_chance = int(day["day"].get("daily_chance_of_rain") or 0)
        forecast.append(
            {
                "date": day["date"],
                "city": data["location"]["name"],
                "country": data["location"]["country"],
                "condition": day["day"]["condition"]["text"],
                "avg_temp_c": round(day["day"]["avgtemp_c"], 1),
                "max_temp_c": round(day["day"]["maxtemp_c"], 1),
                "rain_chance": rain_chance,
                "humidity": day["day"]["avghumidity"],
                "is_rainy": rain_chance >= RAIN_THRESHOLD,
            }
        )
    return forecast


def _weather_api_key() -> str | None:
    return os.getenv("WEATHER_API_KEY") or os.getenv("Weather_API_KEY")


def _build_summary(forecast: list[dict]) -> str:
    if not forecast:
        return "No forecast available."
    first = forecast[0]
    return (
        f"{first['city']}: {first['condition']}, "
        f"{first['avg_temp_c']} C avg, max rain chance "
        f"{max(day['rain_chance'] for day in forecast)}%."
    )


def _fallback_forecast(city: str, days: int) -> list[dict]:
    return [
        {
            "date": f"day_{day_number}",
            "city": city,
            "country": "unknown",
            "condition": "Fallback weather",
            "avg_temp_c": 22.0,
            "max_temp_c": 25.0,
            "rain_chance": 10,
            "humidity": 55,
            "is_rainy": False,
        }
        for day_number in range(1, days + 1)
    ]
