from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    destination = sys.argv[1] if len(sys.argv) > 1 else "Barcelona"
    interests = sys.argv[2].split(",") if len(sys.argv) > 2 else ["food", "culture", "local spots"]
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    from app.tools.places_tool import search_indoor_places, search_places
    from app.tools.weather_tool import get_weather

    weather = get_weather(destination, days=days)
    activities = search_places(destination, interests, limit=10)
    indoor_alternatives = search_indoor_places(destination, limit=6) if weather.get("rain_expected") else []

    context = {
        "destination": destination,
        "interests": interests,
        "weather": weather,
        "activities": [asdict(activity) for activity in activities],
        "indoor_alternatives": [asdict(activity) for activity in indoor_alternatives],
        "planning_hints": {
            "rainy_days": [
                day["date"]
                for day in weather.get("forecast", [])
                if day.get("is_rainy")
            ],
            "indoor_alternatives": [
                activity.name
                for activity in indoor_alternatives
            ],
            "estimated_activity_budget": sum(activity.cost for activity in activities),
        },
    }
    print(json.dumps(context, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
