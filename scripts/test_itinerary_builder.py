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
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 700
    interests = ["food", "culture", "local spots"]

    from app.models.user_profile import UserProfile
    from app.services.itinerary_builder import build_rule_based_itinerary
    from app.tools.places_tool import search_places
    from app.tools.weather_tool import get_weather

    profile = UserProfile(interests=interests, travel_style="relaxed", budget_preference="medium")
    weather = get_weather(destination, days=days)
    activities = search_places(destination, interests, limit=16)
    itinerary = build_rule_based_itinerary(destination, days, budget, activities, weather, profile)

    print(json.dumps(asdict(itinerary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
