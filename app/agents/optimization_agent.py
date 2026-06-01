from __future__ import annotations

from app.models.activity import Activity
from app.models.itinerary import Itinerary
from app.models.user_profile import UserProfile
from app.tools.optimization_tool import optimize_itinerary


def run_optimization(
    itinerary: Itinerary,
    alternatives: list[Activity],
    budget: float,
    weather: dict,
    profile: UserProfile | None = None,
) -> Itinerary:
    return optimize_itinerary(itinerary, alternatives, budget, weather, profile)
