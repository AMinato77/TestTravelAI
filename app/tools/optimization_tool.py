from __future__ import annotations

from app.models.activity import Activity
from app.models.itinerary import Itinerary
from app.models.user_profile import UserProfile
from app.services.itinerary_optimizer import optimize_itinerary_rules
from app.services.itinerary_validator import validate_itinerary_rules


def optimize_itinerary(
    itinerary: Itinerary,
    alternatives: list[Activity],
    budget: float,
    weather: dict,
    profile: UserProfile | None = None,
    constraints: dict | None = None,
) -> Itinerary:
    validation = validate_itinerary_rules(itinerary, budget, weather, profile, constraints=constraints)
    return optimize_itinerary_rules(itinerary, validation, alternatives, budget, weather, profile, constraints=constraints)
