from __future__ import annotations

from app.models.itinerary import Itinerary, ValidationResult
from app.models.user_profile import UserProfile
from app.services.itinerary_validator import validate_itinerary_rules


def validate_itinerary(
    itinerary: Itinerary,
    budget: float,
    weather: dict,
    profile: UserProfile | None = None,
) -> ValidationResult:
    return validate_itinerary_rules(itinerary, budget, weather, profile)
