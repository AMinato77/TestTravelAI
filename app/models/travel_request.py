from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TravelRequest:
    destination: str = ""
    destination_scope: str = "city"
    needs_destination_recommendation: bool = False
    destination_reasoning: str = ""
    duration_days: int = 3
    budget: float = 600
    interests: list[str] = field(default_factory=list)
    must_have: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    travel_style: str = "balanced"
