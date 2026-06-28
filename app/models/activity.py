from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Activity:
    name: str
    category: str
    description: str = ""
    cost: float = 0.0
    duration_hours: float = 1.5
    indoor: bool = False
    latitude: float | None = None
    longitude: float | None = None
    distance_m: float | None = None
    source: str = "local"
