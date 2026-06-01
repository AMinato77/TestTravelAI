from __future__ import annotations

from dataclasses import dataclass, field

from app.models.activity import Activity


@dataclass(slots=True)
class ItineraryDay:
    day: int
    activities: list[Activity] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(activity.cost for activity in self.activities)

    @property
    def total_duration_hours(self) -> float:
        return sum(activity.duration_hours for activity in self.activities)


@dataclass(slots=True)
class Itinerary:
    destination: str
    days: list[ItineraryDay]
    currency: str = "EUR"

    @property
    def total_cost(self) -> float:
        return sum(day.total_cost for day in self.days)


@dataclass(slots=True)
class ValidationIssue:
    severity: str
    message: str
    day: int | None = None
    issue_type: str = "general"
    activity: str | None = None


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
