from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class UserProfile:
    user_id: str = "demo_user_1"
    interest_tags: list[str] = field(default_factory=list)
    preference_notes: list[str] = field(default_factory=list)
    budget_preference: str = "medium"
    travel_style: str = "balanced"
    avoid: list[str] = field(default_factory=list)
    preferred_day_structure: str = "balanced"
    source_notes: list[str] = field(default_factory=list)
    past_destinations: list[str] = field(default_factory=list)
    feedback_history: list[str] = field(default_factory=list)
    uploaded_sources: list[str] = field(default_factory=list)

    def merged_interest_tags(self, current_tags: list[str]) -> list[str]:
        values = [*self.interest_tags, *current_tags]
        return sorted({value.strip().lower() for value in values if value.strip()})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        return cls(
            user_id=data.get("user_id", "demo_user_1"),
            interest_tags=data.get("interest_tags", []),
            preference_notes=data.get("preference_notes", []),
            budget_preference=data.get("budget_preference", "medium"),
            travel_style=data.get("travel_style", "balanced"),
            avoid=data.get("avoid", []),
            preferred_day_structure=data.get("preferred_day_structure", "balanced"),
            source_notes=data.get("source_notes", []),
            past_destinations=data.get("past_destinations", []),
            feedback_history=data.get("feedback_history", []),
            uploaded_sources=data.get("uploaded_sources", []),
        )
