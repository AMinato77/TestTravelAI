from __future__ import annotations

import json
from pathlib import Path

from app.models.user_profile import UserProfile


MEMORY_DIR = Path("data/user_profiles")
NON_INTEREST_TERMS = {
    "budget",
    "budget travel",
    "cheap",
    "cheap travel",
    "low budget",
    "medium budget",
    "high budget",
}


def load_user_profile(user_id: str) -> UserProfile:
    path = _profile_path(user_id)
    if not path.exists():
        return UserProfile(user_id=user_id)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return _profile_from_dict(data, user_id)


def list_user_ids() -> list[str]:
    """Return all saved user ids from data/user_profiles."""
    if not MEMORY_DIR.exists():
        return []

    user_ids: list[str] = []
    for path in MEMORY_DIR.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            user_ids.append(data.get("user_id") or path.stem)
        except (OSError, json.JSONDecodeError):
            user_ids.append(path.stem)
    return sorted(set(user_ids))


def create_user_profile(user_id: str) -> UserProfile:
    """Create a new empty user profile if it does not already exist."""
    profile = load_user_profile(user_id)
    save_user_profile(profile)
    return profile


def save_user_profile(profile: UserProfile) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    with _profile_path(profile.user_id).open("w", encoding="utf-8") as file:
        json.dump(_profile_to_dict(profile), file, indent=2, ensure_ascii=True)


def update_user_profile(
    existing: UserProfile,
    extracted: UserProfile,
    destination: str,
    manual_interests: list[str],
    manual_avoid: list[str] | None = None,
    feedback: str | None = None,
    uploaded_sources: list[str] | None = None,
) -> UserProfile:
    interests = _merge_unique(existing.interests, extracted.interests, manual_interests)
    avoid = _merge_unique(existing.avoid, extracted.avoid, manual_avoid or [])
    past_destinations = _merge_unique(existing.past_destinations, [destination])
    feedback_history = existing.feedback_history[:]
    if feedback and feedback.strip():
        feedback_history.append(feedback.strip())
    uploaded_source_names = _merge_unique(existing.uploaded_sources, extracted.uploaded_sources, uploaded_sources or [])

    profile = UserProfile(
        user_id=existing.user_id,
        interests=interests,
        budget_preference=extracted.budget_preference or existing.budget_preference,
        travel_style=extracted.travel_style or existing.travel_style,
        avoid=avoid,
        preferred_day_structure=_day_structure_from_style(extracted.travel_style or existing.travel_style),
        source_notes=_merge_unique(existing.source_notes, extracted.source_notes),
        past_destinations=past_destinations,
        feedback_history=feedback_history[-20:],
        uploaded_sources=uploaded_source_names,
    )
    save_user_profile(profile)
    return profile


def _profile_path(user_id: str) -> Path:
    safe_user_id = "".join(char for char in user_id if char.isalnum() or char in ("-", "_")).strip()
    return MEMORY_DIR / f"{safe_user_id or 'demo_user_1'}.json"


def _profile_from_dict(data: dict, fallback_user_id: str) -> UserProfile:
    return UserProfile(
        user_id=data.get("user_id", fallback_user_id),
        interests=_clean_interests(data.get("interests", [])),
        budget_preference=data.get("budget_preference", "medium"),
        travel_style=data.get("travel_style", "balanced"),
        avoid=data.get("avoid", []),
        preferred_day_structure=data.get("preferred_day_structure", "balanced"),
        source_notes=data.get("source_notes", []),
        past_destinations=data.get("past_destinations", []),
        feedback_history=data.get("feedback_history", []),
        uploaded_sources=data.get("uploaded_sources", []),
    )


def _profile_to_dict(profile: UserProfile) -> dict:
    return {
        "user_id": profile.user_id,
        "interests": profile.interests,
        "budget_preference": profile.budget_preference,
        "travel_style": profile.travel_style,
        "avoid": profile.avoid,
        "preferred_day_structure": profile.preferred_day_structure,
        "source_notes": profile.source_notes,
        "past_destinations": profile.past_destinations,
        "feedback_history": profile.feedback_history,
        "uploaded_sources": profile.uploaded_sources,
    }


def _merge_unique(*groups: list[str]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            normalized = value.strip().lower()
            if not normalized or normalized in NON_INTEREST_TERMS or normalized in seen:
                continue
            seen.add(normalized)
            values.append(normalized)
    return values


def _clean_interests(values: list[str]) -> list[str]:
    return [
        value.strip().lower()
        for value in values
        if value.strip().lower() and value.strip().lower() not in NON_INTEREST_TERMS
    ]


def _day_structure_from_style(travel_style: str) -> str:
    if travel_style == "relaxed":
        return "max_2_major_activities_per_day"
    if travel_style == "adventure":
        return "active_days_allowed"
    return "balanced"
