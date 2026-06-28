from __future__ import annotations

import json
import time

from app.models.user_profile import UserProfile
from app.rag.chroma_db import get_or_create_collection
from app.rag.embeddings import embed_texts
from app.rag.memory_retrieval import COLLECTION_NAME, list_memory_user_ids
from app.services.destination_normalizer import normalize_destination, normalize_destinations


NON_INTEREST_TERMS = {
    "budget",
    "budget travel",
    "cheap",
    "cheap travel",
    "low budget",
    "medium budget",
    "high budget",
    "none",
    "unknown",
    "null",
    "n/a",
    "-",
}

NULL_TERMS = {"none", "unknown", "null", "n/a", "-", "keine", "kein"}


def load_user_profile(user_id: str) -> UserProfile:
    """Load the latest user profile snapshot from ChromaDB."""
    safe_user_id = _safe_user_id(user_id)
    collection = get_or_create_collection(COLLECTION_NAME)
    result = collection.get(
        ids=[_profile_snapshot_id(safe_user_id)],
        include=["metadatas"],
    )
    metadatas = result.get("metadatas") or []
    if not metadatas:
        return UserProfile(user_id=safe_user_id)
    return _profile_from_metadata(metadatas[0], safe_user_id)


def list_user_ids() -> list[str]:
    """Return all user ids known from ChromaDB memory."""
    return list_memory_user_ids()


def create_user_profile(user_id: str) -> UserProfile:
    """Create an empty embedded user profile snapshot if none exists."""
    profile = load_user_profile(user_id)
    save_user_profile(profile)
    return profile


def save_user_profile(profile: UserProfile) -> None:
    """Persist a profile snapshot as an embedded ChromaDB memory."""
    safe_user_id = _safe_user_id(profile.user_id)
    profile.user_id = safe_user_id
    profile.interest_tags = _clean_interest_tags(_as_list(profile.interest_tags))
    profile.preference_notes = _clean_source_notes(profile.preference_notes)
    profile.avoid = _clean_plain_values(profile.avoid)
    profile.source_notes = _clean_source_notes(profile.source_notes)
    profile.past_destinations = normalize_destinations(_as_list(profile.past_destinations))
    profile.feedback_history = _as_list(profile.feedback_history)[-20:]
    profile.uploaded_sources = _merge_unique(_as_list(profile.uploaded_sources))
    document = _profile_document(profile)
    metadata = {
        "user_id": safe_user_id,
        "memory_kind": "profile_snapshot",
        "source_type": "profile_snapshot",
        "source_name": "current_user_profile",
        "created_at": time.time(),
        "interest_tags_json": json.dumps(profile.interest_tags, ensure_ascii=True),
        "preference_notes_json": json.dumps(profile.preference_notes, ensure_ascii=True),
        "budget_preference": profile.budget_preference,
        "travel_style": profile.travel_style,
        "avoid_json": json.dumps(profile.avoid, ensure_ascii=True),
        "preferred_day_structure": profile.preferred_day_structure,
        "source_notes_json": json.dumps(profile.source_notes, ensure_ascii=True),
        "past_destinations_json": json.dumps(profile.past_destinations, ensure_ascii=True),
        "feedback_history_json": json.dumps(profile.feedback_history[-20:], ensure_ascii=True),
        "uploaded_sources_json": json.dumps(profile.uploaded_sources, ensure_ascii=True),
    }

    collection = get_or_create_collection(COLLECTION_NAME)
    collection.upsert(
        ids=[_profile_snapshot_id(safe_user_id)],
        documents=[document],
        embeddings=embed_texts([document]),
        metadatas=[metadata],
    )


def update_user_profile(
    existing: UserProfile,
    extracted: UserProfile,
    destination: str,
    current_interest_tags: list[str],
    manual_avoid: list[str] | None = None,
    feedback: str | None = None,
    uploaded_sources: list[str] | None = None,
    replace_existing_tags: bool = False,
) -> UserProfile:
    if replace_existing_tags:
        interest_tags = _merge_unique(extracted.interest_tags)
    elif _as_list(current_interest_tags):
        interest_tags = _merge_unique(extracted.interest_tags, current_interest_tags)
    else:
        interest_tags = _merge_unique(existing.interest_tags, extracted.interest_tags)
    avoid = _merge_unique(existing.avoid, extracted.avoid, manual_avoid or [])
    past_destinations = normalize_destinations([*existing.past_destinations, normalize_destination(destination)])
    feedback_history = existing.feedback_history[:]
    if feedback and feedback.strip():
        feedback_history.append(feedback.strip())
    uploaded_source_names = _merge_unique(existing.uploaded_sources, extracted.uploaded_sources, uploaded_sources or [])

    profile = UserProfile(
        user_id=existing.user_id,
        interest_tags=interest_tags,
        preference_notes=_merge_unique(existing.preference_notes, extracted.preference_notes),
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


def _profile_snapshot_id(user_id: str) -> str:
    return f"{user_id}_profile_snapshot"


def _safe_user_id(user_id: str) -> str:
    return "".join(char for char in user_id if char.isalnum() or char in ("-", "_")).strip() or "demo_user_1"


def _profile_from_metadata(metadata: dict, fallback_user_id: str) -> UserProfile:
    return UserProfile(
        user_id=str(metadata.get("user_id") or fallback_user_id),
        interest_tags=_clean_interest_tags(_json_list(metadata.get("interest_tags_json"))),
        preference_notes=_clean_source_notes(_json_list(metadata.get("preference_notes_json"))),
        budget_preference=str(metadata.get("budget_preference") or "medium"),
        travel_style=str(metadata.get("travel_style") or "balanced"),
        avoid=_json_list(metadata.get("avoid_json")),
        preferred_day_structure=str(metadata.get("preferred_day_structure") or "balanced"),
        source_notes=_clean_source_notes(_json_list(metadata.get("source_notes_json"))),
        past_destinations=normalize_destinations(_json_list(metadata.get("past_destinations_json"))),
        feedback_history=_json_list(metadata.get("feedback_history_json")),
        uploaded_sources=_json_list(metadata.get("uploaded_sources_json")),
    )


def _profile_document(profile: UserProfile) -> str:
    lines = [
        f"User profile: {profile.user_id}",
        f"Interest tags: {', '.join(profile.interest_tags) or 'none'}",
        f"Preference notes: {' | '.join(profile.preference_notes[-8:]) or 'none'}",
        f"Avoid: {', '.join(profile.avoid) or 'none'}",
        f"Travel style: {profile.travel_style}",
        f"Budget preference: {profile.budget_preference}",
        f"Preferred day structure: {profile.preferred_day_structure}",
        f"Past destinations: {', '.join(profile.past_destinations) or 'none'}",
        f"Feedback history: {' | '.join(profile.feedback_history[-5:]) or 'none'}",
    ]
    if profile.source_notes:
        lines.append(f"Source notes: {' | '.join(profile.source_notes[:5])}")
    return "\n".join(lines)


def _json_list(value) -> list[str]:
    if not value:
        return []
    try:
        data = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(item) for item in data if str(item).strip()]


def _merge_unique(*groups: list[str]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in _as_list(group):
            normalized = str(value).strip().lower()
            if not normalized or normalized in NON_INTEREST_TERMS or normalized in seen:
                continue
            seen.add(normalized)
            values.append(normalized)
    return values


def _clean_interest_tags(values) -> list[str]:
    return [
        value.strip().lower()
        for value in _as_list(values)
        if value.strip().lower() and value.strip().lower() not in NON_INTEREST_TERMS
    ]


def _clean_plain_values(values) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in _as_list(values):
        normalized = " ".join(str(value).strip().lower().split())
        if not normalized or normalized in NULL_TERMS or normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    return cleaned


def _clean_source_notes(value) -> list[str]:
    values = _as_list(value)
    if not values:
        return []
    single_char_count = sum(1 for item in values if len(item.strip()) == 1)
    if len(values) >= 8 and single_char_count / len(values) > 0.75:
        collapsed = "".join(values).strip()
        values = [collapsed] if collapsed else []

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in values:
        note = " ".join(str(item).split())
        if _is_corrupt_source_note(note):
            continue
        if not note or note.lower() in seen:
            continue
        seen.add(note.lower())
        cleaned.append(note)
    return cleaned


def _is_corrupt_source_note(note: str) -> bool:
    lower = note.lower()
    if not lower:
        return False
    if lower.startswith("profile_snapshot:"):
        return True
    if "past_destinations=" in lower:
        return True
    if "source:" in lower and "profile:" in lower:
        return True
    if "manl_itp" in lower or ";xd" in lower:
        return True
    return False


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]


def _day_structure_from_style(travel_style: str) -> str:
    if travel_style == "relaxed":
        return "max_2_major_activities_per_day"
    if travel_style == "adventure":
        return "active_days_allowed"
    return "balanced"
