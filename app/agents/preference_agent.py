from __future__ import annotations

from app.models.preference_source import PreferenceSource
from app.models.travel_request import TravelRequest
from app.models.user_profile import UserProfile
from app.tools.openai_runtime import demo_fallback_enabled, generate_json


NULL_TERMS = {"", "none", "unknown", "null", "n/a", "-", "keine", "kein"}


def extract_preferences(
    request: TravelRequest,
    budget_preference: str,
    preference_sources: list[PreferenceSource] | None = None,
) -> UserProfile:
    """Summarize natural preference memory instead of keyword-mapping to categories."""
    preference_sources = preference_sources or []
    if not demo_fallback_enabled():
        try:
            data = generate_json(
                system_prompt=(
                    "You are the Preference Memory Agent for an adaptive travel planner. "
                    "Read current request context and retrieved memory. Return strict JSON with keys: "
                    "preference_notes, avoid, interest_tags, travel_style, budget_preference, source_notes. "
                    "preference_notes must be concise natural-language facts useful for query planning "
                    "and activity evaluation, not just category names. "
                    "interest_tags are optional broad UI metadata only. Do not invent venues."
                ),
                payload={
                    "request": {
                        "destination": request.destination,
                        "must_have": request.must_have,
                        "avoid": request.avoid,
                        "query_hints": request.query_hints,
                        "travel_style": request.travel_style,
                    },
                    "budget_preference": budget_preference,
                    "sources": [
                        {"source_type": source.source_type, "name": source.name, "text": source.text[:4000]}
                        for source in preference_sources[:12]
                    ],
                },
                model_env="OPENAI_PREFERENCE_MODEL",
            )
            return UserProfile(
                interest_tags=_clean_values(data.get("interest_tags")),
                preference_notes=_clean_values(data.get("preference_notes")),
                budget_preference=_clean_choice(data.get("budget_preference"), {"low", "medium", "high"}, budget_preference),
                travel_style=_clean_choice(data.get("travel_style"), {"balanced", "relaxed", "adventure", "luxury", "budget"}, request.travel_style),
                avoid=_merge_unique(request.avoid, _clean_values(data.get("avoid"))),
                source_notes=_clean_values(data.get("source_notes")) or ["Preference memory summarized from current request and retrieved context."],
            )
        except Exception:
            pass

    return _fallback_preferences(request, budget_preference, preference_sources)


def _fallback_preferences(
    request: TravelRequest,
    budget_preference: str,
    preference_sources: list[PreferenceSource],
) -> UserProfile:
    notes = _merge_unique(
        request.must_have,
        request.query_hints,
        [
            source.text[:350]
            for source in preference_sources
            if source.text.strip() and source.source_type != "profile_snapshot"
        ],
    )
    return UserProfile(
        interest_tags=request.interest_tags,
        preference_notes=notes,
        budget_preference=budget_preference,
        travel_style=request.travel_style,
        avoid=request.avoid,
        source_notes=[f"Summarized {len(preference_sources)} memory source(s) with deterministic fallback."],
    )


def _clean_values(value) -> list[str]:
    return _merge_unique(_as_list(value))


def _merge_unique(*groups: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in _as_list(group):
            cleaned = " ".join(str(value).strip().split())
            key = cleaned.lower()
            if key in NULL_TERMS or key in seen:
                continue
            seen.add(key)
            result.append(cleaned)
    return result


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]


def _clean_choice(value, allowed: set[str], fallback: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else fallback
