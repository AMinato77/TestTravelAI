from __future__ import annotations

from app.models.preference_source import PreferenceSource
from app.models.user_profile import UserProfile
from app.tools.openai_runtime import demo_fallback_enabled, generate_json


INTEREST_KEYWORDS = {
    "street food": ["street food", "food market", "tapas", "ramen", "sushi", "essen"],
    "food": ["food", "restaurant", "cafe", "bar", "kueche", "küche"],
    "gaming": ["gaming", "game", "games", "nintendo", "playstation", "arcade"],
    "anime": ["anime", "manga", "japan", "tokio", "tokyo"],
    "local culture": ["local culture", "lokal", "local", "neighborhood", "viertel"],
    "nightlife": ["nightlife", "club", "bar", "party"],
    "nature": ["nature", "park", "hiking", "beach", "strand"],
    "history": ["history", "museum", "historic", "geschichte"],
}

AVOID_KEYWORDS = {
    "tourist traps": ["tourist trap", "touristenfalle", "zu touristisch", "touristy"],
    "luxury travel": ["luxury", "teuer", "overpriced", "zu teuer"],
    "stressful schedules": ["stressig", "too much", "zu viel", "hektisch"],
    "crowded places": ["crowded", "voll", "ueberfuellt", "überfüllt"],
}

NON_INTEREST_TERMS = {
    "budget",
    "budget travel",
    "cheap",
    "cheap travel",
    "low budget",
    "medium budget",
    "high budget",
}


def extract_preferences(
    manual_interests: list[str],
    travel_style: str,
    budget_preference: str,
    preference_sources: list[PreferenceSource] | None = None,
) -> UserProfile:
    """Extract a profile from uploads, notes, ratings, feedback, and manual input."""
    preference_sources = preference_sources or []
    if demo_fallback_enabled():
        return _extract_demo_preferences(manual_interests, travel_style, budget_preference, preference_sources)

    payload = {
        "manual_interests": manual_interests,
        "travel_style": travel_style,
        "budget_preference": budget_preference,
        "sources": [
            {
                "source_type": source.source_type,
                "name": source.name,
                "text": source.text[:6000],
            }
            for source in preference_sources
        ],
    }
    data = generate_json(
        system_prompt=(
            "You are the Preference Agent for an adaptive travel planner. "
            "Analyze chat exports, personal notes, travel ratings, feedback, "
            "and manual form inputs. Return strict JSON with keys: interests, "
            "budget_preference, travel_style, avoid, source_notes. Keep values concise."
        ),
        payload=payload,
        model_env="OPENAI_PREFERENCE_MODEL",
    )
    return UserProfile(
        interests=_clean_interests(data.get("interests", manual_interests)),
        budget_preference=data.get("budget_preference", budget_preference),
        travel_style=data.get("travel_style", travel_style),
        avoid=_as_list(data.get("avoid")),
        source_notes=_as_list(data.get("source_notes")) or ["Preferences extracted from user memory and current inputs."],
        uploaded_sources=[],
    )


def _extract_demo_preferences(
    manual_interests: list[str],
    travel_style: str,
    budget_preference: str,
    preference_sources: list[PreferenceSource],
) -> UserProfile:
    combined_text = " ".join(source.text.lower() for source in preference_sources)
    interests = {
        interest.strip().lower()
        for interest in manual_interests
        if interest.strip().lower() and interest.strip().lower() not in NON_INTEREST_TERMS
    }
    avoid: set[str] = set()

    for interest, keywords in INTEREST_KEYWORDS.items():
        if any(keyword in combined_text for keyword in keywords):
            interests.add(interest)

    for avoid_item, keywords in AVOID_KEYWORDS.items():
        if any(keyword in combined_text for keyword in keywords):
            avoid.add(avoid_item)

    inferred_style = travel_style
    if any(word in combined_text for word in ["relaxed", "entspannt", "nicht stressig", "slow travel"]):
        inferred_style = "relaxed"
    elif any(word in combined_text for word in ["adventure", "abenteuer", "hiking"]):
        inferred_style = "adventure"

    inferred_budget = budget_preference
    if any(word in combined_text for word in ["budget", "guenstig", "günstig", "cheap", "low cost"]):
        inferred_budget = "low"
    elif any(word in combined_text for word in ["luxury", "teuer", "premium"]):
        inferred_budget = "high"

    return UserProfile(
        interests=sorted(interests),
        budget_preference=inferred_budget,
        travel_style=inferred_style,
        avoid=sorted(avoid),
        source_notes=[
            f"Analyzed {len(preference_sources)} uploaded preference source(s) in demo mode."
        ],
        uploaded_sources=[],
    )


def _clean_interests(values) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in _as_list(values):
        normalized = str(value).strip().lower()
        if not normalized or normalized in NON_INTEREST_TERMS or normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    return cleaned


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]
