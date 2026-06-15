from __future__ import annotations

from app.models.preference_source import PreferenceSource
from app.models.user_profile import UserProfile
from app.services.interest_taxonomy import normalize_interests, taxonomy_payload
from app.tools.openai_runtime import demo_fallback_enabled, generate_json


INTEREST_KEYWORDS = {
    "street food": ["street food", "food market", "tapas", "ramen", "sushi", "essen"],
    "food": ["food", "restaurant", "cafe", "bar", "kueche", "küche"],
    "gaming": ["gaming", "game", "games", "nintendo", "playstation", "arcade"],
    "anime": ["anime", "manga", "japan", "tokio", "tokyo"],
    "local spots": ["local culture", "lokal", "local", "neighborhood", "viertel"],
    "sport": ["sport", "football", "fussball", "fußball", "stadium", "stadion", "motorsport", "formula 1", "formel 1"],
    "shopping": ["shopping", "shop", "shops", "store", "stores", "markt", "market"],
    "technology": ["technology", "technik", "technologie", "electronics", "computer"],
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
            "budget_preference, travel_style, avoid, source_notes. "
            "Interests must use only the provided allowed_interests categories. "
            "Use interest_descriptions to semantically map concrete wishes to broad categories. "
            "Only include shopping when the source explicitly mentions shopping, shops, stores, markets, buying, or browsing retail. "
            "Do not infer shopping merely because anime, manga, comics, gaming, fashion, or a fandom is mentioned. "
            "Do not create hard required-place constraints and do not invent venue names. "
            "Keep values concise."
        ),
        payload={**payload, **taxonomy_payload()},
        model_env="OPENAI_PREFERENCE_MODEL",
    )
    structured_email_interests = _structured_email_interests(preference_sources)
    if structured_email_interests:
        interests = _merge_unique(manual_interests, structured_email_interests)
    else:
        interests = _clean_interests(data.get("interests", manual_interests))
    return UserProfile(
        interests=interests,
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
    structured_email_interests = _structured_email_interests(preference_sources)
    interests = {
        interest
        for interest in normalize_interests([*manual_interests, *structured_email_interests])
        if interest and interest not in NON_INTEREST_TERMS
    }
    avoid: set[str] = set()

    if not structured_email_interests:
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
    normalized_values = normalize_interests(_as_list(values))
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in normalized_values:
        normalized = str(value).strip().lower()
        if not normalized or normalized in NON_INTEREST_TERMS or normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    return cleaned


def _structured_email_interests(sources: list[PreferenceSource]) -> list[str]:
    values: list[str] = []
    for source in sources:
        if source.source_type != "email_newsletter":
            continue
        for raw_line in source.text.splitlines():
            line = raw_line.strip()
            if not line.lower().startswith("interests:"):
                continue
            raw_values = line.split(":", 1)[1]
            if raw_values.strip().lower() == "none":
                continue
            values.extend(value.strip() for value in raw_values.split(","))
    return _clean_interests(values)


def _merge_unique(*groups: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in _clean_interests(group):
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
    return result


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]
