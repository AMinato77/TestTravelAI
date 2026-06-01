from __future__ import annotations

import re

from app.models.travel_request import TravelRequest
from app.tools.openai_runtime import demo_fallback_enabled, generate_json


KNOWN_INTERESTS = [
    "food",
    "gaming",
    "anime",
    "culture",
    "local spots",
    "nightlife",
    "nature",
    "luxury",
    "adventure",
]

KNOWN_STYLES = ["relaxed", "adventure", "luxury", "budget", "balanced"]

INTEREST_ALIASES = {
    "essen": "food",
    "geschichte": "history",
    "historie": "history",
    "kultur": "culture",
    "sport": "sport",
    "natur": "nature",
    "lokale spots": "local spots",
}


def parse_travel_request(text: str, fallback: TravelRequest) -> TravelRequest:
    """Parse a natural language travel request into structured fields."""
    if not text.strip():
        return fallback

    if not demo_fallback_enabled():
        data = generate_json(
            system_prompt=(
                "Extract a travel request as strict JSON. "
                "Use keys: destination, duration_days, budget, interests, avoid, travel_style. "
                "Put disliked things into avoid, not into interests. "
                "German phrases like 'hasst Essen', 'mag kein Essen', 'keine Restaurants', "
                "or 'ohne Food' mean avoid includes 'food' and 'restaurants'. "
                "travel_style must be one of balanced, relaxed, adventure, luxury, budget. "
                "German phrases like 'keinen stressigen Plan', 'nicht stressig', "
                "or 'entspannt' mean travel_style='relaxed'. "
                "If a field is missing, use the provided fallback."
            ),
            payload={
                "request": text,
                "fallback": {
                    "destination": fallback.destination,
                    "duration_days": fallback.duration_days,
                    "budget": fallback.budget,
                    "interests": fallback.interests,
                    "avoid": fallback.avoid,
                    "travel_style": fallback.travel_style,
                },
            },
            model_env="OPENAI_REQUEST_MODEL",
        )
        return TravelRequest(
            destination=data.get("destination") or fallback.destination,
            duration_days=int(data.get("duration_days") or fallback.duration_days),
            budget=float(data.get("budget") or fallback.budget),
            interests=_remove_avoided_interests(
                _normalize_interests(data.get("interests") or fallback.interests),
                data.get("avoid") or [],
            ),
            avoid=_parse_avoid(text) or data.get("avoid") or fallback.avoid,
            travel_style=_parse_style(text) or data.get("travel_style") or fallback.travel_style,
        )

    avoid = _parse_avoid(text) or fallback.avoid
    return TravelRequest(
        destination=_parse_destination(text) or fallback.destination,
        duration_days=_parse_days(text) or fallback.duration_days,
        budget=_parse_budget(text) or fallback.budget,
        interests=_remove_avoided_interests(_normalize_interests(_parse_interests(text) or fallback.interests), avoid),
        avoid=avoid,
        travel_style=_parse_style(text) or fallback.travel_style,
    )


def _parse_destination(text: str) -> str | None:
    patterns = [
        r"nach\s+([A-Z][A-Za-z\- ]+?)(?:,| mit| fuer| fur| budget|$)",
        r"in\s+([A-Z][A-Za-z\- ]+?)(?:,| mit| fuer| fur| budget|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def _parse_days(text: str) -> int | None:
    match = re.search(r"(\d{1,2})\s*(tage|tag|days|day)", text.lower())
    if not match:
        return None
    return max(1, min(int(match.group(1)), 14))


def _parse_budget(text: str) -> float | None:
    match = re.search(r"(\d{2,5})\s*(eur|euro)", text.lower())
    if not match:
        match = re.search(r"budget\s*(?:von|:)?\s*(\d{2,5})", text.lower())
    return float(match.group(1)) if match else None


def _parse_interests(text: str) -> list[str]:
    lower = text.lower()
    return [interest for interest in KNOWN_INTERESTS if interest in lower]


def _parse_avoid(text: str) -> list[str]:
    lower = text.lower()
    avoid: list[str] = []
    if re.search(r"(hasst|hasse|mag kein|mag keine|keine|kein|ohne)\s+(essen|food|restaurants?|cafes?|cafés?)", lower):
        avoid.extend(["food", "restaurants", "cafes"])
    if re.search(r"(hasst|hasse|mag kein|mag keine|keine|kein|ohne)\s+(museen|museum|museums)", lower):
        avoid.extend(["museums"])
    if re.search(r"(hasst|hasse|mag kein|mag keine|keine|kein|ohne)\s+(clubs?|nightlife|party)", lower):
        avoid.extend(["nightlife", "clubs"])
    return sorted(set(avoid))


def _remove_avoided_interests(interests: list[str], avoid: list[str]) -> list[str]:
    avoid_text = " ".join(avoid).lower()
    filtered: list[str] = []
    for interest in interests:
        normalized = str(interest).strip().lower()
        if not normalized:
            continue
        if normalized == "food" and any(term in avoid_text for term in ["food", "restaurant", "cafe"]):
            continue
        if normalized in avoid_text:
            continue
        filtered.append(normalized)
    return sorted(set(filtered))


def _normalize_interests(interests: list[str]) -> list[str]:
    normalized: list[str] = []
    for interest in interests:
        value = str(interest).strip().lower()
        if not value:
            continue
        normalized.append(INTEREST_ALIASES.get(value, value))
    return sorted(set(normalized))


def _parse_style(text: str) -> str | None:
    lower = text.lower()
    if "nicht stressig" in lower or "keinen stressigen" in lower:
        return "relaxed"
    for style in KNOWN_STYLES:
        if style in lower:
            return style
    return None
