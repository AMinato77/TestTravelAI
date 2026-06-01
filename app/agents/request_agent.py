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
    "budget",
    "adventure",
]

KNOWN_STYLES = ["relaxed", "adventure", "luxury", "budget", "balanced"]


def parse_travel_request(text: str, fallback: TravelRequest) -> TravelRequest:
    """Parse a natural language travel request into structured fields."""
    if not text.strip():
        return fallback

    if not demo_fallback_enabled():
        data = generate_json(
            system_prompt=(
                "Extract a travel request as strict JSON. "
                "Use keys: destination, duration_days, budget, interests, travel_style. "
                "travel_style must be one of balanced, relaxed, adventure, luxury, budget. "
                "If a field is missing, use the provided fallback."
            ),
            payload={
                "request": text,
                "fallback": {
                    "destination": fallback.destination,
                    "duration_days": fallback.duration_days,
                    "budget": fallback.budget,
                    "interests": fallback.interests,
                    "travel_style": fallback.travel_style,
                },
            },
            model_env="OPENAI_REQUEST_MODEL",
        )
        return TravelRequest(
            destination=data.get("destination") or fallback.destination,
            duration_days=int(data.get("duration_days") or fallback.duration_days),
            budget=float(data.get("budget") or fallback.budget),
            interests=data.get("interests") or fallback.interests,
            travel_style=data.get("travel_style") or fallback.travel_style,
        )

    return TravelRequest(
        destination=_parse_destination(text) or fallback.destination,
        duration_days=_parse_days(text) or fallback.duration_days,
        budget=_parse_budget(text) or fallback.budget,
        interests=_parse_interests(text) or fallback.interests,
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


def _parse_style(text: str) -> str | None:
    lower = text.lower()
    if "nicht stressig" in lower or "keinen stressigen" in lower:
        return "relaxed"
    for style in KNOWN_STYLES:
        if style in lower:
            return style
    return None
