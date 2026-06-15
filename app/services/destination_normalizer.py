from __future__ import annotations

import re


DESTINATION_ALIASES = {
    "wien": "Vienna",
    "vienna": "Vienna",
    "rom": "Rome",
    "rome": "Rome",
    "mailand": "Milan",
    "milan": "Milan",
    "milano": "Milan",
    "muenchen": "Munich",
    "munchen": "Munich",
    "münchen": "Munich",
    "munich": "Munich",
    "lissabon": "Lisbon",
    "lisbon": "Lisbon",
    "paris": "Paris",
    "barcelona": "Barcelona",
    "berlin": "Berlin",
    "hamburg": "Hamburg",
    "tokio": "Tokyo",
    "tokyo": "Tokyo",
    "turin": "Turin",
    "torino": "Turin",
}


def normalize_destination(destination: str) -> str:
    """Return a canonical destination name for memory and external APIs."""
    cleaned = _clean_destination(destination)
    if not cleaned:
        return ""
    key = cleaned.lower()
    return DESTINATION_ALIASES.get(key, cleaned.title())


def normalize_destinations(destinations: list[str]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for destination in destinations:
        normalized = normalize_destination(destination)
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        values.append(normalized)
    return values


def _clean_destination(destination: str) -> str:
    value = re.sub(r"\s+", " ", str(destination or "")).strip()
    value = value.strip(".,;:!?")
    return value
