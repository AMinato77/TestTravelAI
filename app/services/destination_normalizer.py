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

DESTINATION_MATCH_TERMS = {
    "vienna": ["vienna", "wien"],
    "rome": ["rome", "roma"],
    "milan": ["milan", "milano", "mailand"],
    "munich": ["munich", "muenchen", "munchen", "münchen"],
    "lisbon": ["lisbon", "lissabon", "lisboa"],
    "tokyo": ["tokyo", "tokio"],
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


def destination_match_terms(destination: str) -> list[str]:
    """Return local and canonical city names that may appear in provider data."""
    normalized = normalize_destination(destination)
    if not normalized:
        return []
    terms = DESTINATION_MATCH_TERMS.get(normalized.lower(), [normalized.lower()])
    result: list[str] = []
    seen: set[str] = set()
    for term in [normalized, *terms]:
        cleaned = _clean_destination(term).lower()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def destination_matches_text(destination: str, text: str) -> bool:
    haystack = str(text or "").lower()
    return any(term in haystack for term in destination_match_terms(destination))


def _clean_destination(destination: str) -> str:
    value = re.sub(r"\s+", " ", str(destination or "")).strip()
    value = value.strip(".,;:!?")
    return value
