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
    "athen": "Athens",
    "athens": "Athens",
    "athina": "Athens",
}

DESTINATION_MATCH_TERMS = {
    "vienna": ["vienna", "wien"],
    "rome": ["rome", "roma"],
    "milan": ["milan", "milano", "mailand"],
    "munich": ["munich", "muenchen", "munchen", "münchen"],
    "lisbon": ["lisbon", "lissabon", "lisboa"],
    "tokyo": ["tokyo", "tokio"],
    "athens": ["athens", "athina", "athen"],
}


def normalize_destination(destination: str) -> str:
    """Return a canonical destination name for memory and external APIs."""
    cleaned = _clean_destination(destination)
    if not cleaned:
        return ""
    if "," in cleaned:
        return ", ".join(part.strip().title() for part in cleaned.split(",") if part.strip())
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
    if "," in normalized:
        parts = [part.strip().lower() for part in normalized.split(",") if part.strip()]
        city_aliases = DESTINATION_MATCH_TERMS.get(parts[0], [parts[0]]) if parts else []
        result: list[str] = []
        seen: set[str] = set()
        for term in [normalized.lower(), *city_aliases, *parts]:
            cleaned = _clean_destination(term).lower()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)
        return result
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
    normalized = normalize_destination(destination)
    if "," in normalized:
        parts = [part.strip().lower() for part in normalized.split(",", 1)]
        city = parts[0] if parts else ""
        country = parts[1] if len(parts) > 1 else ""
        if city and country:
            city_terms = DESTINATION_MATCH_TERMS.get(city, [city])
            return any(term in haystack for term in city_terms) and country in haystack
    return any(term in haystack for term in destination_match_terms(destination))


def _clean_destination(destination: str) -> str:
    value = re.sub(r"\s+", " ", str(destination or "")).strip()
    value = value.strip(".,;:!?")
    return value
