from __future__ import annotations


ALLOWED_INTERESTS = [
    "food",
    "street food",
    "gaming",
    "anime",
    "culture",
    "history",
    "local spots",
    "hidden gems",
    "nightlife",
    "nature",
    "sport",
    "shopping",
    "technology",
    "photography",
    "architecture",
    "luxury",
    "adventure",
]

INTEREST_ALIASES = {
    "essen": "food",
    "restaurant": "food",
    "restaurants": "food",
    "tapas": "food",
    "streetfood": "street food",
    "geschichte": "history",
    "historie": "history",
    "kultur": "culture",
    "lokale spots": "local spots",
    "local culture": "local spots",
    "lokal": "local spots",
    "viertel": "local spots",
    "neighborhood": "local spots",
    "neighbourhood": "local spots",
    "hidden gem": "hidden gems",
    "sport": "sport",
    "football": "sport",
    "soccer": "sport",
    "fussball": "sport",
    "stadium": "sport",
    "stadion": "sport",
    "motorsport": "sport",
    "formula 1": "sport",
    "formel 1": "sport",
    "f1": "sport",
    "natur": "nature",
    "technik": "technology",
    "technologie": "technology",
    "architektur": "architecture",
    "fotografie": "photography",
}

INTEREST_DESCRIPTIONS = {
    "food": "restaurants, cafes, local cuisine, tastings, culinary experiences",
    "street food": "food markets, informal food vendors, street-food style eating",
    "gaming": "arcades, esports, board-game cafes, video-game stores, gaming venues",
    "anime": "anime, manga, comics, pop-culture stores, Japanese fandom places",
    "culture": "art, museums, galleries, cultural sites, performances",
    "history": "historic sites, monuments, heritage, old towns",
    "local spots": "neighborhoods, markets, plazas, everyday local areas",
    "hidden gems": "less obvious local places, small neighborhoods, non-mainstream spots",
    "nightlife": "bars, clubs, live music, late evening venues",
    "nature": "parks, beaches, viewpoints, gardens, outdoor walks",
    "sport": "stadiums, arenas, sport events, motorsport, football, local sport culture",
    "shopping": "shops, malls, markets, shopping streets, specialty stores",
    "technology": "electronics, tech stores, science and technology experiences",
    "photography": "viewpoints, scenic places, strong visual/photo spots",
    "architecture": "notable buildings, architecture routes, design landmarks",
    "luxury": "premium experiences, upscale restaurants, high-end shopping",
    "adventure": "active experiences, outdoor activities, high-energy plans",
}


def normalize_interest(value: str) -> str | None:
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    mapped = INTEREST_ALIASES.get(normalized, normalized)
    return mapped if mapped in ALLOWED_INTERESTS else None


def normalize_interests(values) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        normalized = normalize_interest(str(value))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def taxonomy_payload() -> dict:
    return {
        "allowed_interests": ALLOWED_INTERESTS,
        "interest_descriptions": INTEREST_DESCRIPTIONS,
    }
