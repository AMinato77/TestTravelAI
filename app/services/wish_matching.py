from __future__ import annotations

import re

from app.models.activity import Activity


INTENT_KEYWORDS: dict[str, set[str]] = {
    "food": {
        "food", "foods", "eat", "eating", "cuisine", "culinary", "restaurant", "restaurants",
        "dining", "meal", "meals", "streetfood", "street", "seafood", "breakfast", "lunch",
        "dinner", "brunch", "cafe", "cafes", "market", "markets", "essen", "küche", "kueche",
        "restaurant", "restaurants", "gerichte", "speisen",
    },
    "nature": {
        "nature", "natural", "landscape", "landscapes", "scenery", "scenic", "outdoor",
        "park", "parks", "garden", "gardens", "beach", "beaches", "strand", "strände",
        "straende", "forest", "waterfall", "mountain", "lake", "river", "hike", "hiking",
        "natur", "landschaft", "landschaften",
    },
    "culture": {
        "culture", "cultural", "museum", "museums", "gallery", "architecture", "historic",
        "historical", "history", "ancient", "palace", "castle", "temple", "church", "mosque",
        "monument", "landmark", "heritage", "kultur", "museum", "museen", "architektur",
        "historisch", "geschichte", "gebäude", "gebaeude",
    },
    "shopping": {
        "shop", "shops", "shopping", "store", "stores", "mall", "market", "markets",
        "bazaar", "merchandise", "figure", "figures", "collectible", "collectibles",
        "laden", "läden", "laeden", "markt", "märkte", "maerkte",
    },
    "entertainment": {
        "anime", "gaming", "game", "games", "arcade", "movie", "cinema", "theater",
        "theme", "amusement", "show", "concert", "entertainment",
    },
    "sport": {
        "sport", "sports", "stadium", "football", "soccer", "basketball", "baseball",
        "formula", "racing", "fitness", "gym",
    },
    "nightlife": {
        "nightlife", "club", "clubs", "bar", "bars", "cocktail", "party", "rooftop",
    },
}

ACTIVITY_INTENT_HINTS: dict[str, set[str]] = {
    "food": {"food", "restaurant", "cafe", "bakery", "meal", "bar", "nightlife"},
    "nature": {"nature", "park", "garden", "natural_feature", "beach", "scenic_spot"},
    "culture": {"culture", "museum", "art_gallery", "tourist_attraction", "historical_landmark", "historical_place"},
    "shopping": {"shopping", "shopping_mall", "market", "store", "book_store", "toy_store"},
    "entertainment": {"entertainment", "movie_theater", "amusement_center", "casino"},
    "sport": {"sport", "stadium", "gym", "sports", "fitness"},
    "nightlife": {"nightlife", "bar", "night_club"},
}

STOP_WORDS = {
    "the", "and", "for", "with", "from", "und", "oder", "mit", "von", "fuer", "für",
    "eine", "einen", "einem", "der", "die", "das", "zu", "in", "im", "am", "an",
    "best", "top", "good", "great", "nice", "beste", "besten", "gute", "guten",
    "places", "place", "spots", "spot", "areas", "area", "things", "thing",
    "experience", "experiences", "erlebnis", "erlebnisse", "erleben", "visit",
    "besuch", "besuchen", "sehen", "discover", "explore", "trip", "travel", "tour",
    "city", "country", "local", "generally", "generell", "möglichkeiten", "moeglichkeiten",
}


def infer_intents(text: str) -> set[str]:
    tokens = set(content_tokens(text))
    intents: set[str] = set()
    for intent, keywords in INTENT_KEYWORDS.items():
        if tokens & keywords:
            intents.add(intent)
    return intents


def activity_intents(activity: Activity) -> set[str]:
    text = activity_text(activity)
    intents = infer_intents(text)
    category = str(activity.category or "").lower()
    for intent, hints in ACTIVITY_INTENT_HINTS.items():
        if category in hints:
            intents.add(intent)
    types = description_field(activity.description, "Types").lower()
    type_tokens = set(re.split(r"[\s,_-]+", types))
    for intent, hints in ACTIVITY_INTENT_HINTS.items():
        if type_tokens & hints:
            intents.add(intent)
    return intents


def query_matches_wish(query: str, reason: str, wish: str, destination: str = "") -> bool:
    text = f"{query} {reason}"
    if not intents_compatible(infer_intents(text), infer_intents(wish)):
        return False
    return token_overlap_score(text, wish, destination) >= 0.5


def activity_covers_wish(activity: Activity, wish: str) -> bool:
    wish_intents = infer_intents(wish)
    if not intents_compatible(activity_intents(activity), wish_intents):
        return False
    if not passes_specific_wish_gate(activity, wish):
        return False
    if matched_must_have_covers(activity.description, wish):
        return True
    return token_overlap_score(activity_text(activity), wish) >= 0.5


def activity_wish_score(activity: Activity, wish: str) -> float:
    if not intents_compatible(activity_intents(activity), infer_intents(wish)):
        return 0.0
    if not passes_specific_wish_gate(activity, wish):
        return 0.0
    score = token_overlap_score(activity_text(activity), wish)
    if matched_must_have_covers(activity.description, wish):
        score = max(score, 1.0)
    if activity_intents(activity) & infer_intents(wish):
        score += 0.35
    return score


def passes_specific_wish_gate(activity: Activity, wish: str) -> bool:
    wish_tokens = set(content_tokens(wish))
    wish_intents = infer_intents(wish)
    source_intents = activity_intents(activity)
    strict_intents = {"food", "nature", "shopping", "sport", "nightlife"}
    if wish_intents & strict_intents and not (source_intents & wish_intents):
        return False
    text = activity_text(activity)
    types = description_field(activity.description, "Types").lower()
    searchable = f"{text} {types}"
    if "architecture" in wish_tokens or "architectural" in wish_tokens or "architektur" in wish_tokens:
        searchable_tokens = set(content_tokens(searchable))
        architecture_markers = {
            "architecture", "architectural", "architektur", "building", "buildings", "landmark",
            "historic", "historical", "heritage", "monument", "palace", "castle", "tower",
            "tour", "tours", "guided", "walking", "historical_landmark", "historical_place",
            "tour_agency", "travel_agency",
        }
        return bool(searchable_tokens & architecture_markers) or any(marker in searchable for marker in {"historical_landmark", "historical_place", "tour_agency", "travel_agency"})
    if "tour" in wish_tokens or "tours" in wish_tokens:
        searchable_tokens = set(content_tokens(searchable))
        tour_markers = {"tour", "tours", "guided", "walking", "tour_agency", "travel_agency", "operator"}
        return bool(searchable_tokens & tour_markers) or any(marker in searchable for marker in {"tour_agency", "travel_agency"})
    return True


def intents_compatible(source: set[str], target: set[str]) -> bool:
    if not source or not target:
        return True
    if source & target:
        return True
    food_market = {"food", "shopping"}
    if source <= food_market and target <= food_market and source & food_market and target & food_market:
        return True
    return False


def token_overlap_score(text: str, wish: str, destination: str = "") -> float:
    ignored = set(content_tokens(destination))
    wish_tokens = [token for token in content_tokens(wish) if token not in ignored]
    if not wish_tokens:
        return 0.0
    text_tokens = content_tokens(text)
    matches = sum(1 for token in wish_tokens if token_matches(token, text_tokens))
    return matches / max(1, len(wish_tokens))


def activity_text(activity: Activity) -> str:
    return f"{activity.name} {activity.category} {matching_description(activity.description)}".lower()


def matching_description(description: str) -> str:
    kept: list[str] = []
    blocked_labels = {"matched query", "matched must-have", "google maps", "website", "address", "rating", "reviews"}
    for part in str(description or "").split("|"):
        cleaned = part.strip()
        label = cleaned.split(":", 1)[0].strip().lower() if ":" in cleaned else ""
        if label in blocked_labels:
            continue
        if cleaned.lower().startswith(("http://", "https://")):
            continue
        kept.append(cleaned)
    return " | ".join(kept)


def matched_must_have_covers(description: str, wish: str) -> bool:
    matched = description_field(description, "Matched must-have")
    wanted = " ".join(str(wish or "").lower().split())
    if not matched or not wanted:
        return False
    for part in matched.split(","):
        cleaned = " ".join(part.lower().split())
        if cleaned == wanted:
            return True
        if token_overlap_score(cleaned, wanted) >= 0.5:
            return True
    return False


def description_field(description: str, label: str) -> str:
    marker = f"{label}:"
    for part in str(description or "").split("|"):
        cleaned = part.strip()
        if cleaned.lower().startswith(marker.lower()):
            return cleaned.split(":", 1)[1].strip()
    return ""


def content_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9äöüß]+", str(text).lower())
        if len(token) > 2 and token not in STOP_WORDS
    ]


def token_matches(token: str, text_tokens: list[str]) -> bool:
    if token in text_tokens:
        return True
    variants = {token}
    if token == "natural":
        variants.add("nature")
    if token == "nature":
        variants.add("natural")
    if token == "architektur":
        variants.update({"architecture", "architectural"})
    if token in {"architecture", "architectural"}:
        variants.add("architektur")
    if token == "museen":
        variants.add("museum")
    if token == "museum":
        variants.add("museen")
    if token in {"restaurant", "restaurants"}:
        variants.update({"essen", "dining", "food"})
    if token == "essen":
        variants.update({"food", "restaurant", "restaurants", "dining"})
    if token.endswith("ies") and len(token) > 4:
        variants.add(f"{token[:-3]}y")
    if token.endswith("es") and len(token) > 4:
        variants.add(token[:-2])
    if token.endswith("s") and len(token) > 3:
        variants.add(token[:-1])
    return any(variant in text_tokens for variant in variants)
