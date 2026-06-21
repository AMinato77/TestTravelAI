from __future__ import annotations

import re

from app.models.travel_request import TravelRequest
from app.services.interest_taxonomy import ALLOWED_INTERESTS, normalize_interests, taxonomy_payload
from app.tools.openai_runtime import demo_fallback_enabled, generate_json


KNOWN_STYLES = ["relaxed", "adventure", "luxury", "budget", "balanced"]
NULL_TERMS = {"none", "null", "unknown", "keine", "kein", "n/a", "-"}


def parse_travel_request(text: str, fallback: TravelRequest) -> TravelRequest:
    """Parse a natural language travel request into structured fields."""
    if not text.strip():
        return fallback

    if not demo_fallback_enabled():
        data = generate_json(
            system_prompt=(
                "Extract a travel request as strict JSON. "
                "Use keys: destination, destination_scope, needs_destination_recommendation, "
                "destination_reasoning, duration_days, budget, interests, avoid, travel_style. "
                "interests must use only the provided allowed_interests categories. "
                "Use interest_descriptions to semantically map concrete user wishes to broad categories. "
                "destination_scope must be one of city, country, region, open. "
                "If the user asks which city should be recommended inside a country, keep the country "
                "as destination, set destination_scope='country', and needs_destination_recommendation=true. "
                "Do not replace an explicit country such as Japan with a fallback city. "
                "Do not create hard required-place constraints. "
                "Map concrete wishes to broader interests instead of blocking the itinerary with exact-place requirements. "
                "Do not invent concrete venues or landmarks from broad interests. "
                "A named fandom, manga title, or pop-culture franchise can map to anime. "
                "A named team, race series, stadium, or sport event can map to sport. "
                "A named console, arcade, or esports wish can map to gaming. "
                "A named cuisine, restaurant style, or market can map to food or street food. "
                "Only include shopping when the user explicitly mentions shopping, shops, stores, markets, buying, or browsing retail. "
                "Do not infer shopping merely because anime, manga, comics, gaming, fashion, or a fandom is mentioned. "
                "Example: 'I like anime' means interests includes anime. "
                "Example: 'I like anime shops' means interests includes anime and shopping. "
                "Example: 'I like food' must not become Mercado de San Miguel unless the user named it. "
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
                **taxonomy_payload(),
                "fallback": {
                    "destination": fallback.destination,
                    "destination_scope": fallback.destination_scope,
                    "needs_destination_recommendation": fallback.needs_destination_recommendation,
                    "duration_days": fallback.duration_days,
                    "budget": fallback.budget,
                    "interests": fallback.interests,
                    "avoid": fallback.avoid,
                    "travel_style": fallback.travel_style,
                },
            },
            model_env="OPENAI_REQUEST_MODEL",
        )
        destination = data.get("destination") or _parse_destination(text) or fallback.destination
        scope = str(data.get("destination_scope") or _infer_destination_scope(destination, text, fallback)).lower()
        needs_recommendation = bool(
            data.get("needs_destination_recommendation")
            or _asks_for_destination_recommendation(text)
            or (scope in {"country", "region", "open"} and _asks_for_destination_recommendation(text))
        )
        avoid = _merge_unique(_parse_avoid(text), data.get("avoid") or [], fallback.avoid)
        must_have: list[str] = []
        explicit_interests = _parse_interests(text)
        model_interests = data.get("interests") if _has_explicit_interest_signal(text) else []
        interests = _remove_avoided_interests(
            _normalize_interests(
                [
                    *(model_interests or explicit_interests or []),
                ]
            ),
            text,
            avoid,
        )
        return TravelRequest(
            destination=destination,
            destination_scope=scope if scope in {"city", "country", "region", "open"} else "city",
            needs_destination_recommendation=needs_recommendation,
            destination_reasoning=str(data.get("destination_reasoning") or "").strip(),
            duration_days=int(data.get("duration_days") or fallback.duration_days),
            budget=float(data.get("budget") or fallback.budget),
            interests=interests,
            must_have=[],
            avoid=avoid,
            travel_style=_clean_travel_style(text, data.get("travel_style"), fallback.travel_style),
        )

    avoid = _parse_avoid(text) or fallback.avoid
    destination = _parse_destination(text) or fallback.destination
    scope = _infer_destination_scope(destination, text, fallback)
    return TravelRequest(
        destination=destination,
        destination_scope=scope,
        needs_destination_recommendation=_asks_for_destination_recommendation(text),
        duration_days=_parse_days(text) or fallback.duration_days,
        budget=_parse_budget(text) or fallback.budget,
        interests=_remove_avoided_interests(
            _normalize_interests(
                [
                    *(_parse_interests(text) or fallback.interests),
                ]
            ),
            text,
            avoid,
        ),
        must_have=[],
        avoid=avoid,
        travel_style=_clean_travel_style(text, None, fallback.travel_style),
    )


def _parse_destination(text: str) -> str | None:
    patterns = [
        r"nach\s+([A-Za-zÄÖÜäöüß\- ]+?)(?:,| mit| fuer| fur| wenn| welche| budget|$)",
        r"in\s+([A-Za-zÄÖÜäöüß\- ]+?)(?:,| mit| fuer| fur| wenn| welche| budget|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip(" .?!")
            return _title_destination(value)
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
    interests = [interest for interest in ALLOWED_INTERESTS if interest in lower]
    if any(term in lower for term in ["anime", "manga", "comic", "comics", "cartoon"]):
        interests.append("anime")
    if any(term in lower for term in ["ramen", "noodle", "nudel", "restaurant", "restaurants", "essen"]):
        interests.append("food")
    if any(term in lower for term in ["motorsport", "formula 1", "formel 1", "f1", "stadium", "stadion", "football", "fußball", "fussball"]):
        interests.append("sport")
    if any(term in lower for term in ["stierkampf", "bullfighting", "bullen event", "toros", "las ventas"]):
        interests.append("sport")
    if any(term in lower for term in ["kunst", "art gallery", "gallery", "galerie", "street art"]):
        interests.append("culture")
    if any(term in lower for term in ["natur", "wandern", "strand", "beach", "outdoor", "park"]):
        interests.append("nature")
    if any(term in lower for term in ["architektur", "bauhaus", "gebäude", "gebaeude", "buildings"]):
        interests.append("architecture")
    if any(term in lower for term in ["fotografie", "photo spots", "foto spots", "aussichtspunkte"]):
        interests.append("photography")
    if "shopping" in interests and not _has_shopping_intent(lower):
        interests.remove("shopping")
    return sorted(set(interests))


def _has_explicit_interest_signal(text: str) -> bool:
    lower = text.lower()
    if _parse_interests(text):
        return True
    explicit_patterns = [
        r"\bich\s+(mag|liebe)\b",
        r"\binteressiere mich fuer\b",
        r"\binteressiere mich für\b",
        r"\binteressen?\b",
        r"\bpräferenz\w*\b",
        r"\bpraeferenz\w*\b",
        r"\bgern(e)?\b",
        r"\bfan von\b",
    ]
    return any(re.search(pattern, lower) for pattern in explicit_patterns)


def _infer_interests_from_specific_terms(specific_terms: list[str], request_text: str) -> list[str]:
    text = " ".join([request_text, *specific_terms]).lower()
    inferred: list[str] = []
    if any(
        term in text
        for term in [
            "anime",
            "manga",
            "otaku",
            "comic",
            "comics",
        ]
    ):
        inferred.append("anime")
    if any(term in text for term in ["football", "fussball", "fußball", "stadium", "stadion", "motorsport", "formula", "formel"]):
        inferred.append("sport")
    if any(term in text for term in ["bullfighting", "stierkampf", "toros", "las ventas", "bullen event"]):
        inferred.append("sport")
    if any(term in text for term in ["restaurant", "essen", "food", "tapas", "markt", "market"]):
        inferred.append("food")
    return sorted(set(inferred))


def _parse_avoid(text: str) -> list[str]:
    lower = text.lower()
    avoid: list[str] = []
    if re.search(r"(hasst|hasse|mag kein|mag keine|keine|kein|ohne)\s+(essen|food|restaurants?|cafes?|cafés?)", lower):
        avoid.extend(["food", "restaurants", "cafes"])
    if re.search(r"(hasst|hasse|mag kein|mag keine|keine|kein|ohne)\s+(museen|museum|museums)", lower):
        avoid.extend(["museums"])
    if re.search(r"(hasst|hasse|mag kein|mag keine|keine|kein|ohne)\s+(clubs?|nightlife|party)", lower):
        avoid.extend(["nightlife", "clubs"])
    disliked_match = re.findall(r"(?:hasst|hasse|mag kein|mag keine|ohne|aber nicht|aber keine[nr]?)\s+([a-z0-9 äöüß\-]+)", lower)
    for value in disliked_match:
        cleaned = value.strip(" .?!,")
        if cleaned and len(cleaned) <= 40:
            avoid.append(cleaned)
    return sorted(set(avoid))


def _remove_avoided_interests(interests: list[str], request_text: str, avoid: list[str]) -> list[str]:
    avoid_text = " ".join(avoid).lower()
    request_lower = request_text.lower()
    filtered: list[str] = []
    for interest in interests:
        normalized = str(interest).strip().lower()
        if not normalized:
            continue
        if normalized == "food" and any(term in avoid_text for term in ["food", "restaurant", "cafe"]):
            continue
        if normalized == "shopping" and not _has_shopping_intent(request_lower):
            continue
        if normalized in avoid_text:
            continue
        filtered.append(normalized)
    return sorted(set(filtered))


def _has_shopping_intent(text: str) -> bool:
    return any(
        term in text
        for term in [
            "shopping",
            "shop",
            "shops",
            "store",
            "stores",
            "mall",
            "market",
            "markt",
            "kaufen",
            "einkaufen",
            "boutique",
        ]
    )


def _normalize_interests(interests: list[str]) -> list[str]:
    return sorted(set(normalize_interests(interests)))


def _normalize_specific_terms(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value).strip().split()).lower()
        for term in _split_specific_term(cleaned):
            if not term or term in seen:
                continue
            seen.add(term)
            result.append(term)
    return result


def _parse_specific_terms(text: str) -> list[str]:
    lower = text.lower()
    terms: list[str] = []
    patterns = [
        r"(?:ich will|will|will ich|moechte|m.chte)\s+(?:da\s+)?(?:in|ins|zu|zum|zur)\s+(?:ein(?:e|en|em|er)?\s+)?([^,.!?]{3,80})",
        r"(?:ich will|will ich|will|moechte|m.chte)\s+(?:auch\s+)?(?:ein(?:e|en|em|er)?\s+)?([^,.!?]{0,50}?(?:shop|store|restaurant|cafe|cafÃ©|stadion|stadium|arena|event|experience|rennstrecke|track)[^,.!?]{0,40})",
        r"(?:unbedingt|auf jeden fall|pflicht|speziell|specific|konkret)\s+([^,.!?]{3,80})",
        r"(?:besuchen|sehen|erleben)\s+(?:will|möchte|moechte|wollen)?\s*([^,.!?]{3,80})",
        r"(?:ich will|will|möchte|moechte)\s+(?:unbedingt|auf jeden fall|speziell|konkret)\s*([^,.!?]{3,80})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, lower):
            cleaned = _clean_specific_phrase(match.group(1))
            if cleaned:
                terms.append(cleaned)
    terms.extend(_extract_anchor_specific_terms(lower))
    return _normalize_specific_terms(terms)


def _important_tokens(text: str) -> list[str]:
    replacements = {
        "formel": "formula",
        "strecke": "track",
        "rennstrecke": "track",
        "fuer": "for",
        "für": "for",
        "manga": "anime",
    }
    stop_words = {
        "ich",
        "will",
        "möchte",
        "moechte",
        "nach",
        "tage",
        "tag",
        "budget",
        "euro",
        "eur",
        "mag",
        "keine",
        "kein",
        "und",
        "oder",
        "mit",
        "for",
        "the",
        "visit",
        "shop",
        "shops",
        "store",
        "stores",
        "experience",
        "near",
        "nearby",
        "restaurants",
        "restaurant",
    }
    raw_tokens = re.findall(r"[a-z0-9äöüß]+", text.lower())
    tokens: list[str] = []
    for token in raw_tokens:
        normalized = replacements.get(token, token)
        if len(normalized) <= 2 or normalized in stop_words:
            continue
        tokens.append(normalized)
    return tokens


def _phrase_appears(candidate: str, request_text: str) -> bool:
    candidate_tokens = _important_tokens(candidate)
    if len(candidate_tokens) < 2:
        return False
    return " ".join(candidate_tokens[:2]) in request_text.lower()


def _clean_specific_phrase(value: str) -> str:
    cleaned = value.strip(" .?!,;:")
    stop_markers = [
        " aber ",
        " und ich ",
        " gehen und ",
        " gehen ",
        " wenn ",
        " dann ",
        " budget ",
        " für ",
        " fuer ",
        " auf den",
        " auf dem",
        " auf der",
    ]
    for marker in stop_markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip()
    cleaned = re.sub(r"\b(maessig|m..ig|m..iges|aehnlich|.hnlich)\b", "", cleaned)
    while True:
        shortened = re.sub(
            r"^(?:ich|will|moechte|m.chte|auch|da|in|ins|zu|zum|zur|ein|eine|einen|einem|einer)\s+",
            "",
            cleaned,
        )
        if shortened == cleaned:
            break
        cleaned = shortened
    cleaned = re.sub(r"\bfu\s+ballstadion\b", "fussballstadion", cleaned)
    cleaned = re.sub(r"\s+[a-z]$", "", cleaned)
    cleaned = " ".join(cleaned.split())
    generic = {"tage", "tag", "reise", "trip", "stadt", "plan", "budget"}
    if cleaned in generic or len(cleaned) < 3:
        return ""
    return cleaned


def _split_specific_term(value: str) -> list[str]:
    cleaned = _clean_specific_phrase(value)
    if not cleaned:
        return []
    if " und " not in cleaned:
        return [cleaned]

    anchors = [
        "shop",
        "store",
        "restaurant",
        "cafe",
        "stadion",
        "stadium",
        "arena",
        "event",
        "experience",
        "rennstrecke",
        "track",
    ]
    parts = [_clean_specific_phrase(part) for part in cleaned.split(" und ")]
    parts = [part for part in parts if part]
    anchored_parts = [part for part in parts if any(anchor in part for anchor in anchors)]
    if len(anchored_parts) >= 2:
        return anchored_parts
    return [cleaned]


def _extract_anchor_specific_terms(text: str) -> list[str]:
    anchors = [
        "shop",
        "store",
        "restaurant",
        "cafe",
        "stadion",
        "stadium",
        "arena",
        "event",
        "experience",
        "rennstrecke",
        "track",
    ]
    terms: list[str] = []
    segments = re.split(r"[,.!]|\b(?:dann|außerdem|ausserdem)\b", text)
    for segment in segments:
        for part in re.split(r"\bund\b", segment):
            words = re.findall(r"[\wÀ-ÿ]+", part.lower(), flags=re.UNICODE)
            for index, word in enumerate(words):
                if not any(anchor in word for anchor in anchors):
                    continue
                start = max(0, index - 4)
                phrase = _clean_specific_phrase(" ".join(words[start : index + 1]))
                if phrase:
                    terms.append(phrase)
    return terms


def _asks_for_destination_recommendation(text: str) -> bool:
    lower = text.lower()
    return any(
        phrase in lower
        for phrase in [
            "welche stadt",
            "was fuer eine stadt",
            "was für eine stadt",
            "stadt würdest du",
            "stadt wuerdest du",
            "empfehlen",
            "recommend",
            "which city",
        ]
    )


def _infer_destination_scope(destination: str, text: str, fallback: TravelRequest) -> str:
    if not destination:
        return fallback.destination_scope or "open"
    lower = destination.lower()
    if lower in {"japan", "deutschland", "frankreich", "italien", "spanien", "usa", "vereinigte staaten"}:
        return "country"
    if _asks_for_destination_recommendation(text):
        return "country"
    return "city"


def _title_destination(value: str) -> str:
    normalized = " ".join(value.split())
    known = {
        "japan": "Japan",
        "tokyo": "Tokyo",
        "tokio": "Tokyo",
        "osaka": "Osaka",
        "barcelona": "Barcelona",
    }
    return known.get(normalized.lower(), normalized.title())


def _merge_unique(*groups: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            cleaned = str(value).strip().lower()
            if not cleaned or cleaned in NULL_TERMS or cleaned in seen:
                continue
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _clean_travel_style(text: str, model_value, fallback: str) -> str:
    parsed = _parse_style(text)
    if parsed:
        return parsed
    value = str(model_value or "").strip().lower()
    if value not in KNOWN_STYLES:
        return fallback
    if value == "budget" and not _has_budget_style_intent(text):
        return fallback
    return value


def _parse_style(text: str) -> str | None:
    lower = text.lower()
    if "nicht stressig" in lower or "keinen stressigen" in lower:
        return "relaxed"
    if _has_budget_style_intent(text):
        return "budget"
    for style in KNOWN_STYLES:
        if style != "budget" and style in lower:
            return style
    return None


def _has_budget_style_intent(text: str) -> bool:
    lower = text.lower()
    return any(
        phrase in lower
        for phrase in [
            "budget travel",
            "budget trip",
            "low budget",
            "cheap trip",
            "guenstig",
            "günstig",
            "preiswert",
            "sparsam",
            "billig reisen",
        ]
    )
