from __future__ import annotations

import re

from app.models.travel_request import TravelRequest
from app.tools.openai_runtime import demo_fallback_enabled, generate_json


KNOWN_STYLES = {"balanced", "relaxed", "adventure", "luxury", "budget"}
DESTINATION_STOP_WORDS = {
    "budget",
    "tage",
    "tag",
    "days",
    "day",
    "ich",
    "i",
    "with",
    "mit",
    "und",
    "and",
}
NULL_TERMS = {"", "none", "null", "unknown", "n/a", "-", "keine", "kein"}


def parse_travel_request(text: str, fallback: TravelRequest) -> TravelRequest:
    """Parse the user request without reducing it to fixed demo categories."""
    if not text.strip():
        return fallback

    if not demo_fallback_enabled():
        try:
            data = generate_json(
                system_prompt=(
                    "Extract a travel planning request as strict JSON. "
                    "Use keys: destination, destination_scope, needs_destination_recommendation, "
                    "destination_reasoning, duration_days, budget, must_have, avoid, interest_tags, "
                    "query_hints, travel_style. "
                    "must_have contains concrete free-form wishes exactly as the user means them, "
                    "for example merchandise shops, local food markets, quiet neighborhoods, "
                    "architecture walks, specific events, or niche experiences. "
                    "avoid contains concrete dislikes and hard exclusions. "
                    "Normalize must_have, avoid, interest_tags and query_hints into concise, "
                    "search-friendly English phrases unless a proper noun must stay unchanged. "
                    "For avoid, store the excluded thing only, not negation words like no, not, "
                    "keine, kein, ohne. "
                    "interest_tags are optional broad UI/analytics tags only; never replace concrete "
                    "must_have wishes with tags. "
                    "query_hints are concise search phrases that would help Google Places. "
                    "Do not invent venues. Do not use old memory as destination if the current text "
                    "explicitly names a destination. "
                    "destination_scope must be city, country, region, or open. "
                    "travel_style must be balanced, relaxed, adventure, luxury, or budget. "
                    "If the user asks which city to choose inside a country, keep the country as "
                    "destination, set destination_scope='country', and needs_destination_recommendation=true."
                ),
                payload={
                    "request": text,
                    "fallback": _request_payload(fallback),
                },
                model_env="OPENAI_REQUEST_MODEL",
            )
            return _request_from_data(data, text, fallback)
        except Exception:
            pass

    return _fallback_parse(text, fallback)


def _request_from_data(data: dict, raw_text: str, fallback: TravelRequest) -> TravelRequest:
    destination = str(data.get("destination") or "").strip() or _parse_destination(raw_text) or fallback.destination
    scope = str(data.get("destination_scope") or _infer_destination_scope(destination, raw_text, fallback)).strip().lower()
    if scope not in {"city", "country", "region", "open"}:
        scope = fallback.destination_scope or "city"
    needs_recommendation = bool(
        data.get("needs_destination_recommendation")
        or _asks_for_destination_recommendation(raw_text)
        or (scope in {"country", "region", "open"} and _asks_for_destination_recommendation(raw_text))
    )
    parsed_must_have = _as_list(data.get("must_have"))
    parsed_avoid = _as_list(data.get("avoid"))
    must_have = _merge_unique(parsed_must_have or _fallback_must_have(raw_text))
    avoid = _clean_avoid_terms(_merge_unique(parsed_avoid or _fallback_avoid(raw_text), fallback.avoid))
    query_hints = _merge_unique(_as_list(data.get("query_hints")), must_have)
    return TravelRequest(
        destination=destination,
        destination_scope=scope,
        needs_destination_recommendation=needs_recommendation,
        destination_reasoning=str(data.get("destination_reasoning") or "").strip(),
        duration_days=_safe_int(data.get("duration_days"), fallback.duration_days, minimum=1, maximum=14),
        budget=_safe_float(data.get("budget"), fallback.budget),
        must_have=must_have,
        avoid=avoid,
        interest_tags=_clean_tags(data.get("interest_tags")),
        query_hints=query_hints,
        travel_style=_clean_style(data.get("travel_style"), raw_text, fallback.travel_style),
    )


def _fallback_parse(text: str, fallback: TravelRequest) -> TravelRequest:
    destination = _parse_destination(text) or fallback.destination
    scope = _infer_destination_scope(destination, text, fallback)
    must_have = _fallback_must_have(text)
    avoid = _clean_avoid_terms(_merge_unique(_fallback_avoid(text), fallback.avoid))
    return TravelRequest(
        destination=destination,
        destination_scope=scope,
        needs_destination_recommendation=_asks_for_destination_recommendation(text),
        duration_days=_parse_days(text) or fallback.duration_days,
        budget=_parse_budget(text) or fallback.budget,
        must_have=must_have,
        avoid=avoid,
        interest_tags=[],
        query_hints=must_have[:],
        travel_style=_clean_style(None, text, fallback.travel_style),
    )


def _parse_destination(text: str) -> str | None:
    patterns = [
        r"\bnach\s+([^,.;!?]+?)(?:\s+budget|\s+\d+\s*(?:tage|tag|days|day)|,|\.|!|\?|$)",
        r"\bin\s+([^,.;!?]+?)(?:\s+budget|\s+\d+\s*(?:tage|tag|days|day)|,|\.|!|\?|$)",
        r"\bto\s+([^,.;!?]+?)(?:\s+budget|\s+\d+\s*(?:days|day)|,|\.|!|\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        value = " ".join(match.group(1).split()).strip(" .?!,")
        if value and value.lower() not in DESTINATION_STOP_WORDS:
            return value.title()
    return None


def _parse_days(text: str) -> int | None:
    match = re.search(r"(\d{1,2})\s*(tage|tag|days|day)", text.lower())
    return max(1, min(int(match.group(1)), 14)) if match else None


def _parse_budget(text: str) -> float | None:
    match = re.search(r"(?:budget\s*(?:von|:)?\s*)?(\d{2,5})\s*(eur|euro|€)", text.lower())
    return float(match.group(1)) if match else None


def _fallback_must_have(text: str) -> list[str]:
    cleaned = re.sub(r"\b(?:ich\s+)?(?:will|möchte|moechte|suche|mag|liebe)\b", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:\d{1,2}\s*(?:tage|tag|days|day)|budget\s*\d{2,5}\s*(?:eur|euro|€)?)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:nach|in|to)\s+[A-Za-zÄÖÜäöüß \-]+?(?=,|\.|!|\?|$)", "", cleaned, flags=re.IGNORECASE)
    parts = re.split(r",|\bund\b|\band\b|\baber\b|\bbut\b", cleaned, flags=re.IGNORECASE)
    wishes: list[str] = []
    for part in parts:
        value = " ".join(part.strip(" .?!").split())
        if len(value) < 3:
            continue
        if any(marker in value.lower() for marker in ["hasse", "hate", "kein", "keine", "ohne", "avoid"]):
            continue
        if value.lower() in DESTINATION_STOP_WORDS:
            continue
        wishes.append(value)
    return _merge_unique(wishes)


def _fallback_avoid(text: str) -> list[str]:
    values: list[str] = []
    patterns = [
        r"(?:mag keine|mag kein|hasse|hasst|hate|avoid|ohne|keinen|keine|kein)\s+([^,.!?]+)",
        r"(?:aber nicht|but not|not)\s+([^,.!?]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = " ".join(match.group(1).strip(" .?!").split())
            if value:
                values.append(value)
    return _merge_unique(values)


def _asks_for_destination_recommendation(text: str) -> bool:
    lower = text.lower()
    return any(
        phrase in lower
        for phrase in [
            "welche stadt",
            "was für eine stadt",
            "was fuer eine stadt",
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
    if _asks_for_destination_recommendation(text):
        return "country"
    if destination.lower() in {"japan", "deutschland", "germany", "italy", "italien", "spain", "spanien", "france", "frankreich"}:
        return "country"
    return "city"


def _clean_style(value, text: str, fallback: str) -> str:
    lower = text.lower()
    if any(term in lower for term in ["entspannt", "nicht stressig", "keinen stressigen", "relaxed", "slow"]):
        return "relaxed"
    if any(term in lower for term in ["luxury", "luxus", "premium"]):
        return "luxury"
    if any(term in lower for term in ["low budget", "budget travel", "günstig", "guenstig", "sparsam"]):
        return "budget"
    normalized = str(value or "").strip().lower()
    return normalized if normalized in KNOWN_STYLES else fallback


def _request_payload(request: TravelRequest) -> dict:
    return {
        "destination": getattr(request, "destination", ""),
        "destination_scope": getattr(request, "destination_scope", "open"),
        "needs_destination_recommendation": getattr(request, "needs_destination_recommendation", False),
        "duration_days": getattr(request, "duration_days", 3),
        "budget": getattr(request, "budget", 600),
        "must_have": getattr(request, "must_have", []),
        "avoid": getattr(request, "avoid", []),
        "interest_tags": getattr(request, "interest_tags", []),
        "query_hints": getattr(request, "query_hints", []),
        "travel_style": getattr(request, "travel_style", "balanced"),
    }


def _safe_int(value, fallback: int, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(int(value), maximum))
    except (TypeError, ValueError):
        return fallback


def _safe_float(value, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]


def _clean_tags(value) -> list[str]:
    return _merge_unique(_as_list(value))


def _clean_avoid_terms(values) -> list[str]:
    cleaned_values: list[str] = []
    for value in _as_list(values):
        cleaned = re.sub(
            r"^(?:no|not|without|avoid|kein|keine|keinen|ohne)\s+",
            "",
            str(value).strip(),
            flags=re.IGNORECASE,
        )
        if cleaned.strip():
            cleaned_values.append(cleaned.strip())
    return _merge_unique(cleaned_values)


def _merge_unique(*groups: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in _as_list(group):
            cleaned = " ".join(str(value).strip().split())
            key = cleaned.lower()
            if key in NULL_TERMS or key in seen:
                continue
            seen.add(key)
            result.append(cleaned)
    return result
