from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from app.models.preference_source import PreferenceSource
from app.models.travel_request import TravelRequest
from app.services.wish_matching import content_tokens, infer_intents, query_matches_wish, token_matches
from app.tools.openai_runtime import demo_fallback_enabled, generate_json


@dataclass(slots=True)
class PlaceQuery:
    query: str
    reason: str = ""
    source: str = "query_planner"
    must_have: list[str] = field(default_factory=list)


def plan_place_queries(
    request: TravelRequest,
    memory_context: list[PreferenceSource],
    max_queries: int | None = None,
) -> tuple[list[PlaceQuery], dict]:
    """Create concrete Google Places text queries from request and semantic memory."""
    max_queries = _configured_int("TRAVELAI_MAX_PLACE_QUERIES", max_queries or 6, minimum=1, maximum=20)
    per_must_have = _configured_int("TRAVELAI_QUERIES_PER_MUST_HAVE", 2, minimum=1, maximum=max_queries)
    if not demo_fallback_enabled():
        try:
            data = generate_json(
                system_prompt=(
                    "You are a query planning agent for Google Places Text Search. "
                    "Create concrete search queries from the user's free-form wishes and relevant memory. "
                    "Do not reduce wishes to broad categories. Prefer exact, visitable place intents such as "
                    "'anime figure stores Akihabara' or 'non touristy food markets Rome'. "
                    "Respect avoid terms by not creating queries for avoided topics. "
                    "For each query, must_have must contain only the exact request.must_have item(s) that this query directly searches for. "
                    "Do not copy all must_have items into every query. Do not include broad tags unless they are exact request.must_have values. "
                    "A food query must not list beach/nature/culture wishes. A beach/nature query must not list food/shopping wishes. "
                    "Usually create 1-2 precise queries per concrete wish, up to max_queries. "
                    "If request.use_profile_memory is true, or if retrieved memory contains durable Gmail/newsletter "
                    "or planned-trip preference patterns, actively inspect retrieved memory and add a small number "
                    "of additional preference-based queries when they improve the trip. "
                    "Memory may complement explicit wishes but must never override explicit must_have or avoid terms. "
                    "Memory-derived queries should use must_have=[] unless they directly search for an explicit current must_have. "
                    "Return strict JSON with keys summary, queries, memory_usage, ignored_memories. queries is a list "
                    "of objects with query, reason, must_have, optional source. memory_usage is a list of objects with "
                    "memory, effect, confidence. ignored_memories is a list of objects with memory, reason. "
                    "Every query must include the concrete destination city."
                ),
                payload={
                    "request": _request_payload(request),
                    "memory_context": [
                        {
                            "source_type": source.source_type,
                            "name": source.name,
                            "text": source.text[:1200],
                        }
                        for source in memory_context[:8]
                    ],
                    "max_queries": max_queries,
                },
                model_env="OPENAI_QUERY_PLANNING_MODEL",
            )
            queries = _parse_queries(data, request, max_queries)
            memory_usage = _memory_usage_items(data.get("memory_usage"))
            ignored_memories = _ignored_memory_items(data.get("ignored_memories"))
            queries, inferred_usage = _augment_queries_from_memory(queries, request, memory_context, max_queries)
            memory_usage = [*memory_usage, *inferred_usage]
            if queries:
                balanced = _balanced_query_selection(queries, request, max_queries, per_must_have)
                return balanced, {
                    "enabled": True,
                    "summary": str(data.get("summary") or "").strip(),
                    "max_queries": max_queries,
                    "queries_per_must_have": per_must_have,
                    "generated_queries": len(queries),
                    "selected_queries": len(balanced),
                    "memory_usage": memory_usage[:8],
                    "ignored_memories": ignored_memories,
                }
        except Exception as exc:
            raise RuntimeError(f"Query Planning Agent failed; no deterministic demo fallback is used in OpenAI mode: {exc}") from exc

    fallback = _fallback_queries(request, max_queries)
    balanced = _balanced_query_selection(fallback, request, max_queries, per_must_have)
    return balanced, {
        "enabled": False,
        "summary": "Query Planning Agent used deterministic fallback.",
        "max_queries": max_queries,
        "queries_per_must_have": per_must_have,
        "generated_queries": len(fallback),
        "selected_queries": len(balanced),
        "memory_usage": [],
        "ignored_memories": [],
    }


def _parse_queries(data: dict, request: TravelRequest, max_queries: int) -> list[PlaceQuery]:
    result: list[PlaceQuery] = []
    seen: set[str] = set()
    destination = request.destination
    destination_lower = destination.lower()
    for item in data.get("queries", []):
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        if destination_lower and destination_lower not in query.lower():
            query = f"{query} {destination}"
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        matched_must_have = _matched_requirements_for_query(
            query=query,
            reason=str(item.get("reason") or ""),
            explicit=_as_list(item.get("must_have")),
            request=request,
        )
        result.append(
            PlaceQuery(
                query=query,
                reason=str(item.get("reason") or "").strip(),
                source=str(item.get("source") or "query_planner").strip() or "query_planner",
                must_have=matched_must_have,
            )
        )
        if len(result) >= max_queries:
            break
    return result


def _fallback_queries(request: TravelRequest, max_queries: int) -> list[PlaceQuery]:
    wishes = _merge_unique(request.query_hints, request.must_have)
    if not wishes:
        wishes = ["best things to do", "local experiences"]
    queries: list[PlaceQuery] = []
    seen: set[str] = set()
    for wish in wishes:
        if _conflicts_with_avoid(wish, request.avoid):
            continue
        query = f"{wish} {request.destination}".strip()
        key = query.lower()
        if not query or key in seen:
            continue
        seen.add(key)
        queries.append(PlaceQuery(query=query, reason="Built from user request.", source="fallback", must_have=[wish]))
        if len(queries) >= max_queries:
            break
    if not queries and request.destination:
        queries.append(PlaceQuery(query=f"best things to do {request.destination}", reason="Generic fallback.", source="fallback"))
    return queries


def _augment_queries_from_memory(
    queries: list[PlaceQuery],
    request: TravelRequest,
    memory_context: list[PreferenceSource],
    max_queries: int,
) -> tuple[list[PlaceQuery], list[dict]]:
    if not _memory_should_influence_queries(request, memory_context):
        return queries, []
    destination = request.destination
    existing = {" ".join(query.query.lower().split()) for query in queries}
    avoid_intents = infer_intents(" ".join(request.avoid))
    current_intents = infer_intents(" ".join([*request.must_have, *request.query_hints, *request.interest_tags]))
    memory_text = _positive_memory_text(memory_context[:8])
    memory_intents = infer_intents(memory_text)
    additions: list[PlaceQuery] = []
    usage: list[dict] = []

    templates = {
        "nature": [
            ("relaxed parks and scenic outdoor experiences", "Added because retrieved memory shows nature, beach, or relaxed outdoor travel interest."),
            ("scenic viewpoints and gardens", "Added because retrieved memory shows outdoor or scenic travel preferences."),
        ],
        "sport": [
            ("football stadium tour", "Added because retrieved memory shows football or stadium interest."),
            ("football museum or sports experience", "Added because retrieved memory shows football or sports-related planning choices."),
        ],
        "food": [
            ("local food markets", "Added because retrieved memory shows interest in local food experiences."),
            ("authentic local restaurants", "Added because retrieved memory shows restaurant or cuisine preferences."),
        ],
        "culture": [
            ("architecture walking tour", "Added because retrieved memory shows interest in architecture or cultural additions."),
        ],
        "shopping": [
            ("local markets and specialty shops", "Added because retrieved memory shows market or shopping preferences."),
        ],
        "entertainment": [
            ("gaming anime entertainment spots", "Added because retrieved memory shows entertainment interests."),
        ],
        "nightlife": [
            ("rooftop bars nightlife", "Added because retrieved memory shows nightlife preferences."),
        ],
    }

    for intent in ["nature", "sport", "food", "culture", "shopping", "entertainment", "nightlife"]:
        if intent not in memory_intents or intent in avoid_intents:
            continue
        if intent in current_intents and sum(1 for query in queries if intent in infer_intents(query.query)) >= 2:
            continue
        for suffix, reason in templates.get(intent, [])[:1]:
            query_text = f"{destination} {suffix}".strip()
            key = " ".join(query_text.lower().split())
            if key in existing or _conflicts_with_avoid(query_text, request.avoid):
                continue
            additions.append(
                PlaceQuery(
                    query=query_text,
                    reason=reason,
                    source="profile_memory",
                    must_have=[],
                )
            )
            usage.append(
                {
                    "memory": _memory_preview_for_intent(memory_context, intent),
                    "effect": f"Added optional profile-memory query: {query_text}.",
                    "confidence": 0.75,
                }
            )
            existing.add(key)
            break
        if len(queries) + len(additions) >= max_queries:
            break
    return [*queries, *additions[: max(0, max_queries - len(queries))]], usage


def _memory_should_influence_queries(request: TravelRequest, memory_context: list[PreferenceSource]) -> bool:
    if getattr(request, "use_profile_memory", False) and memory_context:
        return True
    if not memory_context:
        return False
    durable_source_types = {"email_newsletter", "planned_trip_summary"}
    for source in memory_context:
        source_type = str(source.source_type or "").strip().lower()
        text = str(source.text or "").lower()
        if source_type in durable_source_types and any(
            marker in text
            for marker in [
                "reliable travel preference patterns",
                "soft query directions",
                "positive patterns",
                "selected highlights",
                "travel style signal",
                "recurring interests",
            ]
        ):
            return True
    return False


def _memory_preview_for_intent(memory_context: list[PreferenceSource], intent: str) -> str:
    for source in memory_context:
        positive_text = _positive_memory_text([source])
        if intent in infer_intents(positive_text):
            return f"{source.name}: {' '.join(source.text.split())[:160]}"
    return "Retrieved profile memory"


def _positive_memory_text(memory_context: list[PreferenceSource]) -> str:
    positive_parts: list[str] = []
    blocked_markers = {
        "avoid",
        "avoidance",
        "avoided",
        "dislike",
        "disliked",
        "negative",
        "not my style",
        "already known",
        "already visited",
        "kenne ich",
        "nicht mein stil",
        "meiden",
        "vermeiden",
        "skip",
        "skipped",
    }
    for source in memory_context:
        for line in re.split(r"[\n.;]+", source.text):
            lower = line.lower()
            if any(marker in lower for marker in blocked_markers):
                continue
            positive_parts.append(line)
    return " ".join(positive_parts)


def _balanced_query_selection(
    queries: list[PlaceQuery],
    request: TravelRequest,
    max_queries: int,
    per_must_have: int,
) -> list[PlaceQuery]:
    requirements = _merge_unique(request.must_have) or _merge_unique(request.query_hints)
    if len(queries) <= max_queries:
        return _normalize_query_must_haves(queries, request, requirements)

    selected: list[PlaceQuery] = []
    selected_keys: set[str] = set()
    counts_by_requirement: dict[str, int] = {requirement.lower(): 0 for requirement in requirements}

    for requirement in requirements:
        matching = [
            query
            for query in queries
            if query.query.strip().lower() not in selected_keys
            and _query_matches_requirement(query, requirement, request.destination)
        ]
        matching.sort(key=lambda query: _query_specificity_score(query, requirement, request.destination), reverse=True)
        for query in matching[:per_must_have]:
            _append_selected_query(query, selected, selected_keys)
            counts_by_requirement[requirement.lower()] = counts_by_requirement.get(requirement.lower(), 0) + 1
            if len(selected) >= max_queries:
                return selected

    remaining = [
        query
        for query in queries
        if query.query.strip().lower() not in selected_keys
    ]
    remaining.sort(key=lambda query: _coverage_score(query, requirements, counts_by_requirement, request.destination), reverse=True)
    for query in remaining:
        _append_selected_query(query, selected, selected_keys)
        if len(selected) >= max_queries:
            break
    return _normalize_query_must_haves(selected, request, requirements)


def _normalize_query_must_haves(
    queries: list[PlaceQuery],
    request: TravelRequest,
    requirements: list[str] | None = None,
) -> list[PlaceQuery]:
    requirements = requirements or _merge_unique(request.must_have) or _merge_unique(request.query_hints)
    normalized: list[PlaceQuery] = []
    for query in queries:
        matched = [
            requirement
            for requirement in requirements
            if _query_matches_requirement(query, requirement, request.destination)
        ]
        explicit = [
            requirement
            for requirement in _merge_unique(query.must_have)
            if _same_requirement(requirement, requirements)
            and _query_matches_requirement(PlaceQuery(query=query.query, reason=query.reason), requirement, request.destination)
        ]
        normalized.append(
            PlaceQuery(
                query=query.query,
                reason=query.reason,
                source=query.source,
                must_have=_merge_unique(matched, explicit),
            )
        )
    return normalized


def _append_selected_query(query: PlaceQuery, selected: list[PlaceQuery], selected_keys: set[str]) -> None:
    key = query.query.strip().lower()
    if not key or key in selected_keys:
        return
    selected.append(query)
    selected_keys.add(key)


def _query_matches_requirement(query: PlaceQuery, requirement: str, destination: str = "") -> bool:
    return query_matches_wish(query.query, query.reason, requirement, destination)


def _query_specificity_score(query: PlaceQuery, requirement: str, destination: str = "") -> float:
    text = f"{query.query} {query.reason}".lower()
    ignored = set(content_tokens(destination))
    tokens = [token for token in content_tokens(requirement) if token not in ignored]
    if not tokens:
        return 0.0
    text_tokens = [token for token in content_tokens(text) if token not in ignored]
    matches = sum(1 for token in tokens if token_matches(token, text_tokens))
    length_penalty = max(0, len(content_tokens(query.query)) - 8) * 0.05
    return matches / len(tokens) - length_penalty


def _coverage_score(query: PlaceQuery, requirements: list[str], counts_by_requirement: dict[str, int], destination: str = "") -> float:
    score = 0.0
    for requirement in requirements:
        if not _query_matches_requirement(query, requirement, destination):
            continue
        existing_count = counts_by_requirement.get(requirement.lower(), 0)
        score += 1.0 / (existing_count + 1)
    return score


def _conflicts_with_avoid(text: str, avoid: list[str]) -> bool:
    haystack = text.lower()
    return any(term.strip().lower() and term.strip().lower() in haystack for term in avoid)


def _request_payload(request: TravelRequest) -> dict:
    return {
        "destination": request.destination,
        "destination_scope": request.destination_scope,
        "duration_days": request.duration_days,
        "budget": request.budget,
        "must_have": request.must_have,
        "avoid": request.avoid,
        "interest_tags": request.interest_tags,
        "query_hints": request.query_hints,
        "travel_style": request.travel_style,
        "use_profile_memory": getattr(request, "use_profile_memory", False),
    }


def _memory_usage_items(value) -> list[dict]:
    items: list[dict] = []
    if not isinstance(value, list):
        return items
    for item in value:
        if not isinstance(item, dict):
            continue
        memory = str(item.get("memory") or "").strip()
        effect = str(item.get("effect") or "").strip()
        confidence = item.get("confidence")
        if not memory and not effect:
            continue
        items.append(
            {
                "memory": memory,
                "effect": effect,
                "confidence": confidence,
            }
        )
    return items[:8]


def _ignored_memory_items(value) -> list[dict]:
    items: list[dict] = []
    if not isinstance(value, list):
        return items
    for item in value:
        if not isinstance(item, dict):
            continue
        memory = str(item.get("memory") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if not memory and not reason:
            continue
        items.append({"memory": memory, "reason": reason})
    return items[:8]


def _matched_requirements_for_query(
    query: str,
    reason: str,
    explicit: list[str],
    request: TravelRequest,
) -> list[str]:
    requirements = _merge_unique(request.must_have, request.query_hints)
    matched = [
        requirement
        for requirement in requirements
        if query_matches_wish(query, reason, requirement, request.destination)
    ]
    explicit_matches = [
        requirement
        for requirement in explicit
        if _same_requirement(requirement, requirements)
        and query_matches_wish(query, reason, requirement, request.destination)
    ]
    return _merge_unique(matched, explicit_matches)


def _soft_overlap(text: str, requirement: str, ignored_tokens: list[str] | None = None) -> bool:
    ignored = set(ignored_tokens or [])
    tokens = [token for token in _content_tokens(requirement) if token not in ignored]
    if not tokens:
        return False
    text_tokens = [token for token in _content_tokens(text) if token not in ignored]
    matches = sum(1 for token in tokens if _token_matches(token, text_tokens))
    threshold = 1 if len(tokens) <= 2 else max(2, round(len(tokens) * 0.45))
    return matches >= threshold


def _same_requirement(value: str, requirements: list[str]) -> bool:
    key = " ".join(str(value).lower().split())
    return key in {" ".join(requirement.lower().split()) for requirement in requirements}


def _token_matches(token: str, text_tokens: list[str]) -> bool:
    if token in text_tokens:
        return True
    variants = {token}
    if token == "natural":
        variants.add("nature")
    if token == "nature":
        variants.add("natural")
    if token.endswith("ies") and len(token) > 4:
        variants.add(f"{token[:-3]}y")
    if token.endswith("s") and len(token) > 3:
        variants.add(token[:-1])
    return any(variant in text_tokens for variant in variants)


def _content_tokens(text: str) -> list[str]:
    import re

    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "und",
        "oder",
        "mit",
        "von",
        "fuer",
        "fÃ¼r",
        "places",
        "place",
        "spots",
        "spot",
        "areas",
        "area",
        "side",
        "things",
        "trip",
        "travel",
        "tour",
    }
    return [
        token
        for token in re.findall(r"[a-z0-9Ã¤Ã¶Ã¼ÃŸ]+", str(text).lower())
        if len(token) > 2 and token not in stop_words
    ]


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]


def _merge_unique(*groups: list[str]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in _as_list(group):
            cleaned = " ".join(str(value).strip().split())
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            values.append(cleaned)
    return values


def _configured_int(name: str, fallback: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(fallback)))
    except (TypeError, ValueError):
        value = fallback
    return max(minimum, min(value, maximum))
