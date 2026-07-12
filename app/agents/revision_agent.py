from __future__ import annotations

from app.models.itinerary import Itinerary
from app.services.serialization import itinerary_to_dict
from app.tools.openai_runtime import demo_fallback_enabled, generate_json


REVISION_FUNCTION_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["replace_activity", "add_more_similar", "reduce_intensity", "change_budget_level", "general_revision"],
        },
        "target_day": {"type": ["integer", "null"]},
        "target_activity": {"type": ["string", "null"]},
        "avoid_additions": {"type": "array", "items": {"type": "string"}},
        "must_have_additions": {"type": "array", "items": {"type": "string"}},
        "replacement_requirements": {"type": "array", "items": {"type": "string"}},
        "query_hints": {"type": "array", "items": {"type": "string"}},
        "revision_instruction": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": [
        "intent",
        "target_day",
        "target_activity",
        "avoid_additions",
        "must_have_additions",
        "replacement_requirements",
        "query_hints",
        "revision_instruction",
        "reasoning",
    ],
    "additionalProperties": False,
}


def interpret_revision_feedback(
    itinerary: Itinerary,
    feedback: str,
    original_request: dict,
    must_have: list[str],
    avoid: list[str],
) -> dict:
    if demo_fallback_enabled():
        return _fallback_revision(itinerary, feedback)

    data = generate_json(
        system_prompt=(
            "You are a travel agency revision agent. The user reacts to an existing itinerary. "
            "Classify the feedback, identify the affected activity or day, and return a targeted revision. "
            "Preserve unaffected good activities; the orchestrator should change only the requested activity/day. "
            "Do not invent venues. Generate concrete Google Places query_hints for replacement candidates. "
            "If the user already knows or dislikes a place, use intent replace_activity and add that place to avoid_additions. "
            "For replacements, the removed activity is context only: identify it and avoid it, but do not search for "
            "similar old venues unless the user explicitly asks for a similar alternative. Query hints and replacement "
            "requirements must be based on what the user wants now. If feedback says 'instead of X I want Y', search for Y "
            "and put X only into avoid_additions. Keep requirements specific to the current case. "
            "If the user asks for more of something, use add_more_similar and create concrete Google Places query_hints. "
            "If a day is too full or stressful, use reduce_intensity. Return strict JSON only."
        ),
        payload={
            "current_itinerary": itinerary_to_dict(itinerary),
            "feedback": feedback,
            "original_request": original_request,
            "must_have": must_have,
            "avoid": avoid,
        },
        model_env="OPENAI_REVISION_MODEL",
    )
    return _normalize_revision(data)


def _normalize_revision(data: dict) -> dict:
    return {
        "intent": str(data.get("intent") or "general_revision"),
        "target_day": data.get("target_day") if isinstance(data.get("target_day"), int) else None,
        "target_activity": str(data.get("target_activity") or "").strip() or None,
        "avoid_additions": _as_list(data.get("avoid_additions")),
        "must_have_additions": _as_list(data.get("must_have_additions")),
        "replacement_requirements": _as_list(data.get("replacement_requirements")),
        "query_hints": _as_list(data.get("query_hints")),
        "revision_instruction": str(data.get("revision_instruction") or "").strip(),
        "reasoning": str(data.get("reasoning") or "").strip(),
    }


def _fallback_revision(itinerary: Itinerary, feedback: str) -> dict:
    text = feedback.lower()
    target = _find_mentioned_activity(itinerary, text)
    if target:
        requirements = _replacement_requirements_for_target(feedback)
        return {
            "intent": "replace_activity",
            "target_day": target["day"],
            "target_activity": target["name"],
            "avoid_additions": [target["name"]],
            "must_have_additions": [],
            "replacement_requirements": requirements,
            "query_hints": _replacement_queries(itinerary.destination, target, feedback),
            "revision_instruction": f"Replace {target['name']} according to the user's feedback.",
            "reasoning": "The feedback mentioned an existing activity; replacement search follows the new user request.",
        }
    if any(term in text for term in ["mehr davon", "more like", "mehr solche"]):
        return {
            "intent": "add_more_similar",
            "target_day": None,
            "target_activity": None,
            "avoid_additions": [],
            "must_have_additions": [feedback],
            "replacement_requirements": [],
            "query_hints": [f"{feedback} {itinerary.destination}"],
            "revision_instruction": "Add more similar activities without overloading the plan.",
            "reasoning": "The feedback asks for more similar content.",
        }
    if any(term in text for term in ["stressig", "zu voll", "too full", "weniger"]):
        return {
            "intent": "reduce_intensity",
            "target_day": _first_day_number(text),
            "target_activity": None,
            "avoid_additions": [],
            "must_have_additions": [],
            "replacement_requirements": [],
            "query_hints": [],
            "revision_instruction": "Reduce the intensity of the affected day.",
            "reasoning": "The feedback asks for a less packed plan.",
        }
    return {
        "intent": "general_revision",
        "target_day": None,
        "target_activity": None,
        "avoid_additions": [],
        "must_have_additions": [feedback],
        "replacement_requirements": [feedback],
        "query_hints": [f"{feedback} {itinerary.destination}"],
        "revision_instruction": feedback,
        "reasoning": "General feedback was converted into a soft requirement.",
    }


def _find_mentioned_activity(itinerary: Itinerary, feedback_lower: str) -> dict | None:
    best: dict | None = None
    for day in itinerary.days:
        for activity in day.activities:
            name = activity.name.lower()
            if name and name in feedback_lower:
                return {
                    "day": day.day,
                    "name": activity.name,
                    "category": activity.category,
                    "description": activity.description,
                }
            overlap = sum(1 for token in name.split() if len(token) > 3 and token in feedback_lower)
            if overlap and (best is None or overlap > best["overlap"]):
                best = {
                    "day": day.day,
                    "name": activity.name,
                    "category": activity.category,
                    "description": activity.description,
                    "overlap": overlap,
                }
    return best


def _replacement_requirements_for_target(feedback: str) -> list[str]:
    cleaned = " ".join(str(feedback or "").strip().split())
    return [cleaned] if cleaned else ["replacement requested by user feedback"]


def _replacement_queries(destination: str, target: dict, feedback: str) -> list[str]:
    cleaned_feedback = _feedback_without_target(feedback, str(target.get("name") or ""))
    if cleaned_feedback:
        return [f"{cleaned_feedback} {destination}"]
    category = str(target.get("category") or "activity").strip()
    return [f"{destination} alternative {category}"]


def _feedback_without_target(feedback: str, target_name: str) -> str:
    import re

    cleaned = " ".join(str(feedback or "").strip().split())
    target_tokens = [re.escape(token) for token in re.findall(r"[A-Za-z0-9]+", target_name) if len(token) > 2]
    for token in target_tokens:
        cleaned = re.sub(rf"\b{token}\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b(kenne|kenn|schon|war|ich|das|die|der|den|dem|beim|bei|statt|anstatt|stattdessen|instead|alternative|ersetze|gib|mir|bitte|dazu)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    return " ".join(cleaned.split())


def _first_day_number(text: str) -> int | None:
    for number in range(1, 15):
        if f"tag {number}" in text or f"day {number}" in text:
            return number
    return None


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value)]
