from __future__ import annotations

from app.models.activity import Activity
from app.models.itinerary import Itinerary
from app.models.travel_request import TravelRequest
from app.services.serialization import itinerary_to_dict
from app.tools.openai_runtime import demo_fallback_enabled, generate_json


def summarize_planned_trip_memory(
    request: TravelRequest,
    itinerary: Itinerary,
    decisions: dict,
    workflow_steps: list[str],
) -> dict:
    payload = {
        "request": {
            "destination": request.destination,
            "duration_days": request.duration_days,
            "budget": request.budget,
            "required_experiences": request.must_have,
            "avoid": request.avoid,
            "travel_style": request.travel_style,
            "use_profile_memory": request.use_profile_memory,
        },
        "final_itinerary": itinerary_to_dict(itinerary),
        "interactive_decisions": _compact_decisions(decisions),
        "interactive_steps": [
            step
            for step in workflow_steps[-20:]
            if "Interactive" in step or "User feedback" in step or "Revision" in step
        ],
    }
    if not demo_fallback_enabled():
        try:
            data = generate_json(
                system_prompt=(
                    "You are a travel memory agent. Create a compact long-term memory from a finalized planned trip. "
                    "Do not store every click as a separate fact. Summarize durable travel patterns, selected highlights, "
                    "known/disliked places, and planning behavior. Be clear when the memory is only a planned trip and "
                    "not a post-trip rating. Return JSON with keys summary, positive_patterns, negative_patterns, "
                    "already_known_places, selected_highlights, confidence. summary must be one concise paragraph."
                ),
                payload=payload,
                model_env="OPENAI_TRIP_MEMORY_MODEL",
            )
            summary = str(data.get("summary") or "").strip()
            if summary:
                already_known = _decision_names(decisions, "already_visited_names")
                selected = _decision_names(decisions, "include_names")
                return {
                    "summary": summary,
                    "positive_patterns": _as_list(data.get("positive_patterns")),
                    "negative_patterns": _as_list(data.get("negative_patterns")),
                    "already_known_places": already_known,
                    "selected_highlights": selected or _as_list(data.get("selected_highlights")),
                    "confidence": str(data.get("confidence") or "planned_trip_signal"),
                }
        except Exception:
            pass
    return _fallback_planned_trip_memory(request, itinerary, decisions)


def _fallback_planned_trip_memory(request: TravelRequest, itinerary: Itinerary, decisions: dict) -> dict:
    selected = _decision_names(decisions, "include_names")
    more_like = _decision_names(decisions, "more_like_names")
    already_known = _decision_names(decisions, "already_visited_names")
    disliked = _decision_names(decisions, "exclude_names")
    planned_names = [activity.name for day in itinerary.days for activity in day.activities]
    categories = _top_categories([activity for day in itinerary.days for activity in day.activities])
    positive_parts = []
    if selected:
        positive_parts.append("explicitly selected " + ", ".join(selected[:6]))
    if more_like:
        positive_parts.append("asked for more like " + ", ".join(more_like[:4]))
    if categories:
        positive_parts.append("planned activities leaned toward " + ", ".join(categories[:4]))
    negative_parts = []
    if already_known:
        negative_parts.append("already knew " + ", ".join(already_known[:6]))
    if disliked:
        negative_parts.append("rejected " + ", ".join(disliked[:6]))
    summary = (
        f"Planned {request.destination} trip: current required experiences were "
        f"{', '.join(request.must_have) or 'not specified'}. Final plan included "
        f"{', '.join(planned_names[:8]) or 'no named activities'}. "
        f"{'; '.join(positive_parts) if positive_parts else 'No strong positive UI pattern was recorded.'}. "
        f"{'; '.join(negative_parts) if negative_parts else 'No post-trip rating was provided.'} "
        "Treat this as a planning preference signal, weaker than completed-trip feedback."
    )
    return {
        "summary": summary,
        "positive_patterns": positive_parts,
        "negative_patterns": negative_parts,
        "already_known_places": already_known,
        "selected_highlights": selected,
        "confidence": "planned_trip_signal",
    }


def _compact_decisions(decisions: dict) -> dict:
    return {
        "include_names": _decision_names(decisions, "include_names"),
        "exclude_names": _decision_names(decisions, "exclude_names"),
        "already_visited_names": _decision_names(decisions, "already_visited_names"),
        "more_like_names": _decision_names(decisions, "more_like_names"),
        "revision_feedback": str(decisions.get("revision_feedback") or "").strip() if isinstance(decisions, dict) else "",
        "answers": decisions.get("answers") if isinstance(decisions, dict) else {},
    }


def _decision_names(decisions: dict, key: str) -> list[str]:
    if not isinstance(decisions, dict):
        return []
    value = decisions.get(key) or []
    if not isinstance(value, list):
        return []
    return [" ".join(str(item).strip().split()) for item in value if str(item).strip()]


def _top_categories(activities: list[Activity]) -> list[str]:
    counts: dict[str, int] = {}
    for activity in activities:
        category = str(activity.category or "").strip().lower()
        if not category:
            continue
        counts[category] = counts.get(category, 0) + 1
    return [name for name, _count in sorted(counts.items(), key=lambda item: item[1], reverse=True)]


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []
