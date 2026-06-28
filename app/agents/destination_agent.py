from __future__ import annotations

from app.models.travel_request import TravelRequest
from app.tools.openai_runtime import demo_fallback_enabled, generate_json


def resolve_destination(request: TravelRequest) -> dict:
    """Resolve country/open requests into a concrete city before tool calls."""
    destination = request.destination.strip()
    if not request.needs_destination_recommendation and request.destination_scope == "city":
        return {
            "destination": destination,
            "changed": False,
            "summary": "Destination was already a concrete city.",
            "candidates": [],
        }

    if demo_fallback_enabled():
        return _fallback_destination(request)

    try:
        data = generate_json(
            system_prompt=(
                "You are a Destination Decision Agent for a travel planner. "
                "If the user gave a country/region or asked which city is best, choose one concrete city. "
                "Optimize for concrete must-have wishes, query hints, avoid list, travel style, and realistic availability "
                "of activities. Return strict JSON with keys: destination, summary, candidates. "
                "candidates is a list of objects with city, score, reason. "
                "Never choose a fallback city from old memory when the request explicitly names a country."
            ),
            payload={
                "destination": request.destination,
                "destination_scope": request.destination_scope,
                "needs_destination_recommendation": request.needs_destination_recommendation,
                "must_have": request.must_have,
                "query_hints": request.query_hints,
                "interest_tags": request.interest_tags,
                "avoid": request.avoid,
                "duration_days": request.duration_days,
                "budget": {"amount": request.budget, "currency": "EUR"},
                "travel_style": request.travel_style,
            },
            model_env="OPENAI_DESTINATION_MODEL",
        )
    except Exception:
        return _fallback_destination(request)

    selected = str(data.get("destination") or "").strip()
    if not selected:
        return _fallback_destination(request)

    return {
        "destination": selected,
        "changed": selected.lower() != destination.lower(),
        "summary": str(data.get("summary") or "").strip()
        or f"Selected {selected} as the concrete planning city.",
        "candidates": data.get("candidates") if isinstance(data.get("candidates"), list) else [],
    }


def _fallback_destination(request: TravelRequest) -> dict:
    destination = request.destination.strip()
    if request.needs_destination_recommendation or request.destination_scope in {"country", "region", "open"}:
        return {
            "destination": destination,
            "changed": False,
            "summary": (
                "Destination recommendation could not be resolved because the AI decision step failed "
                "or demo mode is active. The app kept the explicit destination instead of using a "
                "hardcoded city fallback."
            ),
            "candidates": [],
            "unresolved": True,
        }
    return {
        "destination": destination,
        "changed": False,
        "summary": "Kept the requested destination because no stronger recommendation rule matched.",
        "candidates": [],
    }
