from __future__ import annotations

from app.models.activity import Activity
from app.models.itinerary import Itinerary, ValidationResult
from app.models.user_profile import UserProfile
from app.services.budget_strategy import budget_utilization, target_budget_range
from app.tools.openai_runtime import demo_fallback_enabled, generate_json


def explain_travel_plan(
    itinerary: Itinerary,
    profile: UserProfile,
    weather: dict,
    activities: list[Activity],
    validation: ValidationResult,
    optimized: bool,
    budget: float | None = None,
) -> dict:
    """Explain why the final travel package fits the user and data context."""
    if demo_fallback_enabled():
        return _demo_explanation(itinerary, profile, weather, activities, validation, optimized, budget)

    budget_quality = _budget_quality_payload(itinerary, profile, budget)
    data = generate_json(
        system_prompt=(
            "You are the Explanation Agent for an adaptive AI travel planner. "
            "Explain the final itinerary in concise German. Return strict JSON with keys: "
            "summary, preference_reasoning, data_sources, validation_result, optimization_result, caveats. "
            "Do not invent APIs or facts. Mention that costs and durations are estimates when relevant."
        ),
        payload={
            "itinerary": _itinerary_payload(itinerary),
            "profile": profile.to_dict(),
            "budget_quality": budget_quality,
            "weather_summary": weather.get("summary"),
            "rain_expected": weather.get("rain_expected"),
            "max_rain_chance": weather.get("max_rain_chance"),
            "activity_sources": sorted({activity.source for activity in activities}),
            "validation": {
                "ok": validation.ok,
                "issues": [
                    {
                        "severity": issue.severity,
                        "type": issue.issue_type,
                        "day": issue.day,
                        "activity": issue.activity,
                        "message": issue.message,
                    }
                    for issue in validation.issues
                ],
            },
            "optimized": optimized,
        },
        model_env="OPENAI_EXPLANATION_MODEL",
    )
    return {
        "summary": data.get("summary", ""),
        "preference_reasoning": _as_list(data.get("preference_reasoning")),
        "data_sources": _as_list(data.get("data_sources")),
        "validation_result": data.get("validation_result", ""),
        "optimization_result": data.get("optimization_result", ""),
        "caveats": _as_list(data.get("caveats")),
    }


def _demo_explanation(
    itinerary: Itinerary,
    profile: UserProfile,
    weather: dict,
    activities: list[Activity],
    validation: ValidationResult,
    optimized: bool,
    budget: float | None,
) -> dict:
    return {
        "summary": (
            f"Der Plan fuer {itinerary.destination} wurde anhand von Profil, Wetter, "
            "gefundenen Aktivitaeten und Budgetregeln erstellt."
        ),
        "preference_reasoning": [
            f"Beruecksichtigte Praeferenzen: {', '.join(profile.preference_notes or profile.interest_tags) or 'keine'}",
            f"Reisestil: {profile.travel_style}",
            f"Budgetpraeferenz: {profile.budget_preference}",
            _budget_quality_sentence(itinerary, profile, budget),
        ],
        "data_sources": [
            f"Aktivitaeten aus: {', '.join(sorted({activity.source for activity in activities}))}",
            weather.get("summary", "Wetterdaten wurden geladen."),
        ],
        "validation_result": "Der finale Plan ist valide." if validation.ok else "Der finale Plan hat offene Hinweise.",
        "optimization_result": "Der Plan wurde optimiert." if optimized else "Keine Optimierung war noetig.",
        "caveats": ["Kosten und Dauer sind geschaetzte Werte."],
    }


def _itinerary_payload(itinerary: Itinerary) -> dict:
    return {
        "destination": itinerary.destination,
        "total_cost": itinerary.total_cost,
        "currency": itinerary.currency,
        "days": [
            {
                "day": day.day,
                "total_cost": day.total_cost,
                "total_duration_hours": day.total_duration_hours,
                "activities": [
                    {
                        "name": activity.name,
                        "category": activity.category,
                        "cost": activity.cost,
                        "duration_hours": activity.duration_hours,
                        "indoor": activity.indoor,
                        "source": activity.source,
                    }
                    for activity in day.activities
                ],
                "notes": day.notes,
            }
            for day in itinerary.days
        ],
    }


def _budget_quality_payload(itinerary: Itinerary, profile: UserProfile, budget: float | None) -> dict:
    if not budget:
        return {
            "planned_cost": itinerary.total_cost,
            "currency": itinerary.currency,
        }
    target_min, target_max = target_budget_range(budget, profile)
    return {
        "available_budget": budget,
        "planned_cost": itinerary.total_cost,
        "utilization": round(budget_utilization(itinerary, budget), 3),
        "target_min": round(target_min, 2),
        "target_max": round(target_max, 2),
        "currency": itinerary.currency,
    }


def _budget_quality_sentence(itinerary: Itinerary, profile: UserProfile, budget: float | None) -> str:
    if not budget:
        return f"Geplante Kosten: {itinerary.total_cost:g} {itinerary.currency}"
    target_min, target_max = target_budget_range(budget, profile)
    return (
        f"Budgetauslastung: {itinerary.total_cost:g}/{budget:g} {itinerary.currency}; "
        f"Zielspanne: {target_min:g}-{target_max:g} {itinerary.currency}."
    )


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]
