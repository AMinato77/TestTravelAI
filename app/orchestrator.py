from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from dotenv import load_dotenv

from app.agents.activity_evaluation_agent import evaluate_activities
from app.agents.agentic_quality_agent import run_agentic_quality_review
from app.agents.agentic_tool_agent import run_agentic_tool_workflow
from app.agents.destination_agent import resolve_destination
from app.agents.explanation_agent import explain_travel_plan
from app.agents.planning_agent import plan_itinerary
from app.agents.preference_agent import extract_preferences
from app.agents.query_planning_agent import PlaceQuery, plan_place_queries
from app.models.activity import Activity
from app.models.itinerary import Itinerary, ValidationResult
from app.models.preference_source import PreferenceSource
from app.models.travel_request import TravelRequest
from app.models.user_profile import UserProfile
from app.rag.memory_retrieval import build_memory_query, ingest_preference_sources, retrieve_user_memory
from app.rag.preference_documents import load_preference_sources
from app.rag.user_memory import load_user_profile, update_user_profile
from app.services.cost_tracker import estimate_tool_cost_report, google_places_trace, openai_llm_trace, trace_to_dict
from app.services.destination_normalizer import normalize_destination
from app.tools.openai_runtime import openai_usage_records, reset_openai_usage_records
from app.tools.optimization_tool import optimize_itinerary
from app.tools.places_tool import search_places_with_metadata
from app.tools.validation_tool import validate_itinerary
from app.tools.weather_tool import get_weather


@dataclass(slots=True)
class TravelPlanResult:
    profile: UserProfile
    activities: list[Activity]
    weather: dict
    itinerary: Itinerary
    validation: ValidationResult
    initial_itinerary: Itinerary
    initial_validation: ValidationResult
    optimized: bool
    loaded_memory: UserProfile
    workflow_steps: list[str]
    explanation: dict
    activity_evaluation: dict
    memory_context: list[PreferenceSource]
    agentic_quality_review: dict
    agentic_tool_workflow: dict
    cost_report: dict
    place_queries: list[PlaceQuery]
    query_planning: dict


def build_travel_plan(
    user_id: str,
    destination: str,
    days: int,
    budget: float,
    travel_style: str = "balanced",
    budget_preference: str = "medium",
    feedback: str | None = None,
    preference_sources: list[PreferenceSource] | None = None,
    manual_avoid: list[str] | None = None,
    destination_scope: str = "city",
    needs_destination_recommendation: bool = False,
    must_have: list[str] | None = None,
    interest_tags: list[str] | None = None,
    query_hints: list[str] | None = None,
) -> TravelPlanResult:
    load_dotenv()
    reset_openai_usage_records()
    workflow_steps = ["Started adaptive travel planning workflow."]

    request = TravelRequest(
        destination=destination,
        destination_scope=destination_scope,
        needs_destination_recommendation=needs_destination_recommendation,
        duration_days=days,
        budget=budget,
        must_have=_merge_unique(must_have or []),
        avoid=_merge_unique(manual_avoid or []),
        interest_tags=_merge_unique(interest_tags or []),
        query_hints=_merge_unique(query_hints or [], must_have or []),
        travel_style=travel_style,
    )

    destination_decision = resolve_destination(request)
    original_destination = request.destination
    request.destination = normalize_destination(str(destination_decision.get("destination") or request.destination))
    if destination_decision.get("changed"):
        workflow_steps.append(f"Destination Decision Agent selected {request.destination} for '{original_destination}'.")
    else:
        workflow_steps.append(destination_decision.get("summary", "Destination Decision Agent kept the requested destination."))

    memory_profile = load_user_profile(user_id)
    workflow_steps.append(f"Loaded ChromaDB profile memory for user_id={user_id}.")

    new_sources = preference_sources or []
    saved_sources = load_preference_sources(user_id)
    all_sources = [*saved_sources, *new_sources]
    workflow_steps.append(f"Loaded {len(saved_sources)} stored memory chunk(s) and {len(new_sources)} new source(s).")

    if new_sources:
        try:
            chunk_count = ingest_preference_sources(user_id, new_sources)
            workflow_steps.append(f"Stored {chunk_count} new embedded memory chunk(s) in ChromaDB.")
        except Exception as exc:
            workflow_steps.append(f"New memory sources were not embedded because ChromaDB/embeddings failed: {exc}")

    memory_context: list[PreferenceSource] = []
    try:
        memory_query = build_memory_query(
            destination=request.destination,
            query_terms=request.query_hints or request.must_have or request.interest_tags,
            avoid=request.avoid,
            travel_style=request.travel_style,
        )
        retrieved_memory = retrieve_user_memory(user_id, memory_query)
        memory_context = [memory.source for memory in retrieved_memory]
        workflow_steps.append(f"ChromaDB semantic retrieval returned {len(memory_context)} memory chunk(s).")
    except Exception as exc:
        workflow_steps.append(f"Memory RAG skipped because ChromaDB/embeddings failed: {exc}")

    preference_context = [*memory_context, *new_sources] or all_sources
    extracted_profile = extract_preferences(
        request=request,
        budget_preference=budget_preference,
        preference_sources=preference_context,
    )
    workflow_steps.append("Preference Agent summarized natural-language memory for query planning.")

    profile = update_user_profile(
        existing=memory_profile,
        extracted=extracted_profile,
        destination=request.destination,
        current_interest_tags=request.interest_tags,
        manual_avoid=request.avoid,
        feedback=feedback,
        uploaded_sources=[source.name for source in new_sources],
        replace_existing_tags=bool(request.interest_tags),
    )
    workflow_steps.append("Saved updated user profile as embedded ChromaDB memory.")

    place_queries, query_planning = plan_place_queries(request, memory_context)
    workflow_steps.append(f"Query Planning Agent produced {len(place_queries)} concrete Google Places query/queries.")

    external_activities, places_metadata = search_places_with_metadata(
        destination=request.destination,
        queries=place_queries,
        avoid=profile.avoid,
    )
    workflow_steps.append(f"Google Places returned {len(external_activities)} candidate(s).")

    activities_before_filter = _deduplicate_activities(external_activities)
    activities, hard_removed_activities = _split_avoided_activities(activities_before_filter, profile.avoid)
    if hard_removed_activities:
        workflow_steps.append(f"Removed {len(hard_removed_activities)} candidate(s) because of avoid constraints.")

    constraints = {
        "destination": request.destination,
        "must_have": request.must_have,
        "query_hints": request.query_hints,
        "avoid": profile.avoid,
        "destination_decision": destination_decision,
    }
    evaluated_activities, activity_evaluation = evaluate_activities(
        destination=request.destination,
        activities=activities,
        profile=profile,
        budget=request.budget,
        constraints={**constraints, "duration_days": request.duration_days},
    )
    if evaluated_activities:
        activities = evaluated_activities
    if hard_removed_activities:
        activity_evaluation["removed"] = [
            *_removed_activity_payload(hard_removed_activities),
            *(activity_evaluation.get("removed") or []),
        ]
    workflow_steps.append(
        f"Activity Evaluation Agent kept {len(activities)} candidate(s) and removed {len(activity_evaluation.get('removed', []))} weak match(es)."
    )

    agentic_tool_workflow = run_agentic_tool_workflow(
        destination=request.destination,
        days=request.duration_days,
        activities=activities,
        must_have=request.must_have,
        query_hints=request.query_hints,
        budget=request.budget,
        profile=profile,
    )
    workflow_steps.append(agentic_tool_workflow.get("summary", "Internal tool workflow inspected candidate quality."))

    weather = get_weather(request.destination, days=request.duration_days)
    workflow_steps.append("Weather tool returned travel weather context.")

    itinerary = plan_itinerary(request.destination, request.duration_days, request.budget, activities, weather, profile, constraints=constraints)
    workflow_steps.append("Planning Agent generated the first itinerary.")
    repair_notes = _enforce_hard_activity_constraints(itinerary, profile.avoid)
    if repair_notes:
        workflow_steps.append(f"Hard constraint guard repaired the first itinerary: {'; '.join(repair_notes)}")

    validation = validate_itinerary(itinerary, request.budget, weather, profile, constraints=constraints)
    initial_itinerary = deepcopy(itinerary)
    initial_validation = deepcopy(validation)
    workflow_steps.append(f"Validation found {len(validation.issues)} issue(s), including semantic request checks.")

    optimized = False
    for attempt in range(1, 4):
        if validation.ok:
            break
        previous_signature = _validation_signature(validation)
        itinerary = optimize_itinerary(itinerary, activities, request.budget, weather, profile, constraints=constraints)
        repair_notes = _enforce_hard_activity_constraints(itinerary, profile.avoid)
        validation = validate_itinerary(itinerary, request.budget, weather, profile, constraints=constraints)
        optimized = True
        workflow_steps.append(f"Optimization Agent adjusted the itinerary and validation ran again (attempt {attempt}).")
        if repair_notes:
            workflow_steps.append(f"Hard constraint guard repaired optimizer output: {'; '.join(repair_notes)}")
        if _validation_signature(validation) == previous_signature:
            workflow_steps.append("Optimization stopped because remaining issues could not be changed by available tools.")
            break

    agentic_quality_review = run_agentic_quality_review(itinerary=itinerary, budget=request.budget, profile=profile, validation=validation)
    workflow_steps.append(agentic_quality_review.get("summary", "Quality review completed."))

    explanation = explain_travel_plan(
        itinerary=itinerary,
        profile=profile,
        weather=weather,
        activities=activities,
        validation=validation,
        optimized=optimized,
        budget=request.budget,
    )
    workflow_steps.append("Explanation Agent generated the final explanation.")

    tool_traces = [
        trace_to_dict(
            google_places_trace(
                query_count=int(places_metadata.get("query_count") or 0),
                cache_hits=int(places_metadata.get("cache_hits") or 0),
            )
        )
    ]
    for record in openai_usage_records():
        tool_traces.append(
            trace_to_dict(
                openai_llm_trace(
                    name=record.get("name") or "openai_llm_call",
                    model=record.get("model") or "gpt-5-nano",
                    input_tokens=int(record.get("input_tokens") or 0),
                    output_tokens=int(record.get("output_tokens") or 0),
                )
            )
        )
    cost_report = estimate_tool_cost_report(tool_traces)

    return TravelPlanResult(
        profile=profile,
        activities=activities,
        weather=weather,
        itinerary=itinerary,
        validation=validation,
        initial_itinerary=initial_itinerary,
        initial_validation=initial_validation,
        optimized=optimized,
        loaded_memory=memory_profile,
        workflow_steps=workflow_steps,
        explanation=explanation,
        activity_evaluation=activity_evaluation,
        memory_context=memory_context,
        agentic_quality_review=agentic_quality_review,
        agentic_tool_workflow=agentic_tool_workflow,
        cost_report=cost_report,
        place_queries=place_queries,
        query_planning=query_planning,
    )


def _deduplicate_activities(activities: list[Activity]) -> list[Activity]:
    seen: set[str] = set()
    unique: list[Activity] = []
    for activity in activities:
        key = activity.name.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(activity)
    return unique


def _merge_unique(*groups: list[str]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            cleaned = " ".join(str(value).strip().split())
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            values.append(cleaned)
    return values


def _split_avoided_activities(activities: list[Activity], avoid: list[str]) -> tuple[list[Activity], list[Activity]]:
    if not avoid:
        return activities, []
    kept: list[Activity] = []
    removed: list[Activity] = []
    for activity in activities:
        if _activity_conflicts_with_avoid(activity, avoid):
            removed.append(activity)
        else:
            kept.append(activity)
    return kept, removed


def _removed_activity_payload(activities: list[Activity]) -> list[dict]:
    return [
        {"name": activity.name, "category": activity.category, "source": activity.source, "score": 0, "reason": "Removed because it conflicts with avoid preferences."}
        for activity in activities
    ]


def _activity_conflicts_with_avoid(activity: Activity, avoid: list[str]) -> bool:
    haystack = f"{activity.name} {activity.category} {activity.description}".lower()
    return any(term.strip().lower() and term.strip().lower() in haystack for term in avoid)


def _enforce_hard_activity_constraints(itinerary: Itinerary, avoid: list[str]) -> list[str]:
    notes: list[str] = []
    used: set[str] = set()
    removed_avoid = 0
    removed_duplicates = 0
    for day in itinerary.days:
        repaired: list[Activity] = []
        for activity in day.activities:
            key = activity.name.strip().lower()
            if _activity_conflicts_with_avoid(activity, avoid):
                removed_avoid += 1
                continue
            if key in used:
                removed_duplicates += 1
                continue
            used.add(key)
            repaired.append(activity)
        day.activities = repaired
    if removed_avoid:
        notes.append(f"removed {removed_avoid} avoid-conflicting activity candidate(s)")
    if removed_duplicates:
        notes.append(f"removed {removed_duplicates} duplicate activity instance(s)")
    return notes


def _validation_signature(validation: ValidationResult) -> tuple:
    return tuple(
        sorted((issue.severity, issue.issue_type, issue.day, issue.activity, issue.message) for issue in validation.issues)
    )
