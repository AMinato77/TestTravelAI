from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from app.agents.agentic_quality_agent import run_agentic_quality_review
from app.agents.agentic_tool_agent import run_agentic_tool_workflow
from dotenv import load_dotenv

from app.agents.activity_evaluation_agent import evaluate_activities
from app.agents.destination_agent import resolve_destination
from app.agents.explanation_agent import explain_travel_plan
from app.agents.planning_agent import plan_itinerary
from app.agents.preference_agent import extract_preferences
from app.models.activity import Activity
from app.models.itinerary import Itinerary, ValidationResult
from app.models.preference_source import PreferenceSource
from app.models.travel_request import TravelRequest
from app.models.user_profile import UserProfile
from app.rag.memory_retrieval import build_memory_query, ingest_preference_sources, retrieve_user_memory
from app.rag.preference_documents import load_preference_sources
from app.rag.user_memory import load_user_profile, update_user_profile
from app.services.destination_normalizer import normalize_destination
from app.services.interest_coverage import coverage_for_itinerary
from app.services.cost_tracker import estimate_tool_cost_report, google_places_trace, openai_llm_trace, trace_to_dict
from app.tools.optimization_tool import optimize_itinerary
from app.tools.openai_runtime import openai_usage_records, reset_openai_usage_records
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


def build_travel_plan(
    user_id: str,
    destination: str,
    days: int,
    budget: float,
    manual_interests: list[str],
    travel_style: str,
    budget_preference: str,
    feedback: str | None = None,
    preference_sources: list[PreferenceSource] | None = None,
    manual_avoid: list[str] | None = None,
    destination_scope: str = "city",
    needs_destination_recommendation: bool = False,
    must_have: list[str] | None = None,
) -> TravelPlanResult:
    load_dotenv()
    reset_openai_usage_records()
    workflow_steps = ["Started travel planning workflow."]
    request_hints = _merge_unique(must_have or [])
    must_have: list[str] = []
    original_destination = destination
    request_for_destination = TravelRequest(
        destination=destination,
        destination_scope=destination_scope,
        needs_destination_recommendation=needs_destination_recommendation,
        duration_days=days,
        budget=budget,
        interests=manual_interests,
        must_have=must_have,
        avoid=manual_avoid or [],
        travel_style=travel_style,
    )
    destination_decision = resolve_destination(request_for_destination)
    if destination_decision.get("changed"):
        workflow_steps.append(
            f"Destination Decision Agent selected {destination_decision['destination']} "
            f"for the broader request '{original_destination}'."
        )
    elif destination_decision.get("summary"):
        workflow_steps.append(f"Destination Decision Agent: {destination_decision['summary']}")
    destination = normalize_destination(str(destination_decision.get("destination") or destination))
    if original_destination.strip() and destination != original_destination.strip():
        workflow_steps.append(f"Normalized destination '{original_destination}' to '{destination}'.")

    # 1. Load old user memory and combine it with the current form input.
    memory_profile = load_user_profile(user_id)
    workflow_steps.append(f"Loaded saved ChromaDB profile memory for user_id={user_id}.")
    new_sources = preference_sources or []
    current_request_interests = _merge_unique(manual_interests)
    if current_request_interests:
        manual_interests = current_request_interests
        workflow_steps.append("Current request interests override stored profile interests for this planning run.")
    elif new_sources:
        manual_interests = []
        workflow_steps.append(
            "No explicit current interests found; using new preference sources instead of old profile interests."
        )
    else:
        manual_interests = memory_profile.merged_interests(manual_interests)
        workflow_steps.append("No explicit current interests found; using stored profile interests for this planning run.")
    manual_avoid = _merge_unique(memory_profile.avoid, manual_avoid or [])

    # 2. Embed new uploads and retrieve relevant chunks with ChromaDB RAG.
    saved_sources = load_preference_sources(user_id)
    all_sources = [*saved_sources, *new_sources]
    workflow_steps.append(
        f"Loaded {len(saved_sources)} stored preference-memory chunk(s) and {len(new_sources)} new source(s)."
    )
    memory_context: list[PreferenceSource] = []
    if new_sources:
        try:
            chunk_count = ingest_preference_sources(user_id, new_sources)
            workflow_steps.append(f"Stored {chunk_count} new embedded user-memory chunk(s) in ChromaDB.")
        except Exception as exc:
            workflow_steps.append(f"New memory sources were not embedded because ChromaDB/embeddings failed: {exc}")
    if _profile_has_memory(memory_profile) or all_sources or new_sources:
        try:
            memory_query = build_memory_query(
                destination=destination,
                interests=manual_interests,
                avoid=manual_avoid,
                travel_style=travel_style,
            )
            retrieved_memory = retrieve_user_memory(user_id, memory_query)
            memory_context = [memory.source for memory in retrieved_memory]
            workflow_steps.append(f"ChromaDB RAG returned {len(memory_context)} relevant memory chunk(s).")
        except Exception as exc:
            workflow_steps.append(f"Memory RAG skipped because ChromaDB/embeddings failed: {exc}")
    if current_request_interests:
        preference_context = new_sources
    elif new_sources:
        preference_context = new_sources
    else:
        preference_context = memory_context or all_sources

    # 3. Let GPT turn uploaded notes, ratings, and form values into a profile.
    extracted_profile = extract_preferences(
        manual_interests=manual_interests,
        travel_style=travel_style,
        budget_preference=budget_preference,
        preference_sources=preference_context,
    )
    workflow_steps.append("Preference Agent created the current user profile.")
    profile = update_user_profile(
        existing=memory_profile,
        extracted=extracted_profile,
        destination=destination,
        manual_interests=manual_interests,
        manual_avoid=manual_avoid,
        feedback=feedback,
        uploaded_sources=[source.name for source in new_sources],
        replace_existing_interests=bool(new_sources and not current_request_interests),
    )
    workflow_steps.append("Saved updated user profile as embedded ChromaDB memory.")
    interests = profile.merged_interests(manual_interests)

    # 4. Get real place candidates from Google Places.
    search_interests = _merge_unique(interests, _supporting_search_interests(request_hints, profile.avoid))
    external_activities, places_metadata = search_places_with_metadata(destination, search_interests, avoid=profile.avoid)
    workflow_steps.append(
        f"Google Places searched interest categories: {', '.join(search_interests[:10])}."
    )
    agentic_tool_workflow = run_agentic_tool_workflow(
        destination=destination,
        days=days,
        activities=external_activities,
        interests=interests,
        request_hints=request_hints,
        budget=budget,
        profile=profile,
    )
    if agentic_tool_workflow.get("enabled"):
        workflow_steps.append("OpenAI Agents SDK Tool Workflow Agent inspected Places and budget quality.")
    else:
        workflow_steps.append(agentic_tool_workflow.get("summary", "Internal tool workflow completed."))

    tool_traces = []
    tool_traces.append(
        trace_to_dict(
            google_places_trace(
                query_count=int(places_metadata.get("query_count") or 0),
                cache_hits=int(places_metadata.get("cache_hits") or 0),
            )
        )
    )
    workflow_steps.append(f"Google Places returned {len(external_activities)} external place candidate(s).")
    activities_before_filter = _deduplicate_activities(external_activities)
    activities, hard_removed_activities = _split_avoided_activities(activities_before_filter, profile.avoid)
    if hard_removed_activities:
        workflow_steps.append(
            f"Removed {len(hard_removed_activities)} activity candidate(s) because of user avoid preferences."
        )
    # 5. Let GPT judge whether the candidate activities really fit the user.
    evaluated_activities, activity_evaluation = evaluate_activities(
        destination=destination,
        activities=activities,
        profile=profile,
        budget=budget,
        constraints={"avoid": profile.avoid, "duration_days": days},
    )
    if evaluated_activities:
        activities = evaluated_activities
    if hard_removed_activities:
        activity_evaluation["removed"] = [
            *_removed_activity_payload(hard_removed_activities),
            *(activity_evaluation.get("removed") or []),
        ]
    workflow_steps.append(
        f"Activity Evaluation Agent kept {len(activities)} candidate(s) and removed "
        f"{len(activity_evaluation.get('removed', []))} weak match(es)."
    )
    # 6. Get weather, create a plan, validate it, and optimize if needed.
    weather = get_weather(destination, days=days)
    workflow_steps.append("Weather tool returned travel weather context.")
    constraints = {
        "avoid": profile.avoid,
        "destination_decision": destination_decision,
    }
    itinerary = plan_itinerary(destination, days, budget, activities, weather, profile, constraints=constraints)
    workflow_steps.append("Planning Agent generated the first itinerary.")
    repair_notes = _enforce_hard_activity_constraints(itinerary, profile.avoid)
    if repair_notes:
        workflow_steps.append(f"Hard constraint guard repaired the first itinerary: {'; '.join(repair_notes)}")
    validation = validate_itinerary(itinerary, budget, weather, profile, constraints=constraints)
    initial_itinerary = deepcopy(itinerary)
    initial_validation = deepcopy(validation)
    workflow_steps.append(f"Validation Agent found {len(validation.issues)} issue(s).")
    optimized = False
    for attempt in range(1, 4):
        if validation.ok:
            break
        previous_signature = _validation_signature(validation)
        itinerary = optimize_itinerary(itinerary, activities, budget, weather, profile, constraints=constraints)
        repair_notes = _enforce_hard_activity_constraints(itinerary, profile.avoid)
        validation = validate_itinerary(itinerary, budget, weather, profile, constraints=constraints)
        optimized = True
        workflow_steps.append(
            f"Optimization Agent adjusted the itinerary and validation ran again (attempt {attempt})."
        )
        if repair_notes:
            workflow_steps.append(f"Hard constraint guard repaired optimizer output: {'; '.join(repair_notes)}")
        if _validation_signature(validation) == previous_signature:
            workflow_steps.append("Optimization stopped because remaining issues could not be changed by available tools.")
            break

    final_interest_coverage = coverage_for_itinerary(itinerary, interests)
    if final_interest_coverage.get("missing"):
        workflow_steps.append(
            "Soft interest coverage gaps in final plan: "
            f"{', '.join(final_interest_coverage['missing'])}."
        )
    else:
        workflow_steps.append("Soft interest coverage check covered all requested interests represented in the profile.")

    # 7. Run an Agents SDK quality review before the final explanation.
    agentic_quality_review = run_agentic_quality_review(
        itinerary=itinerary,
        budget=budget,
        profile=profile,
        validation=validation,
    )
    if agentic_quality_review.get("enabled"):
        workflow_steps.append("OpenAI Agents SDK Quality Review Agent assessed budget and validation quality.")
    else:
        workflow_steps.append(agentic_quality_review.get("summary", "OpenAI Agents SDK Quality Review Agent skipped."))

    # 8. Let GPT explain the final result for the UI.
    explanation = explain_travel_plan(
        itinerary=itinerary,
        profile=profile,
        weather=weather,
        activities=activities,
        validation=validation,
        optimized=optimized,
        budget=budget,
    )
    workflow_steps.append("Explanation Agent generated the final AI explanation.")
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


def _profile_has_memory(profile: UserProfile) -> bool:
    return bool(
        profile.interests
        or profile.avoid
        or profile.past_destinations
        or profile.feedback_history
        or profile.uploaded_sources
        or profile.source_notes
    )


def _merge_unique(*groups: list[str]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            normalized = str(value).strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            values.append(normalized)
    return values


def _supporting_search_interests(request_hints: list[str], avoid: list[str]) -> list[str]:
    support: list[str] = []
    hint_text = " ".join(request_hints).lower()
    if any(term in hint_text for term in ["shop", "store", "shopping", "market", "markt", "kaufen", "einkaufen"]):
        support.append("shopping")
    if any(term in hint_text for term in ["real madrid", "bernab", "football", "stadium", "bullfighting", "toros"]):
        support.append("sport")
    return support


def _validation_signature(validation: ValidationResult) -> tuple:
    return tuple(
        sorted(
            (
                issue.severity,
                issue.issue_type,
                issue.day,
                issue.activity,
                issue.message,
            )
            for issue in validation.issues
        )
    )


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
        {
            "name": activity.name,
            "category": activity.category,
            "source": activity.source,
            "score": 0,
            "reason": "Removed before planning because it conflicts with user avoid preferences.",
        }
        for activity in activities
    ]


def _activity_conflicts_with_avoid(activity: Activity, avoid: list[str]) -> bool:
    haystack = f"{activity.name} {activity.category} {activity.description}".lower()
    avoid_text = " ".join(avoid).lower()
    category = activity.category.strip().lower()
    normalized_avoid = {term.strip().lower() for term in avoid if term.strip()}

    if category in normalized_avoid:
        return True
    if category in {"culture", "history", "architecture", "photography"} and any(
        term in normalized_avoid for term in {"culture", "history", "sightseeing"}
    ):
        return True
    if any(term in avoid_text for term in ["food", "restaurant", "cafe"]):
        return activity.category == "food" or any(
            term in haystack for term in ["restaurant", "food", "cafe", "café", "creperie", "crêperie"]
        )
    if any(term in avoid_text for term in ["museum", "museums", "museen"]):
        return "museum" in haystack or "museen" in haystack or activity.category == "museum"
    if any(term in avoid_text for term in ["nightlife", "club"]):
        return activity.category == "nightlife" or any(term in haystack for term in ["club", "nightlife", "bar"])
    return any(term and term in haystack for term in avoid)


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
        if len(repaired) != len(day.activities):
            day.activities = repaired

    if removed_avoid:
        notes.append(f"removed {removed_avoid} avoid-conflicting activity candidate(s)")
    if removed_duplicates:
        notes.append(f"removed {removed_duplicates} duplicate activity instance(s)")
    return notes
