from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from dotenv import load_dotenv

from app.agents.activity_evaluation_agent import evaluate_activities
from app.agents.agentic_quality_agent import run_agentic_quality_review
from app.agents.agentic_planning_workflow import run_agentic_planning_workflow, run_agentic_validation_workflow
from app.agents.destination_agent import resolve_destination
from app.agents.explanation_agent import explain_travel_plan
from app.agents.planning_agent import plan_itinerary
from app.agents.preference_agent import extract_preferences
from app.agents.query_planning_agent import PlaceQuery, plan_place_queries
from app.agents.revision_agent import interpret_revision_feedback
from app.agents.trip_memory_agent import summarize_planned_trip_memory
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
from app.services.wish_matching import activity_covers_wish, activity_intents, activity_text, activity_wish_score, infer_intents, matched_must_have_covers, token_overlap_score
from app.tools.openai_runtime import openai_usage_records, reset_openai_usage_records
from app.tools.optimization_tool import optimize_itinerary
from app.tools.places_tool import search_places_with_metadata
from app.tools.validation_tool import validate_itinerary


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
    revision: dict | None = None


@dataclass(slots=True)
class PreparedPlanContext:
    request: TravelRequest
    profile: UserProfile
    loaded_memory: UserProfile
    activities: list[Activity]
    weather: dict
    workflow_steps: list[str]
    activity_evaluation: dict
    memory_context: list[PreferenceSource]
    agentic_tool_workflow: dict
    place_queries: list[PlaceQuery]
    query_planning: dict
    constraints: dict
    questions: list[dict]


def _retrieve_planning_memory(user_id: str, request: TravelRequest, workflow_steps: list[str]) -> list[PreferenceSource]:
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

        if getattr(request, "use_profile_memory", False):
            broad_query = (
                "Previous travel experiences, liked activities, disliked activities, already visited places, "
                "interactive planning decisions, user profile preferences, travel behavior, and recurring trip patterns."
            )
            broad_memory = [memory.source for memory in retrieve_user_memory(user_id, broad_query, limit=8)]
            before = len(memory_context)
            memory_context = _deduplicate_memory_sources([*memory_context, *broad_memory])
            workflow_steps.append(
                "Profile-memory mode enabled; broad ChromaDB experience retrieval added "
                f"{len(memory_context) - before} memory chunk(s)."
            )
    except Exception as exc:
        workflow_steps.append(f"Memory RAG skipped because ChromaDB/embeddings failed: {exc}")
    return memory_context


def _deduplicate_memory_sources(sources: list[PreferenceSource]) -> list[PreferenceSource]:
    deduped: list[PreferenceSource] = []
    seen: set[tuple[str, str, str]] = set()
    for source in sources:
        key = (
            str(source.source_type).strip().lower(),
            str(source.name).strip().lower(),
            " ".join(str(source.text).split()).lower(),
        )
        if not key[2] or key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def _store_planned_trip_memory(
    user_id: str,
    request: TravelRequest,
    itinerary: Itinerary,
    decisions: dict,
    workflow_steps: list[str],
) -> None:
    try:
        memory = summarize_planned_trip_memory(
            request=request,
            itinerary=itinerary,
            decisions=decisions,
            workflow_steps=workflow_steps,
        )
        text_parts = [
            str(memory.get("summary") or "").strip(),
            _memory_list_line("Positive patterns", memory.get("positive_patterns") or []),
            _memory_list_line("Negative patterns", memory.get("negative_patterns") or []),
            _memory_list_line("Already known places", memory.get("already_known_places") or []),
            _memory_list_line("Selected highlights", memory.get("selected_highlights") or []),
            f"Signal strength: {memory.get('confidence') or 'planned_trip_signal'}",
        ]
        text = "\n".join(part for part in text_parts if part.strip())
        if not text.strip():
            return
        chunk_count = ingest_preference_sources(
            user_id,
            [
                PreferenceSource(
                    source_type="planned_trip_summary",
                    name=f"planned_trip_{itinerary.destination}_{len(itinerary.days)}d",
                    text=text,
                )
            ],
        )
        workflow_steps.append(f"Saved planned trip memory summary to ChromaDB ({chunk_count} chunk(s)).")
    except Exception as exc:
        workflow_steps.append(f"Planned trip memory was not saved because ChromaDB/embeddings failed: {exc}")


def _memory_list_line(label: str, values) -> str:
    if not isinstance(values, list):
        return ""
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    return f"{label}: {', '.join(cleaned[:10])}." if cleaned else ""


def _request_from_revision_inputs(
    original_inputs: dict,
    destination: str,
    budget: float,
    must_have: list[str],
    avoid: list[str],
) -> TravelRequest:
    return TravelRequest(
        destination=str(original_inputs.get("destination") or destination),
        duration_days=int(original_inputs.get("days") or original_inputs.get("duration_days") or 1),
        budget=float(original_inputs.get("budget") or budget),
        must_have=_merge_unique(must_have),
        avoid=_merge_unique(avoid),
        interest_tags=_merge_unique(original_inputs.get("interest_tags") or []),
        query_hints=_merge_unique(original_inputs.get("query_hints") or []),
        travel_style=str(original_inputs.get("travel_style") or "balanced"),
        use_profile_memory=bool(original_inputs.get("use_profile_memory")),
    )


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
    use_profile_memory: bool = False,
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
        use_profile_memory=use_profile_memory,
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

    memory_context = _retrieve_planning_memory(user_id, request, workflow_steps)

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

    agentic_preplan = run_agentic_planning_workflow(
        destination=request.destination,
        days=request.duration_days,
        budget=request.budget,
        profile=profile,
        place_queries=place_queries,
        avoid=profile.avoid,
        must_have=request.must_have,
        query_hints=request.query_hints,
        memory_context=memory_context,
    )
    external_activities = agentic_preplan.activities
    places_metadata = agentic_preplan.places_metadata
    weather = agentic_preplan.weather
    agentic_tool_workflow = agentic_preplan.workflow
    workflow_steps.append(agentic_tool_workflow.get("summary", "Agentic tool workflow inspected planning inputs."))
    for call in (agentic_tool_workflow.get("tool_calls") or [])[:5]:
        workflow_steps.append(f"Tool decision: {call.get('tool')} - {call.get('decision')}")
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

    agentic_tool_workflow["activity_evaluation"] = {
        "kept_candidates": len(activities),
        "removed_candidates": len(activity_evaluation.get("removed", [])),
    }
    workflow_steps.append("Weather tool returned travel weather context through the agentic tool workflow.")

    itinerary = plan_itinerary(request.destination, request.duration_days, request.budget, activities, weather, profile, constraints=constraints)
    workflow_steps.append("Planning Agent generated the first itinerary.")
    coverage_notes = _repair_must_have_coverage(itinerary, activities, request.must_have)
    if coverage_notes:
        workflow_steps.append(f"Coverage guard adjusted the first itinerary: {'; '.join(coverage_notes)}")
    repair_notes = _enforce_hard_activity_constraints(itinerary, profile.avoid)
    if repair_notes:
        workflow_steps.append(f"Hard constraint guard repaired the first itinerary: {'; '.join(repair_notes)}")

    validation_workflow_result = run_agentic_validation_workflow(
        itinerary=itinerary,
        budget=request.budget,
        weather=weather,
        profile=profile,
        constraints=constraints,
    )
    validation = validation_workflow_result.validation
    agentic_tool_workflow["validation_workflow"] = validation_workflow_result.workflow
    agentic_tool_workflow["tool_calls"] = [
        *(agentic_tool_workflow.get("tool_calls") or []),
        *(validation_workflow_result.workflow.get("tool_calls") or []),
    ]
    initial_itinerary = deepcopy(itinerary)
    initial_validation = deepcopy(validation)
    workflow_steps.append(f"Validation found {len(validation.issues)} issue(s), including semantic request checks.")
    workflow_steps.append(validation_workflow_result.workflow.get("summary", "Validation tool workflow completed."))

    optimized = False
    for attempt in range(1, 4):
        if not _needs_optimization(validation):
            break
        previous_signature = _validation_signature(validation)
        itinerary = optimize_itinerary(itinerary, activities, request.budget, weather, profile, constraints=constraints)
        coverage_notes = _repair_must_have_coverage(itinerary, activities, request.must_have)
        repair_notes = _enforce_hard_activity_constraints(itinerary, profile.avoid)
        validation = validate_itinerary(itinerary, request.budget, weather, profile, constraints=constraints)
        optimized = True
        workflow_steps.append(f"Optimization Agent adjusted the itinerary and validation ran again (attempt {attempt}).")
        if coverage_notes:
            workflow_steps.append(f"Coverage guard repaired optimizer output: {'; '.join(coverage_notes)}")
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


def prepare_interactive_plan(
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
    use_profile_memory: bool = False,
) -> PreparedPlanContext:
    """Run the real research/tool phase and pause before final itinerary planning."""

    load_dotenv()
    reset_openai_usage_records()
    workflow_steps = ["Started interactive travel planning preparation."]

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
        use_profile_memory=use_profile_memory,
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

    memory_context = _retrieve_planning_memory(user_id, request, workflow_steps)

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

    agentic_preplan = run_agentic_planning_workflow(
        destination=request.destination,
        days=request.duration_days,
        budget=request.budget,
        profile=profile,
        place_queries=place_queries,
        avoid=profile.avoid,
        must_have=request.must_have,
        query_hints=request.query_hints,
        memory_context=memory_context,
    )
    external_activities = agentic_preplan.activities
    weather = agentic_preplan.weather
    agentic_tool_workflow = agentic_preplan.workflow
    workflow_steps.append(agentic_tool_workflow.get("summary", "Agentic tool workflow inspected planning inputs."))
    for call in (agentic_tool_workflow.get("tool_calls") or [])[:5]:
        workflow_steps.append(f"Tool decision: {call.get('tool')} - {call.get('decision')}")
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
        "interactive": True,
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
    agentic_tool_workflow["activity_evaluation"] = {
        "kept_candidates": len(activities),
        "removed_candidates": len(activity_evaluation.get("removed", [])),
    }
    workflow_steps.append("Interactive pause: candidate preview and user decisions are now available.")

    questions = _build_interactive_questions(request, activities, weather, agentic_tool_workflow)

    return PreparedPlanContext(
        request=request,
        profile=profile,
        loaded_memory=memory_profile,
        activities=activities,
        weather=weather,
        workflow_steps=workflow_steps,
        activity_evaluation=activity_evaluation,
        memory_context=memory_context,
        agentic_tool_workflow=agentic_tool_workflow,
        place_queries=place_queries,
        query_planning=query_planning,
        constraints=constraints,
        questions=questions,
    )


def finalize_interactive_plan(
    prepared_context: PreparedPlanContext,
    user_decisions: dict | None = None,
) -> TravelPlanResult:
    """Build, validate, optimize and explain a plan after the user has shaped the candidate pool."""

    decisions = _merge_interactive_decisions(
        prepared_context.constraints.get("interactive_decisions") if isinstance(prepared_context.constraints, dict) else {},
        user_decisions or {},
    )
    request = deepcopy(prepared_context.request)
    profile = deepcopy(prepared_context.profile)
    workflow_steps = [
        *prepared_context.workflow_steps,
        "User confirmed interactive planning decisions.",
    ]
    activities = _apply_interactive_decisions(prepared_context.activities, profile, decisions)
    if len(activities) != len(prepared_context.activities):
        workflow_steps.append(f"Interactive filter kept {len(activities)} of {len(prepared_context.activities)} candidate(s).")

    constraints = deepcopy(prepared_context.constraints)
    constraints["avoid"] = profile.avoid
    constraints["interactive_decisions"] = _compact_interactive_decisions(decisions)

    itinerary = plan_itinerary(
        request.destination,
        request.duration_days,
        request.budget,
        activities,
        prepared_context.weather,
        profile,
        constraints=constraints,
    )
    workflow_steps.append("Planning Agent generated the final itinerary from interactive decisions.")
    coverage_notes = _repair_must_have_coverage(itinerary, activities, request.must_have)
    if coverage_notes:
        workflow_steps.append(f"Coverage guard adjusted the final itinerary: {'; '.join(coverage_notes)}")
    include_notes = _repair_interactive_includes(itinerary, activities, decisions)
    if include_notes:
        workflow_steps.append(f"Interactive include guard adjusted the final itinerary: {'; '.join(include_notes)}")
    repair_notes = _enforce_hard_activity_constraints(itinerary, profile.avoid)
    if repair_notes:
        workflow_steps.append(f"Hard constraint guard repaired the final itinerary: {'; '.join(repair_notes)}")

    validation_workflow_result = run_agentic_validation_workflow(
        itinerary=itinerary,
        budget=request.budget,
        weather=prepared_context.weather,
        profile=profile,
        constraints=constraints,
    )
    validation = validation_workflow_result.validation
    agentic_tool_workflow = deepcopy(prepared_context.agentic_tool_workflow)
    agentic_tool_workflow["validation_workflow"] = validation_workflow_result.workflow
    agentic_tool_workflow["tool_calls"] = [
        *(agentic_tool_workflow.get("tool_calls") or []),
        *(validation_workflow_result.workflow.get("tool_calls") or []),
    ]
    initial_itinerary = deepcopy(itinerary)
    initial_validation = deepcopy(validation)
    workflow_steps.append(f"Validation found {len(validation.issues)} issue(s), including semantic request checks.")
    workflow_steps.append(validation_workflow_result.workflow.get("summary", "Validation tool workflow completed."))

    optimized = False
    for attempt in range(1, 4):
        if not _needs_optimization(validation):
            break
        previous_signature = _validation_signature(validation)
        itinerary = optimize_itinerary(itinerary, activities, request.budget, prepared_context.weather, profile, constraints=constraints)
        coverage_notes = _repair_must_have_coverage(itinerary, activities, request.must_have)
        include_notes = _repair_interactive_includes(itinerary, activities, decisions)
        repair_notes = _enforce_hard_activity_constraints(itinerary, profile.avoid)
        validation = validate_itinerary(itinerary, request.budget, prepared_context.weather, profile, constraints=constraints)
        optimized = True
        workflow_steps.append(f"Optimization Agent adjusted the itinerary and validation ran again (attempt {attempt}).")
        if coverage_notes:
            workflow_steps.append(f"Coverage guard repaired optimizer output: {'; '.join(coverage_notes)}")
        if include_notes:
            workflow_steps.append(f"Interactive include guard repaired optimizer output: {'; '.join(include_notes)}")
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
        weather=prepared_context.weather,
        activities=activities,
        validation=validation,
        optimized=optimized,
        budget=request.budget,
    )
    workflow_steps.append("Explanation Agent generated the final explanation.")
    _store_planned_trip_memory(
        user_id=profile.user_id,
        request=request,
        itinerary=itinerary,
        decisions=decisions,
        workflow_steps=workflow_steps,
    )

    places_metadata = prepared_context.agentic_tool_workflow.get("places_metadata") or {"query_count": len(prepared_context.place_queries), "cache_hits": 0}
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

    return TravelPlanResult(
        profile=profile,
        activities=activities,
        weather=prepared_context.weather,
        itinerary=itinerary,
        validation=validation,
        initial_itinerary=initial_itinerary,
        initial_validation=initial_validation,
        optimized=optimized,
        loaded_memory=prepared_context.loaded_memory,
        workflow_steps=workflow_steps,
        explanation=explanation,
        activity_evaluation=prepared_context.activity_evaluation,
        memory_context=prepared_context.memory_context,
        agentic_quality_review=agentic_quality_review,
        agentic_tool_workflow=agentic_tool_workflow,
        cost_report=estimate_tool_cost_report(tool_traces),
        place_queries=prepared_context.place_queries,
        query_planning=prepared_context.query_planning,
    )


def expand_interactive_plan(
    prepared_context: PreparedPlanContext,
    user_feedback: str,
    user_decisions: dict | None = None,
    limit: int = 6,
) -> PreparedPlanContext:
    """Add more real Places candidates during the interactive preview phase."""

    feedback = " ".join(str(user_feedback or "").strip().split())
    if not feedback:
        return prepared_context

    decisions = user_decisions or {}
    profile = deepcopy(prepared_context.profile)
    base_activities = _apply_interactive_decisions(prepared_context.activities, profile, decisions)
    profile.preference_notes = _merge_unique(
        profile.preference_notes,
        [f"Interactive user request during candidate preview: {feedback}"],
    )

    request = prepared_context.request
    queries = _select_interactive_expansion_queries(feedback, request.must_have, request.destination)
    if not queries:
        return prepared_context

    activities, places_metadata = search_places_with_metadata(
        destination=request.destination,
        queries=queries,
        avoid=profile.avoid,
    )
    existing_names = {activity.name.strip().lower() for activity in base_activities}
    new_candidates = [
        activity
        for activity in _deduplicate_activities(activities)
        if activity.name.strip().lower() not in existing_names
    ][: max(1, int(limit))]

    if not new_candidates:
        workflow_steps = [
            *prepared_context.workflow_steps,
            f"Interactive expansion for '{feedback}' returned no new candidate(s).",
        ]
        return PreparedPlanContext(
            request=prepared_context.request,
            profile=profile,
            loaded_memory=prepared_context.loaded_memory,
            activities=base_activities,
            weather=prepared_context.weather,
            workflow_steps=workflow_steps,
            activity_evaluation=prepared_context.activity_evaluation,
            memory_context=prepared_context.memory_context,
            agentic_tool_workflow=prepared_context.agentic_tool_workflow,
            place_queries=[*prepared_context.place_queries, *queries],
            query_planning={
                **prepared_context.query_planning,
                "interactive_expansion": {
                    "feedback": feedback,
                    "added_candidates": 0,
                    "places_metadata": places_metadata,
                },
            },
            constraints=prepared_context.constraints,
            questions=prepared_context.questions,
        )

    evaluated, evaluation = evaluate_activities(
        destination=request.destination,
        activities=new_candidates,
        profile=profile,
        budget=request.budget,
        constraints={**prepared_context.constraints, "interactive_feedback": feedback},
    )
    if evaluated:
        new_candidates = evaluated[: max(1, int(limit))]

    activities = _deduplicate_activities([*new_candidates, *base_activities])
    workflow_steps = [
        *prepared_context.workflow_steps,
        f"Interactive expansion for '{feedback}' added {len(new_candidates)} candidate(s).",
    ]
    tool_workflow = deepcopy(prepared_context.agentic_tool_workflow)
    tool_workflow.setdefault("interactive_expansions", [])
    tool_workflow["interactive_expansions"].append(
        {
            "feedback": feedback,
            "query_count": places_metadata.get("query_count"),
            "queries": places_metadata.get("queries"),
            "added_candidates": [activity.name for activity in new_candidates],
        }
    )

    activity_evaluation = deepcopy(prepared_context.activity_evaluation)
    activity_evaluation.setdefault("interactive_expansions", [])
    activity_evaluation["interactive_expansions"].append(evaluation)

    return PreparedPlanContext(
        request=prepared_context.request,
        profile=profile,
        loaded_memory=prepared_context.loaded_memory,
        activities=activities,
        weather=prepared_context.weather,
        workflow_steps=workflow_steps,
        activity_evaluation=activity_evaluation,
        memory_context=prepared_context.memory_context,
        agentic_tool_workflow=tool_workflow,
        place_queries=[*prepared_context.place_queries, *queries],
        query_planning={
            **prepared_context.query_planning,
            "interactive_expansion": {
                "feedback": feedback,
                "added_candidates": len(new_candidates),
                "places_metadata": places_metadata,
            },
        },
        constraints={
            **prepared_context.constraints,
            "avoid": profile.avoid,
            "interactive_feedback": _merge_unique(
                prepared_context.constraints.get("interactive_feedback") or [],
                [feedback],
            ),
            "interactive_decisions": _merge_interactive_decisions(
                prepared_context.constraints.get("interactive_decisions") if isinstance(prepared_context.constraints, dict) else {},
                decisions,
            ),
        },
        questions=_build_interactive_questions(request, activities, prepared_context.weather, tool_workflow),
    )


def revise_travel_plan(
    previous_result: TravelPlanResult,
    feedback: str,
    original_inputs: dict,
) -> TravelPlanResult:
    load_dotenv()
    reset_openai_usage_records()
    itinerary = deepcopy(previous_result.itinerary)
    profile = deepcopy(previous_result.profile)

    must_have = _merge_unique(original_inputs.get("must_have") or [])
    avoid = _merge_unique(profile.avoid, original_inputs.get("avoid") or [])
    revision = interpret_revision_feedback(
        itinerary=itinerary,
        feedback=feedback,
        original_request=original_inputs,
        must_have=must_have,
        avoid=avoid,
    )
    revision["feedback"] = feedback
    revision = _augment_revision_replacement_context(itinerary, revision)
    avoid = _merge_unique(avoid, revision.get("avoid_additions") or [])
    must_have = _merge_unique(must_have, revision.get("must_have_additions") or [])
    query_hints = _merge_unique(revision.get("query_hints") or [], [feedback])
    profile.avoid = avoid

    workflow_steps = [
        *previous_result.workflow_steps,
        f"User feedback for revision: {feedback}",
        f"Revision Agent classified feedback as {revision.get('intent')}: {revision.get('reasoning')}",
    ]
    if revision.get("replacement_requirements"):
        workflow_steps.append(f"Replacement requirements: {', '.join(revision.get('replacement_requirements') or [])}")

    new_queries = _select_revision_queries(query_hints, must_have, itinerary.destination)
    new_activities: list[Activity] = []
    places_metadata = {"query_count": 0, "cache_hits": 0, "queries": []}
    if new_queries:
        try:
            new_activities, places_metadata = search_places_with_metadata(
                destination=itinerary.destination,
                queries=new_queries,
                avoid=avoid,
            )
            workflow_steps.append(f"Revision search returned {len(new_activities)} candidate(s).")
        except Exception as exc:
            workflow_steps.append(f"Revision search failed and used existing candidates only: {exc}")

    activities = _deduplicate_activities([*new_activities, *previous_result.activities])
    change_note = _apply_revision_to_itinerary(itinerary, activities, revision, avoid)
    if change_note:
        workflow_steps.append(change_note)
    else:
        workflow_steps.append("Revision kept the existing itinerary because no targeted replacement was available.")
    cleanup_notes = _replace_revision_avoid_conflicts(itinerary, activities, avoid, revision)
    workflow_steps.extend(cleanup_notes)
    _refresh_revision_cost_notes(itinerary)

    budget = float(original_inputs.get("budget") or itinerary.total_cost or 0)
    constraints = {
        "destination": itinerary.destination,
        "must_have": must_have,
        "query_hints": query_hints,
        "avoid": avoid,
        "revision": revision,
    }
    validation = validate_itinerary(itinerary, budget, previous_result.weather, profile, constraints=constraints)
    workflow_steps.append(f"Validation after revision found {len(validation.issues)} issue(s).")
    explanation = explain_travel_plan(
        itinerary=itinerary,
        profile=profile,
        weather=previous_result.weather,
        activities=activities,
        validation=validation,
        optimized=True,
        budget=budget,
    )
    explanation["optimization_result"] = f"Plan angepasst: {revision.get('revision_instruction') or feedback}"
    _store_planned_trip_memory(
        user_id=profile.user_id,
        request=_request_from_revision_inputs(original_inputs, itinerary.destination, budget, must_have, avoid),
        itinerary=itinerary,
        decisions={"revision_feedback": feedback, "revision": revision},
        workflow_steps=workflow_steps,
    )

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

    return TravelPlanResult(
        profile=profile,
        activities=activities,
        weather=previous_result.weather,
        itinerary=itinerary,
        validation=validation,
        initial_itinerary=previous_result.itinerary,
        initial_validation=previous_result.validation,
        optimized=True,
        loaded_memory=previous_result.loaded_memory,
        workflow_steps=workflow_steps,
        explanation=explanation,
        activity_evaluation=previous_result.activity_evaluation,
        memory_context=previous_result.memory_context,
        agentic_quality_review=previous_result.agentic_quality_review,
        agentic_tool_workflow=previous_result.agentic_tool_workflow,
        cost_report=estimate_tool_cost_report(tool_traces),
        place_queries=[*previous_result.place_queries, *new_queries],
        query_planning={
            "enabled": True,
            "summary": "Revision Agent produced targeted follow-up queries.",
            "revision_places_metadata": places_metadata,
        },
        revision=revision,
    )


def _build_interactive_questions(
    request: TravelRequest,
    activities: list[Activity],
    weather: dict,
    workflow: dict,
) -> list[dict]:
    questions: list[dict] = []
    wish_text = " ".join([*request.must_have, *request.query_hints, *request.interest_tags])
    intents = infer_intents(wish_text)

    if "food" in intents and _count_by_intent(activities, "food") >= 2:
        questions.append(
            {
                "id": "food_style",
                "title": "Essen gewichten",
                "question": "Welche Art von Essens-Erlebnissen soll staerker in den Plan?",
                "options": [
                    {"label": "Ausgewogen", "value": "balanced", "note": "Mische Restaurants, lokale Kueche und besondere Food-Spots."},
                    {"label": "Lokale Kueche", "value": "local_food", "note": "Priorisiere typische lokale Restaurants und traditionelle Gerichte."},
                    {"label": "Street Food & Maerkte", "value": "street_food", "note": "Priorisiere Maerkte, Food-Touren und lockere Essens-Spots."},
                    {"label": "Fine Dining", "value": "fine_dining", "note": "Nutze mehr Budget fuer besondere Restaurants."},
                ],
            }
        )

    if "nature" in intents and _count_by_intent(activities, "nature") >= 2:
        questions.append(
            {
                "id": "nature_style",
                "title": "Natur-Stil",
                "question": "Welche Naturerlebnisse passen besser?",
                "options": [
                    {"label": "Ausgewogen", "value": "balanced", "note": "Mische zentrale Gruenflaechen und Aussichtspunkte."},
                    {"label": "Stadtparks", "value": "city_parks", "note": "Bevorzuge kurze, leicht erreichbare Naturstopps."},
                    {"label": "Aussichtspunkte", "value": "viewpoints", "note": "Bevorzuge Orte mit Panorama und Fotomotiven."},
                    {"label": "Kurzer Ausflug", "value": "short_trip", "note": "Plane eher ein groesseres Naturerlebnis mit etwas Transferzeit."},
                ],
            }
        )

    if request.budget and request.budget > 0:
        budget = workflow.get("budget_assessment") if isinstance(workflow.get("budget_assessment"), dict) else {}
        target_min = budget.get("target_min")
        target_max = budget.get("target_max")
        questions.append(
            {
                "id": "budget_use",
                "title": "Budget nutzen",
                "question": "Wie soll das Budget bei der Planung eingesetzt werden?",
                "options": [
                    {"label": "Ausgewogen", "value": "balanced", "note": f"Plane im Zielbereich {target_min}-{target_max} EUR, falls moeglich."},
                    {"label": "Preisbewusst", "value": "save", "note": "Nutze mehr kostenlose oder guenstige Optionen."},
                    {"label": "Besondere Erlebnisse", "value": "spend", "note": "Nutze mehr Budget fuer Touren, Tickets oder bessere Restaurants."},
                ],
            }
        )

    if bool(weather.get("rain_expected")) or int(weather.get("max_rain_chance") or 0) >= 60:
        questions.insert(
            0,
            {
                "id": "weather_strategy",
                "title": "Wetterstrategie",
                "question": "Es gibt Regenrisiko. Wie soll der Plan damit umgehen?",
                "options": [
                    {"label": "Indoor priorisieren", "value": "indoor", "note": "Bevorzuge Museen, Restaurants und wetterfeste Orte."},
                    {"label": "Outdoor mit Backup", "value": "backup", "note": "Outdoor bleibt drin, aber mit indoorfreundlicher Alternative."},
                    {"label": "Unveraendert", "value": "unchanged", "note": "Plane nach Wunsch, Wetter nur als Hinweis."},
                ],
            },
        )

    missing = []
    coverage = workflow.get("wish_coverage") if isinstance(workflow.get("wish_coverage"), dict) else {}
    if isinstance(coverage.get("missing"), list):
        missing = [str(item) for item in coverage.get("missing") if str(item).strip()]
    if missing:
        questions.insert(
            0,
            {
                "id": "coverage_strategy",
                "title": "Offene Wuensche",
                "question": "Einige Wuensche haben noch wenige oder keine Kandidaten. Wie soll TravelAI reagieren?",
                "options": [
                    {"label": "Gezielt weitersuchen", "value": "search_more", "note": "Der finale Plan priorisiert Ersatzqueries und passende Alternativen."},
                    {"label": "Aehnliche Orte", "value": "nearby", "note": "Nutze thematisch aehnliche Kandidaten aus der vorhandenen Suche."},
                    {"label": "Trotzdem planen", "value": "continue", "note": "Erstelle den Plan mit den vorhandenen Kandidaten."},
                ],
                "context": missing[:4],
            },
        )

    return questions[:3]


def _select_interactive_expansion_queries(feedback: str, must_have: list[str], destination: str) -> list[PlaceQuery]:
    max_queries = _configured_int("TRAVELAI_MAX_INTERACTIVE_EXPANSION_QUERIES", 3, minimum=1, maximum=5)
    hints = _merge_unique(_generic_intent_query_hints(feedback), [feedback], _matched_must_have_for_query(feedback, must_have))
    selected: list[PlaceQuery] = []
    seen: set[str] = set()
    for hint in hints:
        cleaned = _clean_revision_query_hint(hint)
        if not cleaned:
            continue
        if destination and destination.lower() not in cleaned.lower():
            cleaned = f"{destination} {cleaned}"
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            PlaceQuery(
                query=cleaned,
                reason="Interactive user feedback query.",
                source="interactive_feedback",
                must_have=_interactive_query_must_have(cleaned, feedback, must_have, destination),
            )
        )
        if len(selected) >= max_queries:
            break
    return selected


def _interactive_query_must_have(query: str, feedback: str, must_have: list[str], destination: str = "") -> list[str]:
    matched = _matched_must_have_for_query(f"{query} {feedback}", must_have, destination)
    if matched:
        return matched
    feedback_intents = infer_intents(feedback)
    intent_matches = []
    for wish in must_have:
        if _is_gaming_feedback(feedback) and _is_anime_feedback(wish) and not _is_gaming_feedback(wish):
            continue
        if _is_anime_feedback(feedback) and _is_gaming_feedback(wish) and not _is_anime_feedback(wish):
            continue
        if infer_intents(wish) & feedback_intents:
            intent_matches.append(wish)
    if intent_matches:
        return intent_matches
    requirement = _interactive_feedback_requirement(feedback)
    return [requirement] if requirement else []


def _generic_intent_query_hints(feedback: str) -> list[str]:
    intents = infer_intents(feedback)
    tokens = set(_match_tokens(feedback))
    hints: list[str] = []
    gaming = _is_gaming_feedback(feedback)
    anime = _is_anime_feedback(feedback)
    if gaming:
        hints.extend(["gaming shops", "video game stores", "retro game shops", "arcades"])
    if anime:
        hints.extend(["anime shops", "manga stores", "anime merchandise stores", "figure stores"])
    if "food" in intents:
        hints.extend(["local food experiences", "restaurants", "food markets"])
    if "nature" in intents:
        hints.extend(["nature experiences", "scenic viewpoints", "city parks"])
    if "culture" in intents:
        if tokens & {"museum", "museen"}:
            hints.extend(["museums", "must-see museums"])
        if tokens & {"architecture", "architectural", "architektur", "building", "buildings"}:
            hints.extend(["architecture walking tour", "architectural landmarks"])
        if not hints:
            hints.extend(["cultural attractions", "historic landmarks"])
    if "shopping" in intents:
        hints.extend(["shopping streets", "local markets", "specialty stores"])
    if "entertainment" in intents and not (gaming or anime):
        hints.extend(["entertainment experiences", "themed attractions"])
    if "nightlife" in intents:
        hints.extend(["cocktail bars", "nightlife spots"])
    return hints


def _interactive_feedback_requirement(feedback: str) -> str:
    intents = infer_intents(feedback)
    tokens = set(_match_tokens(feedback))
    if _is_gaming_feedback(feedback):
        return "gaming places"
    if _is_anime_feedback(feedback):
        return "anime and manga shops"
    if "food" in intents:
        return "food experiences"
    if "nature" in intents:
        return "nature experiences"
    if tokens & {"architecture", "architectural", "architektur", "building", "buildings"}:
        return "architecture experiences"
    if tokens & {"museum", "museums", "museen"}:
        return "museums"
    if "shopping" in intents:
        return "shopping places"
    if "nightlife" in intents:
        return "nightlife spots"
    if "entertainment" in intents:
        return "entertainment experiences"
    return ""


def _is_gaming_feedback(feedback: str) -> bool:
    tokens = set(_match_tokens(feedback))
    text = str(feedback or "").lower()
    gaming_terms = {
        "gaming",
        "game",
        "games",
        "gamer",
        "arcade",
        "arcades",
        "retro",
        "videogame",
        "videogames",
        "videospiel",
        "videospiele",
        "spiel",
        "spiele",
        "spielhalle",
        "zocken",
    }
    return bool(tokens & gaming_terms) or "video game" in text or "video games" in text


def _is_anime_feedback(feedback: str) -> bool:
    tokens = set(_match_tokens(feedback))
    anime_terms = {
        "anime",
        "manga",
        "mangas",
        "otaku",
        "figure",
        "figures",
        "figur",
        "figuren",
        "merch",
        "merchandise",
    }
    return bool(tokens & anime_terms)


def _count_by_intent(activities: list[Activity], intent: str) -> int:
    return sum(1 for activity in activities if intent in activity_intents(activity))


def _apply_interactive_decisions(
    activities: list[Activity],
    profile: UserProfile,
    decisions: dict,
) -> list[Activity]:
    excluded = _decision_names(decisions, "exclude_names")
    already_visited = _decision_names(decisions, "already_visited_names")
    included = _decision_names(decisions, "include_names")
    more_like = _decision_names(decisions, "more_like_names")
    blocked = excluded | already_visited

    if blocked:
        profile.avoid = _merge_unique(profile.avoid, sorted(blocked))

    notes = _interactive_preference_notes(decisions, activities)
    if notes:
        profile.preference_notes = _merge_unique(profile.preference_notes, notes)

    kept = [activity for activity in activities if activity.name.strip().lower() not in blocked]
    kept.sort(
        key=lambda activity: (
            0 if activity.name.strip().lower() in included else 1,
            0 if activity.name.strip().lower() in more_like else 1,
            activity.name.strip().lower(),
        )
    )
    return kept


def _decision_names(decisions: dict, key: str) -> set[str]:
    value = decisions.get(key) if isinstance(decisions, dict) else []
    if not isinstance(value, list):
        return set()
    return {" ".join(str(item).strip().lower().split()) for item in value if str(item).strip()}


def _interactive_preference_notes(decisions: dict, activities: list[Activity]) -> list[str]:
    notes: list[str] = []
    answers = decisions.get("answers") if isinstance(decisions, dict) else {}
    if isinstance(answers, dict):
        label_map = {
            "food_style": {
                "local_food": "Prefer typical local cuisine and traditional restaurants.",
                "street_food": "Prefer street food, food markets, and casual food experiences.",
                "fine_dining": "Use more budget for premium dining experiences.",
                "balanced": "Use a balanced mix of food experiences.",
            },
            "nature_style": {
                "city_parks": "Prefer central city parks and easy nature stops.",
                "viewpoints": "Prefer scenic viewpoints and photo-friendly landscapes.",
                "short_trip": "Prefer one stronger nature excursion if it fits the schedule.",
                "balanced": "Use a balanced mix of nature experiences.",
            },
            "budget_use": {
                "save": "Plan price-consciously and preserve more budget.",
                "spend": "Use more of the budget for special experiences.",
                "balanced": "Use the budget target range without forcing expensive choices.",
            },
            "weather_strategy": {
                "indoor": "Prioritize indoor activities when weather risk is high.",
                "backup": "Keep outdoor activities only with indoor-friendly alternatives.",
                "unchanged": "Do not over-optimize for weather unless validation flags a conflict.",
            },
            "coverage_strategy": {
                "search_more": "Prioritize concrete coverage for wishes with weak candidate support.",
                "nearby": "Use thematically similar places if exact candidates are limited.",
                "continue": "Proceed with the strongest available candidate pool.",
            },
        }
        for question_id, answer in answers.items():
            note = label_map.get(str(question_id), {}).get(str(answer))
            if note:
                notes.append(note)

    more_like = _decision_names(decisions, "more_like_names")
    if more_like:
        by_name = {activity.name.strip().lower(): activity for activity in activities}
        selected = [by_name[name].name for name in more_like if name in by_name]
        if selected:
            notes.append("Prefer more activities similar to: " + ", ".join(selected[:5]) + ".")

    included = _decision_names(decisions, "include_names")
    if included:
        by_name = {activity.name.strip().lower(): activity for activity in activities}
        selected = [by_name[name].name for name in included if name in by_name]
        if selected:
            notes.append("User explicitly wants to include: " + ", ".join(selected[:5]) + ".")
    return notes


def _compact_interactive_decisions(decisions: dict) -> dict:
    if not isinstance(decisions, dict):
        return {}
    return {
        "answers": decisions.get("answers") if isinstance(decisions.get("answers"), dict) else {},
        "include_names": list(_decision_names(decisions, "include_names"))[:10],
        "exclude_names": list(_decision_names(decisions, "exclude_names"))[:10],
        "already_visited_names": list(_decision_names(decisions, "already_visited_names"))[:10],
        "more_like_names": list(_decision_names(decisions, "more_like_names"))[:10],
    }


def _merge_interactive_decisions(*decision_sets: dict | None) -> dict:
    merged = {
        "answers": {},
        "include_names": [],
        "exclude_names": [],
        "already_visited_names": [],
        "more_like_names": [],
    }
    for decisions in decision_sets:
        if not isinstance(decisions, dict):
            continue
        answers = decisions.get("answers")
        if isinstance(answers, dict):
            merged["answers"].update(answers)
        for key in ("include_names", "exclude_names", "already_visited_names", "more_like_names"):
            merged[key] = _merge_unique(merged[key], decisions.get(key) or [])

    blocked = _decision_names(merged, "exclude_names") | _decision_names(merged, "already_visited_names")
    if blocked:
        merged["include_names"] = [
            name for name in merged["include_names"] if " ".join(str(name).strip().lower().split()) not in blocked
        ]
        merged["more_like_names"] = [
            name for name in merged["more_like_names"] if " ".join(str(name).strip().lower().split()) not in blocked
        ]
    return _compact_interactive_decisions(merged)


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


def _select_revision_queries(query_hints: list[str], must_have: list[str], destination: str) -> list[PlaceQuery]:
    max_queries = _configured_int("TRAVELAI_MAX_REVISION_QUERIES", 3, minimum=1, maximum=8)
    selected: list[PlaceQuery] = []
    seen: set[str] = set()
    for query in query_hints:
        cleaned = _clean_revision_query_hint(str(query))
        if not cleaned:
            continue
        if destination and destination.lower() not in cleaned.lower():
            cleaned = f"{cleaned} {destination}"
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            PlaceQuery(
                query=cleaned,
                reason="Revision feedback query.",
                source="revision_agent",
                must_have=_matched_must_have_for_query(cleaned, must_have),
            )
        )
        if len(selected) >= max_queries:
            break
    return selected


def _clean_revision_query_hint(query: str) -> str:
    import re

    cleaned = " ".join(str(query or "").strip().split())
    if not cleaned:
        return ""
    cleaned = re.sub(r"\banstatt\b.+?\bbitte\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\binstead of\b.+?\bplease\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(gib|gebe|mir|bitte|please|stattdessen|instead)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b(ich|moechte|möchte|will|haette|hätte|gerne|ein|eine|einen|das|die|der|da|noch|mehr|bisschen|etwas|fehlt|fehlen|suche|such|brauche|brauch)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = " ".join(cleaned.split())
    return cleaned or " ".join(str(query or "").strip().split())


def _matched_must_have_for_query(query: str, must_have: list[str], destination: str = "") -> list[str]:
    matched: list[str] = []
    query_intents = infer_intents(query)
    for wish in must_have:
        wish_intents = infer_intents(wish)
        if query_intents and wish_intents and not (query_intents & wish_intents):
            continue
        if _is_gaming_feedback(query) and _is_anime_feedback(wish) and not _is_gaming_feedback(wish):
            continue
        if _is_anime_feedback(query) and _is_gaming_feedback(wish) and not _is_anime_feedback(wish):
            continue
        if token_overlap_score(query.lower(), wish, destination) >= 0.67:
            matched.append(wish)
    return matched


def _configured_int(name: str, fallback: int, minimum: int, maximum: int) -> int:
    import os

    try:
        value = int(os.getenv(name, str(fallback)))
    except (TypeError, ValueError):
        value = fallback
    return max(minimum, min(value, maximum))


def _augment_revision_replacement_context(itinerary: Itinerary, revision: dict) -> dict:
    if str(revision.get("intent") or "") != "replace_activity":
        return revision
    target_terms = _merge_unique(
        [str(revision.get("target_activity") or "")],
        revision.get("avoid_additions") or [],
        [str(revision.get("feedback") or "")],
    )
    target = _find_itinerary_activity(itinerary, target_terms, revision.get("target_day"))
    if not target:
        return revision
    _day, _index, activity = target
    feedback = str(revision.get("feedback") or "")
    requirements = _merge_unique(revision.get("replacement_requirements") or [], [feedback])
    queries = _merge_unique(revision.get("query_hints") or [], [_clean_revision_search_text(feedback, activity)])
    revision["target_activity"] = activity.name
    revision["replacement_requirements"] = requirements
    revision["query_hints"] = queries
    revision["avoid_additions"] = _merge_unique(revision.get("avoid_additions") or [], [activity.name])
    return revision


def _apply_revision_to_itinerary(
    itinerary: Itinerary,
    activities: list[Activity],
    revision: dict,
    avoid: list[str],
) -> str:
    intent = str(revision.get("intent") or "")
    target_terms = _merge_unique(
        [str(revision.get("target_activity") or "")],
        revision.get("avoid_additions") or [],
        [str(revision.get("feedback") or "")],
    )
    target_day = revision.get("target_day")
    used = {activity.name.strip().lower() for day in itinerary.days for activity in day.activities}

    if intent == "reduce_intensity":
        days = [day for day in itinerary.days if not target_day or day.day == target_day]
        if not days and itinerary.days:
            days = [max(itinerary.days, key=lambda item: len(item.activities))]
        for day in days[:1]:
            if len(day.activities) <= 1:
                continue
            removed = day.activities.pop()
            day.notes.append(f"Removed {removed.name} after revision feedback to make the day less packed.")
            return f"Removed {removed.name} from day {day.day} to reduce intensity."
        return ""

    if intent != "reduce_intensity" and target_terms:
        target = _find_itinerary_activity(itinerary, target_terms, target_day)
        if not target and intent in {"replace_activity", "general_revision"}:
            target = _find_itinerary_activity_by_revision_intent(itinerary, revision, target_day)
        if target:
            day, index, activity = target
            replacement = _find_replacement_activity(activity, activities, used, avoid, revision)
            if not replacement:
                day.activities.pop(index)
                day.notes.append(f"Removed {activity.name} after revision feedback; no unused replacement was available.")
                _remove_note_mentions(day, [activity.name, *(revision.get("avoid_additions") or [])])
                return f"Removed {activity.name} from day {day.day}; no replacement was available."
            day.activities[index] = replacement
            _remove_note_mentions(day, [activity.name, *(revision.get("avoid_additions") or [])])
            day.notes.append(f"Added {replacement.name} as replacement after revision feedback.")
            return f"Replaced {activity.name} with {replacement.name} on day {day.day}."

    if intent in {"replace_activity", "general_revision", "change_budget_level"}:
        for day in itinerary.days:
            for index, activity in enumerate(list(day.activities)):
                if not _activity_conflicts_with_avoid(activity, avoid):
                    continue
                replacement = _find_replacement_activity(activity, activities, used, avoid, revision)
                if not replacement:
                    day.activities.pop(index)
                    day.notes.append(f"Removed {activity.name} after revision feedback; no unused replacement was available.")
                    _remove_note_mentions(day, [activity.name, *(revision.get("avoid_additions") or [])])
                    return f"Removed {activity.name} from day {day.day}; no replacement was available."
                day.activities[index] = replacement
                _remove_note_mentions(day, [activity.name, *(revision.get("avoid_additions") or [])])
                day.notes.append(f"Added {replacement.name} as replacement after revision feedback.")
                return f"Replaced {activity.name} with {replacement.name} on day {day.day}."

    if intent == "add_more_similar" or (intent == "general_revision" and _revision_requests_addition(revision)):
        candidate = _find_activity_for_revision_feedback(activities, used, avoid, revision)
        if not candidate or not itinerary.days:
            return ""
        day = min(itinerary.days, key=lambda item: (item.total_duration_hours, len(item.activities)))
        if len(day.activities) >= 4:
            removed = day.activities.pop()
            day.notes.append(f"Removed {removed.name} to make room for a requested similar activity.")
        day.activities.append(candidate)
        day.notes.append(f"Added {candidate.name} after revision feedback.")
        return f"Added {candidate.name} to day {day.day}."

    return ""


def _find_itinerary_activity_by_revision_intent(
    itinerary: Itinerary,
    revision: dict,
    target_day: int | None,
) -> tuple | None:
    text = " ".join(
        [
            str(revision.get("feedback") or ""),
            " ".join(revision.get("replacement_requirements") or []),
            " ".join(revision.get("query_hints") or []),
            str(revision.get("revision_instruction") or ""),
        ]
    )
    desired_intents = infer_intents(text)
    if not desired_intents:
        return None
    for day in itinerary.days:
        if target_day and day.day != target_day:
            continue
        for index, activity in enumerate(day.activities):
            if activity_intents(activity) & desired_intents:
                return day, index, activity
    return None


def _find_replacement_activity(
    original: Activity,
    activities: list[Activity],
    used: set[str],
    avoid: list[str],
    revision: dict | None = None,
) -> Activity | None:
    revision = revision or {}
    desired_tokens = set(_desired_revision_tokens(revision, original))
    allow_old_type_fallback = not desired_tokens or _revision_requests_similar_fallback(revision)
    candidates: list[tuple[float, Activity]] = []
    for activity in activities:
        key = activity.name.strip().lower()
        if key in used or key == original.name.strip().lower():
            continue
        if _activity_conflicts_with_avoid(activity, avoid):
            continue
        score = _replacement_score(original, activity, revision, desired_tokens, allow_old_type_fallback)
        if score > 0:
            candidates.append((score, activity))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]
    same_category = _first_unused_same_category(activities, used, avoid, original.category)
    if same_category:
        return same_category
    if allow_old_type_fallback:
        return _first_unused_activity(activities, used, avoid)
    return None


def _replacement_score(
    original: Activity,
    candidate: Activity,
    revision: dict,
    desired_tokens: set[str],
    allow_old_type_fallback: bool,
) -> float:
    desired_intents = infer_intents(
        " ".join(
            [
                str(revision.get("feedback") or ""),
                " ".join(revision.get("replacement_requirements") or []),
                " ".join(revision.get("query_hints") or []),
                str(revision.get("revision_instruction") or ""),
            ]
        )
    )
    if desired_intents and not (activity_intents(candidate) & desired_intents):
        return 0.0
    candidate_text = _activity_search_text(candidate)
    candidate_tokens = set(_match_tokens(candidate_text))
    desired_overlap = desired_tokens & candidate_tokens
    if desired_tokens:
        score = float(len(desired_overlap) * 5)
        if candidate.category == original.category:
            score += 2.0
        return score

    score = 0.0
    if allow_old_type_fallback and candidate.category == original.category:
        score += 2.0
    return score


def _find_activity_for_revision_feedback(
    activities: list[Activity],
    used: set[str],
    avoid: list[str],
    revision: dict,
) -> Activity | None:
    desired_tokens = set(_desired_revision_tokens(revision, None))
    scored: list[tuple[float, Activity]] = []
    for activity in activities:
        key = activity.name.strip().lower()
        if key in used:
            continue
        if _activity_conflicts_with_avoid(activity, avoid):
            continue
        candidate_tokens = set(_match_tokens(_activity_search_text(activity)))
        score = len(desired_tokens & candidate_tokens) * 4.0
        if not desired_tokens:
            score = 1.0
        if score > 0:
            scored.append((score, activity))
    if not scored:
        return _first_unused_activity(activities, used, avoid)
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _desired_revision_tokens(revision: dict, original: Activity | None) -> list[str]:
    text = " ".join(
        [
            str(revision.get("feedback") or ""),
            " ".join(revision.get("replacement_requirements") or []),
            " ".join(revision.get("must_have_additions") or []),
            " ".join(revision.get("query_hints") or []),
            str(revision.get("revision_instruction") or ""),
        ]
    )
    tokens = _match_tokens(text)
    if original is None:
        return tokens
    original_tokens = set(_match_tokens(f"{original.name} {original.category}"))
    return [token for token in tokens if token not in original_tokens]


def _revision_requests_addition(revision: dict) -> bool:
    text = str(revision.get("feedback") or "").lower()
    return any(marker in text for marker in ["hinzuf", "add ", "another", "weitere", "noch eine", "mehr davon", "more"])


def _revision_requests_similar_fallback(revision: dict) -> bool:
    text = str(revision.get("feedback") or "").lower()
    return any(marker in text for marker in ["alternative", "similar", "aehnlich", "aequivalent", "ersatz"]) and not _revision_requests_addition(revision)


def _clean_revision_search_text(feedback: str, original: Activity) -> str:
    import re

    cleaned = " ".join(str(feedback or "").strip().split())
    for token in re.findall(r"[A-Za-z0-9]+", original.name):
        if len(token) > 2:
            cleaned = re.sub(rf"\b{re.escape(token)}\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b(kenne|kenn|schon|war|ich|das|die|der|den|dem|beim|bei|statt|anstatt|stattdessen|instead|alternative|ersetze|gib|mir|bitte|dazu)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = " ".join(cleaned.split())
    if len(_match_tokens(cleaned)) >= 2:
        return cleaned
    category_queries = {
        "food": "traditional local restaurant",
        "culture": "museum historic cultural attraction",
        "nature": "scenic nature outdoor attraction",
        "shopping": "local shops market stores",
        "entertainment": "entertainment experience",
        "nightlife": "bar nightlife experience",
        "sport": "sport activity venue",
    }
    return category_queries.get(original.category, f"{original.category} local experience")


def _activity_search_text(activity: Activity) -> str:
    return f"{activity.name} {activity.category} {_clean_activity_description_for_prompt(activity.description)}".lower()


def _clean_activity_description_for_prompt(description: str) -> str:
    kept: list[str] = []
    blocked_labels = {"matched query", "matched must-have", "google maps", "website"}
    for part in str(description or "").split("|"):
        cleaned = " ".join(part.strip().split())
        if not cleaned:
            continue
        label = cleaned.split(":", 1)[0].strip().lower() if ":" in cleaned else ""
        if label in blocked_labels:
            continue
        if cleaned.lower().startswith(("http://", "https://")):
            continue
        kept.append(cleaned)
    return " | ".join(kept)[:500]


def _first_unused_activity(activities: list[Activity], used: set[str], avoid: list[str]) -> Activity | None:
    for activity in activities:
        key = activity.name.strip().lower()
        if key in used:
            continue
        if _activity_conflicts_with_avoid(activity, avoid):
            continue
        return activity
    return None


def _first_unused_same_category(
    activities: list[Activity],
    used: set[str],
    avoid: list[str],
    category: str,
) -> Activity | None:
    for activity in activities:
        key = activity.name.strip().lower()
        if key in used:
            continue
        if activity.category != category:
            continue
        if _activity_conflicts_with_avoid(activity, avoid):
            continue
        return activity
    return None


def _replace_revision_avoid_conflicts(
    itinerary: Itinerary,
    activities: list[Activity],
    avoid: list[str],
    revision: dict,
) -> list[str]:
    notes: list[str] = []
    used = {activity.name.strip().lower() for day in itinerary.days for activity in day.activities}
    for day in itinerary.days:
        repaired: list[Activity] = []
        for activity in day.activities:
            if not _activity_conflicts_with_avoid(activity, avoid):
                repaired.append(activity)
                continue
            replacement = _find_replacement_activity(activity, activities, used, avoid, revision)
            if replacement:
                repaired.append(replacement)
                used.add(replacement.name.strip().lower())
                _remove_note_mentions(day, [activity.name, *(revision.get("avoid_additions") or [])])
                day.notes.append(f"Added {replacement.name} as replacement because of revision avoid constraints.")
                notes.append(f"Revision cleanup replaced {activity.name} with {replacement.name} on day {day.day}.")
            else:
                _remove_note_mentions(day, [activity.name, *(revision.get("avoid_additions") or [])])
                day.notes.append(f"Removed {activity.name} because it conflicts with revision avoid constraints.")
                notes.append(f"Revision cleanup removed {activity.name} from day {day.day}.")
        day.activities = repaired
    return notes


def _remove_note_mentions(day, terms: list[str]) -> None:
    cleaned_terms = [str(term).strip().lower() for term in terms if str(term).strip()]
    if not cleaned_terms:
        return
    filtered: list[str] = []
    for note in day.notes:
        note_lower = str(note).lower()
        if any(term and term in note_lower for term in cleaned_terms):
            continue
        filtered.append(note)
    day.notes = filtered


def _refresh_revision_cost_notes(itinerary: Itinerary) -> None:
    stale_markers = [
        "budgetziel",
        "gesamtausgaben geplant",
        "gesamtaktiv",
        "aktuelle aktivkostenerwartung",
        "restbudget",
        "aktualisierte kosten nach anpassung",
    ]
    total = _format_cost_value(itinerary.total_cost)
    for day in itinerary.days:
        fresh_notes: list[str] = []
        for note in day.notes:
            note_text = str(note).strip()
            if not note_text:
                continue
            note_lower = note_text.lower()
            if any(marker in note_lower for marker in stale_markers):
                continue
            fresh_notes.append(note_text)
        day_cost = _format_cost_value(day.total_cost)
        duration = _format_duration_value(day.total_duration_hours)
        fresh_notes.insert(
            0,
            (
                f"Aktualisierte Kosten nach Anpassung: Tag {day.day} ca. "
                f"{day_cost} {itinerary.currency}, Dauer ca. {duration}; "
                f"Gesamtplan ca. {total} {itinerary.currency}."
            ),
        )
        day.notes = fresh_notes


def _format_cost_value(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def _format_duration_value(value: float) -> str:
    return f"{int(value)} h" if float(value).is_integer() else f"{value:.1f} h"


def _find_itinerary_activity(
    itinerary: Itinerary,
    target_terms: list[str],
    target_day: int | None,
) -> tuple | None:
    candidates: list[tuple[float, Any, int, Activity]] = []
    for day in itinerary.days:
        if target_day and day.day != target_day:
            continue
        for index, activity in enumerate(day.activities):
            score = max((_name_match_score(activity.name, term) for term in target_terms), default=0.0)
            if score >= 0.45:
                candidates.append((score, day, index, activity))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _score, day, index, activity = candidates[0]
    return day, index, activity


def _name_match_score(activity_name: str, term: str) -> float:
    name_tokens = _match_tokens(activity_name)
    term_tokens = _match_tokens(term)
    if not name_tokens or not term_tokens:
        return 0.0
    if " ".join(name_tokens) in " ".join(term_tokens) or " ".join(term_tokens) in " ".join(name_tokens):
        return 1.0
    overlap = set(name_tokens) & set(term_tokens)
    return len(overlap) / max(1, min(len(name_tokens), len(term_tokens)))


def _match_tokens(text: str) -> list[str]:
    import re

    stop_words = {
        "the",
        "and",
        "und",
        "oder",
        "ich",
        "kenne",
        "schon",
        "ersetze",
        "ersetz",
        "mal",
        "mit",
        "einem",
        "anderen",
        "andere",
        "alternative",
        "statt",
        "anstatt",
        "stattdessen",
        "instead",
        "rather",
        "lieber",
        "will",
        "möchte",
        "moechte",
        "bitte",
        "gehen",
        "geben",
        "hinzufügen",
        "hinzufuegen",
        "weitere",
        "noch",
        "eine",
        "einen",
        "das",
        "die",
        "der",
        "mich",
        "mir",
    }
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 2 and token not in stop_words
    ]


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
    haystack = _activity_search_text(activity)
    return any(term.strip().lower() and term.strip().lower() in haystack for term in avoid)


def _repair_must_have_coverage(
    itinerary: Itinerary,
    candidates: list[Activity],
    must_have: list[str],
) -> list[str]:
    notes: list[str] = []
    if not must_have or not itinerary.days:
        return notes

    used_names = {activity.name.strip().lower() for day in itinerary.days for activity in day.activities}
    for wish in must_have:
        if _itinerary_covers_wish(itinerary, wish):
            continue
        replacement = _best_unused_activity_for_wish(candidates, used_names, wish)
        if not replacement:
            continue
        target_day = min(itinerary.days, key=lambda day: (len(day.activities), day.total_duration_hours))
        if len(target_day.activities) >= 4:
            removed = _least_relevant_activity(target_day.activities, must_have)
            if removed:
                target_day.activities.remove(removed)
                used_names.discard(removed.name.strip().lower())
                target_day.notes.append(f"{removed.name} wurde ersetzt, damit ein offener Wunsch abgedeckt wird.")
        target_day.activities.append(replacement)
        used_names.add(replacement.name.strip().lower())
        target_day.notes.append(f"{replacement.name} wurde ergaenzt, um den Wunsch '{wish}' abzudecken.")
        notes.append(f"added {replacement.name} for missing wish '{wish}'")
    return notes


def _repair_interactive_includes(
    itinerary: Itinerary,
    candidates: list[Activity],
    decisions: dict,
) -> list[str]:
    notes: list[str] = []
    if not itinerary.days:
        return notes
    included = _decision_names(decisions, "include_names")
    if not included:
        return notes

    used_names = {activity.name.strip().lower() for day in itinerary.days for activity in day.activities}
    by_name = {activity.name.strip().lower(): activity for activity in candidates}
    for include_name in included:
        if include_name in used_names:
            continue
        candidate = by_name.get(include_name)
        if not candidate:
            continue
        target_day = min(itinerary.days, key=lambda day: (len(day.activities), day.total_duration_hours))
        if len(target_day.activities) >= 4:
            removable = _least_relevant_non_included_activity(target_day.activities, included)
            if removable:
                target_day.activities.remove(removable)
                used_names.discard(removable.name.strip().lower())
                target_day.notes.append(f"{removable.name} wurde ersetzt, damit ein explizit gewuenschter Kandidat eingeplant wird.")
        target_day.activities.append(candidate)
        used_names.add(candidate.name.strip().lower())
        target_day.notes.append(f"{candidate.name} wurde eingeplant, weil du es als 'Unbedingt einplanen' markiert hast.")
        notes.append(f"added explicitly included candidate {candidate.name}")
    return notes


def _least_relevant_non_included_activity(activities: list[Activity], included: set[str]) -> Activity | None:
    removable = [activity for activity in activities if activity.name.strip().lower() not in included]
    if not removable:
        return None
    removable.sort(key=lambda activity: (activity.cost, activity.duration_hours, activity.name.strip().lower()))
    return removable[0]


def _itinerary_covers_wish(itinerary: Itinerary, wish: str) -> bool:
    return any(
        _activity_covers_wish(activity, wish)
        for day in itinerary.days
        for activity in day.activities
    )


def _best_unused_activity_for_wish(
    candidates: list[Activity],
    used_names: set[str],
    wish: str,
) -> Activity | None:
    scored: list[tuple[float, Activity]] = []
    for activity in candidates:
        key = activity.name.strip().lower()
        if key in used_names:
            continue
        score = _wish_coverage_score(activity, wish)
        if score > 0:
            scored.append((score, activity))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _least_relevant_activity(activities: list[Activity], must_have: list[str]) -> Activity | None:
    if not activities:
        return None
    scored = [
        (
            max((_wish_coverage_score(activity, wish) for wish in must_have), default=0),
            activity,
        )
        for activity in activities
    ]
    scored.sort(key=lambda item: item[0])
    return scored[0][1]


def _text_matches_requirement(text: str, requirement: str) -> bool:
    return token_overlap_score(text, requirement) >= 0.5


def _activity_covers_wish(activity: Activity, wish: str) -> bool:
    return activity_covers_wish(activity, wish)


def _wish_coverage_score(activity: Activity, wish: str) -> float:
    return activity_wish_score(activity, wish)


def _matched_must_have_covers(description: str, wish: str) -> bool:
    return matched_must_have_covers(description, wish)


def _description_field(description: str, label: str) -> str:
    marker = f"{label}:"
    for part in str(description or "").split("|"):
        cleaned = part.strip()
        if cleaned.lower().startswith(marker.lower()):
            return cleaned.split(":", 1)[1].strip()
    return ""


def _requirement_match_score(text: str, requirement: str) -> float:
    return token_overlap_score(text, requirement)


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


def _needs_optimization(validation: ValidationResult) -> bool:
    if not validation.ok:
        return True
    actionable_warnings = {"budget_underused", "must_have_gap", "day_underfilled", "day_overload", "schedule_overload", "rain_conflict"}
    return any(issue.severity == "warning" and issue.issue_type in actionable_warnings for issue in validation.issues)
