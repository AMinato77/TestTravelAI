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
from app.agents.revision_agent import interpret_revision_feedback
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
    revision: dict | None = None


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


def revise_travel_plan(
    previous_result: TravelPlanResult,
    feedback: str,
    original_inputs: dict,
) -> TravelPlanResult:
    load_dotenv()
    reset_openai_usage_records()
    itinerary = deepcopy(previous_result.itinerary)
    profile = deepcopy(previous_result.profile)

    must_have = _merge_unique(
        original_inputs.get("must_have") or [],
        original_inputs.get("query_hints") or [],
    )
    avoid = _merge_unique(profile.avoid, original_inputs.get("avoid") or [])
    revision = interpret_revision_feedback(
        itinerary=itinerary,
        feedback=feedback,
        original_request=original_inputs,
        must_have=must_have,
        avoid=avoid,
    )
    revision["feedback"] = feedback
    avoid = _merge_unique(avoid, revision.get("avoid_additions") or [])
    must_have = _merge_unique(must_have, revision.get("must_have_additions") or [])
    query_hints = _merge_unique(
        original_inputs.get("query_hints") or [],
        revision.get("query_hints") or [],
        [feedback],
        must_have,
    )
    profile.avoid = avoid

    workflow_steps = [
        *previous_result.workflow_steps,
        f"User feedback for revision: {feedback}",
        f"Revision Agent classified feedback as {revision.get('intent')}: {revision.get('reasoning')}",
    ]

    new_queries = [
        PlaceQuery(query=query, reason="Revision feedback query.", source="revision_agent", must_have=must_have)
        for query in query_hints
    ]
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

    if intent == "add_more_similar":
        candidate = _first_unused_activity(activities, used, avoid)
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


def _find_replacement_activity(
    original: Activity,
    activities: list[Activity],
    used: set[str],
    avoid: list[str],
    revision: dict | None = None,
) -> Activity | None:
    preferred_categories = _preferred_replacement_categories(original, revision or {})
    for activity in activities:
        key = activity.name.strip().lower()
        if key in used or key == original.name.strip().lower():
            continue
        if activity.category not in preferred_categories:
            continue
        if _activity_conflicts_with_avoid(activity, avoid):
            continue
        return activity
    return _first_unused_activity(activities, used, avoid)


def _first_unused_activity(activities: list[Activity], used: set[str], avoid: list[str]) -> Activity | None:
    for activity in activities:
        key = activity.name.strip().lower()
        if key in used:
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

    stop_words = {"the", "and", "und", "oder", "ich", "kenne", "schon", "ersetze", "ersetz", "mal", "mit", "einem", "anderen"}
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 2 and token not in stop_words
    ]


def _preferred_replacement_categories(original: Activity, revision: dict) -> set[str]:
    text = " ".join(
        [
            str(revision.get("feedback") or ""),
            " ".join(revision.get("must_have_additions") or []),
            " ".join(revision.get("query_hints") or []),
            str(revision.get("revision_instruction") or ""),
        ]
    ).lower()
    if any(term in text for term in ["natur", "nature", "park", "garden", "tea garden", "viewpoint", "outdoor"]):
        return {"nature", "culture"} if original.category == "culture" else {"nature"}
    if any(term in text for term in ["restaurant", "essen", "food", "kueche", "küche"]):
        return {"food"}
    if any(term in text for term in ["anime", "shop", "shopping", "store"]):
        return {"shopping", "anime"}
    return {original.category}


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
