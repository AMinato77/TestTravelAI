from __future__ import annotations

import json
from dataclasses import dataclass

from app.agents.query_planning_agent import PlaceQuery
from app.models.activity import Activity
from app.models.preference_source import PreferenceSource
from app.models.user_profile import UserProfile
from app.services.budget_strategy import target_budget_range
from app.services.serialization import activity_from_dict, itinerary_to_dict, validation_from_dict, validation_to_dict
from app.services.wish_matching import activity_covers_wish
from app.tools.openai_runtime import ai_provider, openai_model
from app.tools.places_tool import search_places_with_metadata
from app.tools.tool_server_client import post_tool, tool_server_enabled
from app.tools.validation_tool import validate_itinerary
from app.tools.weather_tool import get_weather


@dataclass(slots=True)
class AgenticPlanningWorkflowResult:
    activities: list[Activity]
    weather: dict
    places_metadata: dict
    workflow: dict


@dataclass(slots=True)
class AgenticValidationWorkflowResult:
    validation: object
    workflow: dict


def run_agentic_planning_workflow(
    *,
    destination: str,
    days: int,
    budget: float,
    profile: UserProfile,
    place_queries: list[PlaceQuery],
    avoid: list[str],
    must_have: list[str],
    query_hints: list[str],
    memory_context: list[PreferenceSource],
) -> AgenticPlanningWorkflowResult:
    """Let an Agents SDK agent call the core planning tools and expose the trace.

    The orchestration still controls data flow, but Places/Weather/Memory/Coverage/Budget
    are presented as actual tool decisions instead of invisible sequential function calls.
    """

    state: dict = {
        "activities": None,
        "weather": None,
        "places_metadata": None,
        "memory_summary": None,
        "wish_coverage": None,
        "budget_assessment": None,
    }
    tool_calls: list[dict] = []
    direct_notes: list[str] = []

    def inspect_memory_context() -> dict:
        if state["memory_summary"] is None:
            state["memory_summary"] = {
                "memory_chunk_count": len(memory_context),
                "sources": [
                    {
                        "source_type": source.source_type,
                        "name": source.name,
                        "text_preview": source.text[:220],
                    }
                    for source in memory_context[:5]
                ],
            }
        return state["memory_summary"]

    def search_google_places() -> dict:
        if state["activities"] is not None and state["places_metadata"] is not None:
            return _places_result_summary(state["activities"], state["places_metadata"])

        activities, metadata, executor = _search_places_tool(destination, place_queries, avoid)
        state["activities"] = activities
        state["places_metadata"] = metadata
        return {
            **_places_result_summary(activities, metadata),
            "executor": executor,
        }

    def get_weather_context() -> dict:
        if state["weather"] is None:
            weather, executor = _weather_tool(destination, days)
            state["weather"] = weather
            state["weather_executor"] = executor
        return {
            "executor": state.get("weather_executor") or "local_tool_adapter",
            "provider": state["weather"].get("provider"),
            "summary": state["weather"].get("summary"),
            "rain_expected": state["weather"].get("rain_expected"),
            "max_rain_chance": state["weather"].get("max_rain_chance"),
        }

    def inspect_wish_coverage() -> dict:
        if state["activities"] is None:
            search_google_places()
        state["wish_coverage"] = _wish_coverage(state["activities"] or [], must_have)
        return state["wish_coverage"]

    def assess_budget_target() -> dict:
        if state["budget_assessment"] is None:
            target_min, target_max = target_budget_range(budget, profile)
            state["budget_assessment"] = {
                "available_budget": round(float(budget or 0), 2),
                "target_min": round(target_min, 2),
                "target_max": round(target_max, 2),
                "currency": "EUR",
                "travel_style": profile.travel_style,
                "budget_preference": profile.budget_preference,
            }
        return state["budget_assessment"]

    def record_tool(name: str, decision: str, result: dict) -> str:
        tool_calls.append(
            {
                "tool": name,
                "decision": decision,
                "result": _compact_tool_result(result),
            }
        )
        return json.dumps(result, ensure_ascii=True)

    parsed_agent_output: dict = {}
    mode = "direct_tool_pipeline"
    sdk_error = ""

    if ai_provider() == "openai":
        try:
            from agents import Agent, Runner, function_tool

            @function_tool
            def memory_context_tool() -> str:
                """Inspect retrieved ChromaDB memory context before choosing planning guidance."""
                return record_tool(
                    "memory_context_tool",
                    "Check whether stored preferences should influence query/plan decisions.",
                    inspect_memory_context(),
                )

            @function_tool
            def google_places_tool() -> str:
                """Execute the prepared Google Places Text Search queries and inspect candidates."""
                return record_tool(
                    "google_places_tool",
                    "Fetch real places for the concrete query plan.",
                    search_google_places(),
                )

            @function_tool
            def weather_context_tool() -> str:
                """Fetch weather context for the destination and travel duration."""
                return record_tool(
                    "weather_context_tool",
                    "Check weather before planning outdoor/indoor balance.",
                    get_weather_context(),
                )

            @function_tool
            def wish_coverage_tool() -> str:
                """Inspect whether candidate places cover the concrete must-have wishes."""
                return record_tool(
                    "wish_coverage_tool",
                    "Verify candidate coverage before the itinerary is built.",
                    inspect_wish_coverage(),
                )

            @function_tool
            def budget_target_tool() -> str:
                """Inspect the target spend range for the given budget and profile."""
                return record_tool(
                    "budget_target_tool",
                    "Assess whether candidate costs can fit the budget strategy.",
                    assess_budget_target(),
                )

            agent = Agent(
                name="TravelAI Planning Tool Agent",
                model=openai_model("OPENAI_TOOL_WORKFLOW_MODEL"),
                instructions=(
                    "You are the tool-using planning coordinator for TravelAI. "
                    "You do not write the itinerary. Your job is to decide which tools are needed "
                    "before itinerary planning. You must call memory_context_tool, google_places_tool, "
                    "weather_context_tool, wish_coverage_tool, and budget_target_tool. "
                    "Then return strict JSON with keys summary, agent_decisions, planning_guidance, risks. "
                    "Emphasize that real tool results, not general chat knowledge, are used."
                ),
                tools=[
                    memory_context_tool,
                    google_places_tool,
                    weather_context_tool,
                    wish_coverage_tool,
                    budget_target_tool,
                ],
            )
            result = Runner.run_sync(
                agent,
                json.dumps(
                    {
                        "destination": destination,
                        "duration_days": days,
                        "budget": budget,
                        "must_have": must_have,
                        "query_hints": query_hints,
                        "place_queries": [
                            {
                                "query": query.query,
                                "reason": query.reason,
                                "must_have": query.must_have,
                            }
                            for query in place_queries
                        ],
                        "avoid": avoid,
                        "profile": profile.to_dict(),
                    },
                    ensure_ascii=True,
                ),
                max_turns=10,
            )
            parsed_agent_output = _parse_json(str(result.final_output))
            mode = "agents_sdk"
        except Exception as exc:
            sdk_error = str(exc)
            mode = "agents_sdk_recovered"
    else:
        mode = "direct_tool_pipeline"

    required_direct_tools = [
        ("memory_context_tool", "Directly inspect memory context for planning.", inspect_memory_context),
        ("google_places_tool", "Directly fetch real Google Places candidates.", search_google_places),
        ("weather_context_tool", "Directly fetch weather context.", get_weather_context),
        ("wish_coverage_tool", "Directly inspect candidate wish coverage.", inspect_wish_coverage),
        ("budget_target_tool", "Directly assess budget target range.", assess_budget_target),
    ]
    called_names = {call.get("tool") for call in tool_calls}
    if ai_provider() == "openai":
        missing = [name for name, _decision, _callback in required_direct_tools if name not in called_names]
        if missing:
            sdk_error = sdk_error or f"Agents SDK planning workflow did not call required tools: {', '.join(missing)}"
            mode = "agents_sdk_recovered"
    for name, decision, callback in required_direct_tools:
        if name in called_names:
            continue
        result = callback()
        direct_notes.append(f"{name} completed by deterministic tool pipeline.")
        tool_calls.append(
            {
                "tool": name,
                "decision": decision,
                "result": _compact_tool_result(result),
                "recovery": mode != "agents_sdk",
            }
        )

    activities = state["activities"] or []
    weather = state["weather"] or {}
    places_metadata = state["places_metadata"] or {"query_count": 0, "cache_hits": 0, "queries": []}
    wish_coverage = state["wish_coverage"] or _wish_coverage(activities, must_have)
    budget_assessment = state["budget_assessment"] or assess_budget_target()

    summary = str(parsed_agent_output.get("summary") or "").strip()
    if not summary:
        summary = _workflow_summary(mode, activities, wish_coverage, budget_assessment)

    workflow = {
        "enabled": mode in {"agents_sdk", "agents_sdk_recovered"},
        "mode": mode,
        "summary": summary,
        "tool_calls": tool_calls,
        "agent_decisions": _plain_string_list(parsed_agent_output.get("agent_decisions")) or _default_decisions(tool_calls),
        "planning_guidance": _plain_string_list(parsed_agent_output.get("planning_guidance")) or _default_guidance(wish_coverage),
        "risks": _plain_string_list(parsed_agent_output.get("risks")),
        "wish_coverage": wish_coverage,
        "budget_assessment": budget_assessment,
        "places_metadata": places_metadata,
        "weather_summary": {
            "provider": weather.get("provider"),
            "summary": weather.get("summary"),
            "rain_expected": weather.get("rain_expected"),
            "max_rain_chance": weather.get("max_rain_chance"),
        },
    }
    if sdk_error:
        workflow["agents_sdk_error"] = sdk_error
    if direct_notes:
        workflow["direct_tool_notes"] = direct_notes
    if sdk_error:
        workflow["agents_sdk_error"] = sdk_error

    return AgenticPlanningWorkflowResult(
        activities=activities,
        weather=weather,
        places_metadata=places_metadata,
        workflow=workflow,
    )


def run_agentic_validation_workflow(
    *,
    itinerary,
    budget: float,
    weather: dict,
    profile: UserProfile,
    constraints: dict,
) -> AgenticValidationWorkflowResult:
    """Let an Agents SDK agent explicitly call the validation tool after planning."""

    state: dict = {"validation": None}
    tool_calls: list[dict] = []
    parsed_agent_output: dict = {}
    mode = "direct_tool_pipeline"
    sdk_error = ""

    def validation_tool_call() -> dict:
        if state["validation"] is not None:
            return validation_to_dict(state["validation"])
        validation, executor = _validation_tool(itinerary, budget, weather, profile, constraints)
        state["validation"] = validation
        return {
            **validation_to_dict(validation),
            "executor": executor,
        }

    def record_tool(name: str, decision: str, result: dict) -> str:
        tool_calls.append(
            {
                "tool": name,
                "decision": decision,
                "result": _compact_tool_result(result),
            }
        )
        return json.dumps(result, ensure_ascii=True)

    if ai_provider() == "openai":
        try:
            from agents import Agent, Runner, function_tool

            @function_tool
            def validate_itinerary_tool() -> str:
                """Validate the generated itinerary against budget, weather, avoid rules, and concrete wishes."""
                return record_tool(
                    "validate_itinerary_tool",
                    "Check whether the generated itinerary is actually acceptable before showing it.",
                    validation_tool_call(),
                )

            agent = Agent(
                name="TravelAI Validation Tool Agent",
                model=openai_model("OPENAI_TOOL_WORKFLOW_MODEL"),
                instructions=(
                    "You are the validation coordinator for TravelAI. "
                    "You must call validate_itinerary_tool before returning. "
                    "Return strict JSON with keys summary, agent_decisions, validation_decision. "
                    "Do not invent validation results; rely on the tool output."
                ),
                tools=[validate_itinerary_tool],
            )
            result = Runner.run_sync(
                agent,
                json.dumps(
                    {
                        "itinerary": itinerary_to_dict(itinerary),
                        "budget": budget,
                        "weather_summary": {
                            "provider": weather.get("provider"),
                            "summary": weather.get("summary"),
                            "rain_expected": weather.get("rain_expected"),
                        },
                        "constraints": constraints,
                    },
                    ensure_ascii=True,
                ),
                max_turns=4,
            )
            parsed_agent_output = _parse_json(str(result.final_output))
            mode = "agents_sdk"
        except Exception as exc:
            raise RuntimeError(f"Agents SDK validation workflow failed; no local recovery is used in OpenAI mode: {exc}") from exc

    if state["validation"] is None:
        if ai_provider() == "openai":
            raise RuntimeError("Agents SDK validation workflow did not call validate_itinerary_tool.")
        result = validation_tool_call()
        tool_calls.append(
            {
                "tool": "validate_itinerary_tool",
                "decision": "Directly validate itinerary after planning.",
                "result": _compact_tool_result(result),
                "recovery": mode != "agents_sdk",
            }
        )

    validation = state["validation"]
    summary = str(parsed_agent_output.get("summary") or "").strip()
    if not summary:
        summary = (
            f"Validation tool workflow ({mode}) returned "
            f"{validation.error_count} error(s) and {validation.warning_count} warning(s)."
        )

    workflow = {
        "enabled": mode == "agents_sdk",
        "mode": mode,
        "summary": summary,
        "tool_calls": tool_calls,
        "agent_decisions": _plain_string_list(parsed_agent_output.get("agent_decisions"))
        or _default_decisions(tool_calls),
        "validation_decision": str(parsed_agent_output.get("validation_decision") or "").strip(),
    }
    if sdk_error:
        workflow["agents_sdk_error"] = sdk_error

    return AgenticValidationWorkflowResult(validation=validation, workflow=workflow)


def _search_places_tool(destination: str, queries: list[PlaceQuery], avoid: list[str]) -> tuple[list[Activity], dict, str]:
    if ai_provider() == "openai" and not tool_server_enabled():
        raise RuntimeError("TRAVEL_TOOL_SERVER_URL is required in OpenAI mode for the Places tool.")
    if tool_server_enabled():
        try:
            data = post_tool(
                "/tools/places/search",
                {
                    "destination": destination,
                    "queries": [
                        {
                            "query": query.query,
                            "reason": query.reason,
                            "source": query.source,
                            "must_have": query.must_have,
                        }
                        for query in queries
                    ],
                    "avoid": avoid,
                    "limit": 20,
                },
            )
            activities = [activity_from_dict(activity) for activity in data.get("activities", [])]
            metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else _metadata_from_queries(queries)
            return activities, metadata, "tool_server"
        except Exception as exc:
            if ai_provider() == "openai":
                raise RuntimeError(f"Tool Server Places tool failed: {exc}") from exc
            pass
    activities, metadata = search_places_with_metadata(destination=destination, queries=queries, avoid=avoid)
    return activities, metadata, "local_tool_adapter"


def _weather_tool(destination: str, days: int) -> tuple[dict, str]:
    if ai_provider() == "openai" and not tool_server_enabled():
        raise RuntimeError("TRAVEL_TOOL_SERVER_URL is required in OpenAI mode for the Weather tool.")
    if tool_server_enabled():
        try:
            return post_tool("/tools/weather", {"destination": destination, "days": days}), "tool_server"
        except Exception as exc:
            if ai_provider() == "openai":
                raise RuntimeError(f"Tool Server Weather tool failed: {exc}") from exc
            pass
    return get_weather(destination, days=days), "local_tool_adapter"


def _validation_tool(itinerary, budget: float, weather: dict, profile: UserProfile, constraints: dict) -> tuple[object, str]:
    if ai_provider() == "openai" and not tool_server_enabled():
        raise RuntimeError("TRAVEL_TOOL_SERVER_URL is required in OpenAI mode for the Validation tool.")
    if tool_server_enabled():
        try:
            data = post_tool(
                "/tools/itinerary/validate",
                {
                    "itinerary": itinerary_to_dict(itinerary),
                    "budget": budget,
                    "weather": weather,
                    "profile": profile.to_dict(),
                    "constraints": constraints,
                },
            )
            return validation_from_dict(data), "tool_server"
        except Exception as exc:
            if ai_provider() == "openai":
                raise RuntimeError(f"Tool Server Validation tool failed: {exc}") from exc
            pass
    return validate_itinerary(itinerary, budget, weather, profile, constraints=constraints), "local_tool_adapter"


def _places_result_summary(activities: list[Activity], metadata: dict) -> dict:
    return {
        "candidate_count": len(activities),
        "query_count": int(metadata.get("query_count") or 0),
        "cache_hits": int(metadata.get("cache_hits") or 0),
        "queries": [item.get("query") for item in metadata.get("queries", []) if isinstance(item, dict)][:8],
        "top_candidates": [
            {
                "name": activity.name,
                "category": activity.category,
                "source": activity.source,
                "cost": activity.cost,
                "indoor": activity.indoor,
            }
            for activity in activities[:8]
        ],
    }


def _wish_coverage(activities: list[Activity], wishes: list[str]) -> dict:
    cleaned = _merge_unique(wishes)
    covered = {
        wish: [activity.name for activity in activities if _matches_wish(activity, wish)]
        for wish in cleaned
    }
    return {
        "covered": {wish: names for wish, names in covered.items() if names},
        "missing": [wish for wish, names in covered.items() if not names],
        "counts": {wish: len(names) for wish, names in covered.items()},
    }


def _matches_wish(activity: Activity, wish: str) -> bool:
    return activity_covers_wish(activity, wish)


def _matched_must_have_covers(description: str, wish: str) -> bool:
    matched = _description_field(description, "Matched must-have")
    wanted = " ".join(str(wish or "").lower().split())
    if not matched or not wanted:
        return False
    return any(" ".join(part.lower().split()) == wanted for part in matched.split(","))


def _matching_description(description: str) -> str:
    kept: list[str] = []
    blocked_labels = {"matched query", "matched must-have", "google maps", "website"}
    for part in str(description or "").split("|"):
        cleaned = part.strip()
        label = cleaned.split(":", 1)[0].strip().lower() if ":" in cleaned else ""
        if label in blocked_labels:
            continue
        if cleaned.lower().startswith(("http://", "https://")):
            continue
        kept.append(cleaned)
    return " | ".join(kept)


def _description_field(description: str, label: str) -> str:
    marker = f"{label}:"
    for part in str(description or "").split("|"):
        cleaned = part.strip()
        if cleaned.lower().startswith(marker.lower()):
            return cleaned.split(":", 1)[1].strip()
    return ""


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
        "in",
        "near",
        "nach",
        "city",
        "country",
        "places",
        "things",
        "trip",
        "travel",
        "tour",
        "discover",
        "explore",
    }
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 2 and token not in stop_words
    ]


def _metadata_from_queries(queries: list[PlaceQuery]) -> dict:
    return {
        "query_count": len(queries),
        "cache_hits": 0,
        "queries": [
            {
                "query": query.query,
                "reason": query.reason,
                "source": query.source,
                "must_have": query.must_have,
            }
            for query in queries
        ],
    }


def _compact_tool_result(result: dict) -> dict:
    text = json.dumps(result, ensure_ascii=True)
    if len(text) <= 3000:
        return result
    compact = {
        key: value
        for key, value in result.items()
        if key not in {"sources", "top_candidates", "queries", "covered"}
    }
    if "top_candidates" in result:
        compact["top_candidates"] = result["top_candidates"][:5]
    if "queries" in result:
        compact["queries"] = result["queries"][:5]
    if "covered" in result:
        compact["covered"] = {key: value[:5] for key, value in result["covered"].items()}
    return compact


def _workflow_summary(mode: str, activities: list[Activity], wish_coverage: dict, budget_assessment: dict) -> str:
    missing = wish_coverage.get("missing") or []
    coverage = "all concrete wishes have candidate support" if not missing else f"missing support for {', '.join(missing[:4])}"
    return (
        f"Agentic tool workflow ({mode}) inspected {len(activities)} real candidate(s), "
        f"{coverage}, and assessed the {budget_assessment.get('target_min')}-{budget_assessment.get('target_max')} EUR budget target."
    )


def _default_decisions(tool_calls: list[dict]) -> list[str]:
    return [
        f"{call.get('tool')}: {call.get('decision')}"
        for call in tool_calls
        if call.get("tool") and call.get("decision")
    ][:8]


def _default_guidance(wish_coverage: dict) -> list[str]:
    missing = wish_coverage.get("missing") or []
    if missing:
        return [f"Retrieve or prioritize candidates for missing concrete wishes: {', '.join(missing[:4])}."]
    return ["Use verified Places candidates, weather context, memory context, and validation before finalizing the itinerary."]


def _plain_string_list(value) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:8]


def _parse_json(text: str) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        data = json.loads(text[start : end + 1], strict=False) if start >= 0 and end > start else {}
    return data if isinstance(data, dict) else {"summary": text}


def _merge_unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value).strip().split())
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result
