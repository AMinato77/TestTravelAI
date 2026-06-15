from __future__ import annotations

import json
from collections import Counter

from app.models.activity import Activity
from app.models.user_profile import UserProfile
from app.services.budget_strategy import target_budget_range
from app.services.interest_coverage import coverage_for_activities
from app.tools.openai_runtime import ai_provider, openai_model


def run_agentic_tool_workflow(
    destination: str,
    days: int,
    activities: list[Activity],
    interests: list[str],
    request_hints: list[str],
    budget: float,
    profile: UserProfile,
) -> dict:
    """Run a lightweight Agents SDK workflow over internal TravelAI tools only."""
    candidate_summary = _candidate_summary(activities)
    interest_coverage = coverage_for_activities(activities, interests)
    target_min, target_max = target_budget_range(budget, profile)
    fallback = {
        "enabled": False,
        "summary": _workflow_summary(destination, days, activities, interests, interest_coverage, target_min, target_max),
        "tool_calls": [
            {
                "tool": "candidate_pool_summary",
                "result": candidate_summary,
            },
            {
                "tool": "interest_coverage",
                "result": interest_coverage,
            },
            {
                "tool": "budget_target_range",
                "result": {
                    "target_min": round(target_min, 2),
                    "target_max": round(target_max, 2),
                    "currency": "EUR",
                },
            },
            {
                "tool": "request_hint_summary",
                "result": {
                    "hints": request_hints,
                    "note": "Hints are soft planning context, not hard validation blockers.",
                },
            },
        ],
        "planning_guidance": _fallback_guidance(interest_coverage),
        "interest_coverage": interest_coverage,
    }

    if ai_provider() != "openai":
        fallback["summary"] = "Agents SDK tool workflow skipped because AI_PROVIDER is not openai."
        return fallback

    try:
        from agents import Agent, Runner, function_tool
    except Exception as exc:
        fallback["summary"] = f"Agents SDK tool workflow unavailable: {exc}"
        return fallback

    tool_calls: list[dict] = []

    @function_tool
    def inspect_candidate_pool() -> str:
        """Inspect available Google Places candidates by source and category."""
        tool_calls.append({"tool": "inspect_candidate_pool"})
        return json.dumps(candidate_summary, ensure_ascii=True)

    @function_tool
    def inspect_interest_coverage() -> str:
        """Inspect which requested interests have matching Google Places candidates."""
        tool_calls.append({"tool": "inspect_interest_coverage"})
        return json.dumps(interest_coverage, ensure_ascii=True)

    @function_tool
    def assess_budget_target() -> str:
        """Return the target activity-spend range for this trip."""
        result = {
            "available_budget": budget,
            "target_min": round(target_min, 2),
            "target_max": round(target_max, 2),
            "currency": "EUR",
            "travel_style": profile.travel_style,
            "budget_preference": profile.budget_preference,
        }
        tool_calls.append({"tool": "assess_budget_target"})
        return json.dumps(result, ensure_ascii=True)

    @function_tool
    def inspect_request_hints() -> str:
        """Inspect soft request hints extracted from the user request."""
        result = {
            "hints": request_hints,
            "policy": "Use hints to prefer matching candidates when available, but do not block the itinerary.",
        }
        tool_calls.append({"tool": "inspect_request_hints"})
        return json.dumps(result, ensure_ascii=True)

    agent = Agent(
        name="Internal Travel Tool Workflow Agent",
        model=openai_model("OPENAI_TOOL_WORKFLOW_MODEL"),
        instructions=(
            "You are an internal workflow agent for a travel planner. "
            "Use the available function tools to inspect candidate coverage, interest coverage, soft request hints, "
            "and budget targets. Do not search the web and do not invent activities. "
            "Do not create an itinerary, schedule, segment list, day plan, times, or venue sequence. "
            "The Planning Agent handles the itinerary later. "
            "Return strict JSON with keys summary, planning_guidance, risks, tool_decision. "
            "planning_guidance and risks must be short lists of plain strings. "
            "tool_decision must be a short string. Be concrete and keep request hints non-blocking."
        ),
        tools=[inspect_candidate_pool, inspect_interest_coverage, assess_budget_target, inspect_request_hints],
    )
    prompt = json.dumps(
        {
            "destination": destination,
            "duration_days": days,
            "interests": interests,
            "interest_coverage": interest_coverage,
            "request_hints": request_hints,
            "budget": budget,
            "profile": profile.to_dict(),
        },
        ensure_ascii=True,
    )
    try:
        result = Runner.run_sync(agent, prompt, max_turns=6)
        parsed = _parse_json(str(result.final_output))
    except Exception as exc:
        fallback["summary"] = f"Agents SDK tool workflow failed: {exc}"
        fallback["tool_calls"] = tool_calls or fallback["tool_calls"]
        return fallback

    return _normalize_agent_output(
        parsed=parsed,
        fallback=fallback,
        tool_calls=tool_calls,
        destination=destination,
        days=days,
        activities=activities,
        interests=interests,
        interest_coverage=interest_coverage,
        target_min=target_min,
        target_max=target_max,
    )


def _candidate_summary(activities: list[Activity]) -> dict:
    categories = Counter(activity.category for activity in activities)
    sources = Counter(activity.source for activity in activities)
    return {
        "candidate_count": len(activities),
        "categories": dict(sorted(categories.items())),
        "sources": dict(sorted(sources.items())),
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


def _fallback_guidance(interest_coverage: dict) -> list[str]:
    missing = interest_coverage.get("missing") or []
    guidance = ["Proceed with planning from verified candidates, then validate and optimize iteratively."]
    if missing:
        guidance.append(f"Soft coverage gap: no candidate found for {', '.join(missing)}.")
    else:
        guidance.append("Candidate pool has at least one match for every requested interest.")
    return guidance


def _workflow_summary(
    destination: str,
    days: int,
    activities: list[Activity],
    interests: list[str],
    interest_coverage: dict,
    target_min: float,
    target_max: float,
) -> str:
    missing = interest_coverage.get("missing") or []
    coverage_text = "all requested interests have candidates" if not missing else f"missing candidates for {', '.join(missing)}"
    return (
        f"Internal tool workflow inspected {len(activities)} Google Places candidates for a "
        f"{days}-day {destination} trip across {', '.join(interests) or 'general interests'}; "
        f"{coverage_text}; target activity spend is {round(target_min)}-{round(target_max)} EUR."
    )


def _normalize_agent_output(
    parsed: dict,
    fallback: dict,
    tool_calls: list[dict],
    destination: str,
    days: int,
    activities: list[Activity],
    interests: list[str],
    interest_coverage: dict,
    target_min: float,
    target_max: float,
) -> dict:
    guidance = _plain_string_list(parsed.get("planning_guidance"))
    if not guidance or _looks_like_schedule(parsed.get("planning_guidance")):
        guidance = _fallback_guidance(interest_coverage)
    risks = _plain_string_list(parsed.get("risks"))
    tool_decision = parsed.get("tool_decision")
    if not isinstance(tool_decision, str) or not tool_decision.strip():
        tool_decision = "Use verified Google Places candidates, then let the Planning Agent create the itinerary."
    return {
        "enabled": True,
        "summary": _workflow_summary(destination, days, activities, interests, interest_coverage, target_min, target_max),
        "tool_calls": tool_calls or fallback["tool_calls"],
        "planning_guidance": guidance,
        "risks": risks,
        "tool_decision": tool_decision.strip(),
        "interest_coverage": interest_coverage,
    }


def _plain_string_list(value) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result[:6]


def _looks_like_schedule(value) -> bool:
    text = json.dumps(value, ensure_ascii=True).lower() if isinstance(value, (list, dict)) else str(value).lower()
    schedule_terms = ["segment", "time", "venue", "activity", "09:", "10:", "11:", "12:", "13:", "14:"]
    return any(term in text for term in schedule_terms)


def _parse_json(text: str) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        data = json.loads(text[start : end + 1], strict=False) if start >= 0 and end > start else {}
    return data if isinstance(data, dict) else {"summary": text}
