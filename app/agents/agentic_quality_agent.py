from __future__ import annotations

import json

from app.models.itinerary import Itinerary, ValidationResult
from app.models.user_profile import UserProfile
from app.services.budget_strategy import budget_utilization, target_budget_range
from app.tools.openai_runtime import ai_provider, openai_model


def run_agentic_quality_review(
    itinerary: Itinerary,
    budget: float,
    profile: UserProfile,
    validation: ValidationResult,
) -> dict:
    """Run an OpenAI Agents SDK quality review with function tools."""
    if ai_provider() != "openai":
        return {
            "enabled": False,
            "summary": "Agents SDK quality review skipped because AI_PROVIDER is not openai.",
        }

    try:
        from agents import Agent, Runner, function_tool
    except Exception as exc:
        return {
            "enabled": False,
            "summary": f"Agents SDK quality review unavailable: {exc}",
        }

    @function_tool
    def assess_budget_quality(available_budget: float, planned_cost: float, travel_style: str, budget_preference: str) -> str:
        """Assess whether the itinerary uses the available budget meaningfully."""
        target_min, target_max = target_budget_range(
            available_budget,
            UserProfile(travel_style=travel_style, budget_preference=budget_preference),
        )
        utilization = planned_cost / available_budget if available_budget else 0
        if planned_cost < target_min:
            status = "underused"
        elif planned_cost > target_max:
            status = "high_but_allowed"
        else:
            status = "good"
        return json.dumps(
            {
                "status": status,
                "utilization": round(utilization, 3),
                "target_min": round(target_min, 2),
                "target_max": round(target_max, 2),
            },
            ensure_ascii=True,
        )

    @function_tool
    def assess_validation_quality(error_count: int, warning_count: int) -> str:
        """Assess whether deterministic validation left unresolved issues."""
        if error_count:
            status = "blocking_errors"
        elif warning_count:
            status = "warnings_remaining"
        else:
            status = "clean"
        return json.dumps(
            {"status": status, "errors": error_count, "warnings": warning_count},
            ensure_ascii=True,
        )

    agent = Agent(
        name="Travel Quality Review Agent",
        model=openai_model("OPENAI_QUALITY_REVIEW_MODEL"),
        instructions=(
            "You are a quality review agent for a personalized travel planner. "
            "Use the provided tools to assess budget quality and validation quality. "
            "Return only strict JSON with keys: summary, budget_assessment, validation_assessment, improvements. "
            "improvements must be a short list. Be concrete and concise."
        ),
        tools=[assess_budget_quality, assess_validation_quality],
    )

    prompt = json.dumps(
        {
            "destination": itinerary.destination,
            "budget": budget,
            "planned_cost": itinerary.total_cost,
            "budget_utilization": round(budget_utilization(itinerary, budget), 3),
            "profile": profile.to_dict(),
            "validation": {
                "ok": validation.ok,
                "error_count": validation.error_count,
                "warning_count": validation.warning_count,
                "issues": [
                    {
                        "severity": issue.severity,
                        "type": issue.issue_type,
                        "message": issue.message,
                    }
                    for issue in validation.issues
                ],
            },
        },
        ensure_ascii=True,
    )

    try:
        result = Runner.run_sync(agent, prompt, max_turns=6)
        return _parse_review(str(result.final_output))
    except Exception as exc:
        return {
            "enabled": False,
            "summary": f"Agents SDK quality review failed: {exc}",
        }


def _parse_review(text: str) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {
            "enabled": True,
            "summary": text,
            "budget_assessment": "",
            "validation_assessment": "",
            "improvements": [],
        }
    if not isinstance(data, dict):
        return {"enabled": True, "summary": text}
    data["enabled"] = True
    data.setdefault("improvements", [])
    return data
