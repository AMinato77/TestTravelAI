from __future__ import annotations

import json
from collections import Counter

from app.models.activity import Activity
from app.models.user_profile import UserProfile
from app.services.budget_strategy import target_budget_range
from app.tools.openai_runtime import ai_provider, openai_model


def run_agentic_tool_workflow(
    destination: str,
    days: int,
    activities: list[Activity],
    must_have: list[str],
    query_hints: list[str],
    budget: float,
    profile: UserProfile,
) -> dict:
    candidate_summary = _candidate_summary(activities)
    wish_coverage = _wish_coverage(activities, must_have)
    target_min, target_max = target_budget_range(budget, profile)
    fallback = {
        "enabled": False,
        "summary": _workflow_summary(destination, days, activities, wish_coverage, target_min, target_max),
        "tool_calls": [
            {"tool": "candidate_pool_summary", "result": candidate_summary},
            {"tool": "wish_coverage", "result": wish_coverage},
            {"tool": "budget_target_range", "result": {"target_min": round(target_min, 2), "target_max": round(target_max, 2), "currency": "EUR"}},
        ],
        "planning_guidance": _fallback_guidance(wish_coverage),
        "wish_coverage": wish_coverage,
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
        tool_calls.append({"tool": "inspect_candidate_pool"})
        return json.dumps(candidate_summary, ensure_ascii=True)

    @function_tool
    def inspect_wish_coverage() -> str:
        tool_calls.append({"tool": "inspect_wish_coverage"})
        return json.dumps(wish_coverage, ensure_ascii=True)

    @function_tool
    def assess_budget_target() -> str:
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

    agent = Agent(
        name="Internal Travel Tool Workflow Agent",
        model=openai_model("OPENAI_TOOL_WORKFLOW_MODEL"),
        instructions=(
            "Inspect whether Google Places candidates can support the concrete user wishes, "
            "avoid constraints, and budget target. Do not create an itinerary. "
            "Return strict JSON with keys summary, planning_guidance, risks, tool_decision."
        ),
        tools=[inspect_candidate_pool, inspect_wish_coverage, assess_budget_target],
    )
    try:
        result = Runner.run_sync(
            agent,
            json.dumps(
                {
                    "destination": destination,
                    "duration_days": days,
                    "must_have": must_have,
                    "query_hints": query_hints,
                    "wish_coverage": wish_coverage,
                    "budget": budget,
                    "profile": profile.to_dict(),
                },
                ensure_ascii=True,
            ),
            max_turns=6,
        )
        parsed = _parse_json(str(result.final_output))
    except Exception as exc:
        fallback["summary"] = f"Agents SDK tool workflow failed: {exc}"
        fallback["tool_calls"] = tool_calls or fallback["tool_calls"]
        return fallback

    return {
        "enabled": True,
        "summary": str(parsed.get("summary") or _workflow_summary(destination, days, activities, wish_coverage, target_min, target_max)),
        "tool_calls": tool_calls or fallback["tool_calls"],
        "planning_guidance": _plain_string_list(parsed.get("planning_guidance")) or _fallback_guidance(wish_coverage),
        "risks": _plain_string_list(parsed.get("risks")),
        "tool_decision": str(parsed.get("tool_decision") or "Use verified Google Places candidates for planning.").strip(),
        "wish_coverage": wish_coverage,
    }


def _candidate_summary(activities: list[Activity]) -> dict:
    return {
        "candidate_count": len(activities),
        "categories": dict(sorted(Counter(activity.category for activity in activities).items())),
        "sources": dict(sorted(Counter(activity.source for activity in activities).items())),
        "top_candidates": [
            {"name": activity.name, "category": activity.category, "source": activity.source, "cost": activity.cost, "indoor": activity.indoor}
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
    if _matched_must_have_covers(activity.description, wish):
        return True
    text = f"{activity.name} {activity.category} {_matching_description(activity.description)}".lower()
    tokens = _content_tokens(wish)
    if not tokens:
        return False
    matches = sum(1 for token in tokens if token in text)
    threshold = 1 if len(tokens) == 1 else max(2, round(len(tokens) * 0.65))
    return matches >= threshold


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
        "fÃ¼r",
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
        for token in re.findall(r"[a-z0-9Ã¤Ã¶Ã¼ÃŸ]+", str(text).lower())
        if len(token) > 2 and token not in stop_words
    ]


def _fallback_guidance(wish_coverage: dict) -> list[str]:
    missing = wish_coverage.get("missing") or []
    guidance = ["Plan only with verified Google Places candidates, then validate and optimize."]
    if missing:
        guidance.append(f"Concrete wish coverage gap: {', '.join(missing[:5])}.")
    return guidance


def _workflow_summary(destination: str, days: int, activities: list[Activity], wish_coverage: dict, target_min: float, target_max: float) -> str:
    missing = wish_coverage.get("missing") or []
    coverage_text = "concrete wishes have candidate support" if not missing else f"missing candidate support for {', '.join(missing[:5])}"
    return f"Inspected {len(activities)} Google Places candidates for {days} days in {destination}; {coverage_text}; target spend {round(target_min)}-{round(target_max)} EUR."


def _plain_string_list(value) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:6]


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
