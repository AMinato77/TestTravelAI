from __future__ import annotations

from app.models.activity import Activity
from app.models.user_profile import UserProfile
from app.tools.openai_runtime import demo_fallback_enabled, generate_json


def evaluate_activities(
    destination: str,
    activities: list[Activity],
    profile: UserProfile,
    budget: float,
    limit: int = 14,
    constraints: dict | None = None,
) -> tuple[list[Activity], dict]:
    """
    Ask GPT to judge the activity candidates before planning.

    Google Places gives us raw places, not perfect travel activities. This step
    removes weak matches such as unrelated shops, random objects, or categories
    the user did not ask for.
    """
    if not activities:
        return [], {"evaluations": [], "removed": []}

    if demo_fallback_enabled():
        return _demo_evaluate_activities(activities, profile, limit)
    constraints = constraints or {}

    activity_payload = [
        {
            "name": activity.name,
            "category": activity.category,
            "description": activity.description,
            "cost": activity.cost,
            "duration_hours": activity.duration_hours,
            "indoor": activity.indoor,
            "source": activity.source,
        }
        for activity in activities
    ]

    # GPT only receives a short, clean version of each candidate. It is not
    # allowed to invent new places; it only decides keep/remove + score.
    try:
        data = generate_json(
            system_prompt=(
                "You are an Activity Evaluation Agent for a travel planner. "
                "Your job is to judge whether each provided activity is a meaningful fit "
                "for the user's destination, concrete wishes, preference notes, avoid preferences, budget, and travel style. "
                "Do not invent new activities. Only evaluate activities from the input list. "
                "The destination match and concrete user wishes are critical. "
                "Remove candidates that are in another city/country or only weakly match a required theme. "
                "Be strict: if an activity only incidentally mentions an interest but its main category "
                "does not match, give it a low score or keep=false. "
                "Return strict JSON with key evaluations. evaluations is a list with keys: "
                "name, keep, score, reason. score must be 0-10."
            ),
            payload={
                "destination": destination,
                "budget": budget,
                "profile": profile.to_dict(),
                "constraints": constraints,
                "activities": activity_payload,
            },
            model_env="OPENAI_ACTIVITY_EVALUATION_MODEL",
        )
    except Exception as exc:
        kept, report = _demo_evaluate_activities(activities, profile, limit)
        report["fallback_reason"] = f"Activity Evaluation Agent fallback used because AI JSON parsing failed: {exc}"
        return kept, report

    evaluations = _parse_evaluations(data.get("evaluations", []))
    kept, report = _apply_evaluations(activities, evaluations, limit, profile, constraints)
    return _restore_minimum_coverage(kept, activities, report, limit, constraints)


def _apply_evaluations(
    activities: list[Activity],
    evaluations: list[dict],
    limit: int,
    profile: UserProfile,
    constraints: dict | None = None,
) -> tuple[list[Activity], dict]:
    """Convert GPT's JSON decision into a filtered and sorted activity list."""
    by_name = {activity.name.strip().lower(): activity for activity in activities}
    ranked: list[tuple[int, Activity]] = []
    removed: list[dict] = []
    evaluation_by_name: dict[str, dict] = {}

    for evaluation in evaluations:
        name = str(evaluation.get("name", "")).strip()
        activity = by_name.get(name.lower())
        if not activity:
            continue
        score = _safe_score(evaluation.get("score"))
        keep = bool(evaluation.get("keep", score >= 5))
        reason = str(evaluation.get("reason", "")).strip()
        keep, score, reason = _apply_hard_category_guard(activity, profile, keep, score, reason, constraints or {})
        evaluation_by_name[activity.name] = {
            "name": activity.name,
            "category": activity.category,
            "source": activity.source,
            "keep": keep,
            "score": score,
            "reason": reason,
        }
        if keep:
            ranked.append((score, activity))
        else:
            removed.append(evaluation_by_name[activity.name])

    evaluated_names = {name.lower() for name in evaluation_by_name}
    for activity in activities:
        if activity.name.lower() not in evaluated_names:
            ranked.append((5, activity))
            evaluation_by_name[activity.name] = {
                "name": activity.name,
                "category": activity.category,
                "source": activity.source,
                "keep": True,
                "score": 5,
                "reason": "Kept because the evaluator did not return an explicit decision.",
            }

    ranked.sort(key=lambda item: item[0], reverse=True)
    kept = [activity for _, activity in ranked[:limit]]
    return kept, {
        "evaluations": list(evaluation_by_name.values()),
        "removed": removed,
    }


def _restore_minimum_coverage(
    kept: list[Activity],
    original: list[Activity],
    report: dict,
    limit: int,
    constraints: dict | None,
) -> tuple[list[Activity], dict]:
    constraints = constraints or {}
    try:
        duration_days = int(constraints.get("duration_days") or 1)
    except (TypeError, ValueError):
        duration_days = 1
    minimum = min(len(original), limit, max(3, duration_days * 2))
    if len(kept) >= minimum:
        return kept, report

    kept_names = {activity.name.strip().lower() for activity in kept}
    restored: list[dict] = []
    avoid = _constraint_avoid_terms(constraints)
    for activity in original:
        key = activity.name.strip().lower()
        if key in kept_names:
            continue
        if _activity_conflicts_with_avoid(activity, avoid):
            continue
        kept.append(activity)
        kept_names.add(key)
        restored.append(
            {
                "name": activity.name,
                "category": activity.category,
                "source": activity.source,
                "keep": True,
                "score": 5,
                "reason": "Restored to keep enough real API candidates for the requested trip length.",
            }
        )
        if len(kept) >= minimum:
            break

    if restored:
        report["coverage_restored"] = restored
        removed_names = {item["name"].strip().lower() for item in restored}
        report["removed"] = [
            item for item in report.get("removed", []) if str(item.get("name", "")).strip().lower() not in removed_names
        ]
        report.setdefault("warnings", []).append(
            f"Restored {len(restored)} candidate(s) because the evaluator kept too few activities for the trip length."
        )
    return kept, report


def _apply_hard_category_guard(
    activity: Activity,
    profile: UserProfile,
    keep: bool,
    score: int,
    reason: str,
    constraints: dict,
) -> tuple[bool, int, str]:
    """Simple Python safety rule after GPT evaluation."""
    activity_text = f"{activity.name} {activity.category} {activity.description}".lower()
    if _activity_conflicts_with_avoid(activity, _constraint_avoid_terms(constraints) or profile.avoid):
        return (
            False,
            0,
            reason or "Removed because it conflicts with user avoid preferences.",
        )
    wishes = _constraint_wishes(constraints) or profile.preference_notes
    if wishes and not _matches_any_wish(activity_text, wishes):
        return keep, min(score, 6), reason or "Kept with lower confidence because it only weakly matches concrete wishes."
    return keep, score, reason


def _constraint_avoid_terms(constraints: dict | None) -> list[str]:
    if not constraints:
        return []
    avoid = constraints.get("avoid") or []
    return [str(item).strip().lower() for item in avoid if str(item).strip()]


def _constraint_wishes(constraints: dict | None) -> list[str]:
    if not constraints:
        return []
    values = [*(constraints.get("must_have") or []), *(constraints.get("query_hints") or [])]
    return [str(item).strip().lower() for item in values if str(item).strip()]


def _activity_conflicts_with_avoid(activity: Activity, avoid: list[str]) -> bool:
    haystack = f"{activity.name} {activity.category} {activity.description}".lower()
    avoid_text = " ".join(avoid).lower()
    category = activity.category.strip().lower()
    if not avoid_text:
        return False
    if category in {term.strip().lower() for term in avoid if term.strip()}:
        return True
    if any(term in avoid_text for term in ["museum", "museums", "museen"]):
        return "museum" in haystack or "museen" in haystack or category == "museum"
    if any(term in avoid_text for term in ["food", "restaurant", "restaurants", "cafe", "cafes"]):
        return category == "food" or any(term in haystack for term in ["restaurant", "food", "cafe", "caf"])
    if any(term in avoid_text for term in ["nightlife", "club", "clubs"]):
        return category == "nightlife" or any(term in haystack for term in ["club", "nightlife", "bar"])
    return any(term and term in haystack for term in avoid)


def _matches_any_wish(activity_text: str, wishes: list[str]) -> bool:
    return any(_overlaps_soft_request_hint(activity_text, wish) for wish in wishes)


def _overlaps_soft_request_hint(activity_text: str, request_hint_text: str) -> bool:
    tokens = {
        token
        for token in request_hint_text.replace("-", " ").split()
        if len(token) > 2 and token not in {"and", "the", "for", "und", "oder", "mit", "von", "fuer", "für"}
    }
    if not tokens:
        return False
    matches = sum(1 for token in tokens if token in activity_text)
    return matches / len(tokens) >= 0.4


def _parse_evaluations(raw_evaluations) -> list[dict]:
    if not isinstance(raw_evaluations, list):
        return []
    parsed: list[dict] = []
    for item in raw_evaluations:
        if isinstance(item, dict):
            parsed.append(item)
    return parsed


def _safe_score(value) -> int:
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        return 5
    return max(0, min(score, 10))


def _demo_evaluate_activities(
    activities: list[Activity],
    profile: UserProfile,
    limit: int,
) -> tuple[list[Activity], dict]:
    wish_text = " ".join(profile.preference_notes).lower()
    avoid_text = " ".join(profile.avoid).lower()
    evaluations: list[dict] = []
    ranked: list[tuple[int, Activity]] = []

    for activity in activities:
        haystack = f"{activity.name} {activity.category} {activity.description}".lower()
        score = 5
        reason = "Kept by demo evaluator."
        keep = True

        if _matches_any_wish(haystack, profile.preference_notes):
            score += 3
        if activity.category in avoid_text:
            score -= 7
            reason = "Removed because category conflicts with avoid preferences."
        if any(term in avoid_text for term in ["food", "restaurant", "cafe"]) and activity.category == "food":
            score -= 8
            reason = "Removed because food conflicts with avoid preferences."

        keep = score >= 5
        evaluation = {
            "name": activity.name,
            "category": activity.category,
            "source": activity.source,
            "keep": keep,
            "score": max(0, min(score, 10)),
            "reason": reason,
        }
        evaluations.append(evaluation)
        if keep:
            ranked.append((evaluation["score"], activity))

    ranked.sort(key=lambda item: item[0], reverse=True)
    kept = [activity for _, activity in ranked[:limit]]
    return kept, {
        "evaluations": evaluations,
        "removed": [evaluation for evaluation in evaluations if not evaluation["keep"]],
    }
