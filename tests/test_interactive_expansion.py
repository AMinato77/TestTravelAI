from __future__ import annotations

from app.models.activity import Activity
from app.models.travel_request import TravelRequest
from app.models.user_profile import UserProfile
from app.orchestrator import PreparedPlanContext, expand_interactive_plan


def test_interactive_expansion_does_not_readd_already_visited_candidate(monkeypatch):
    request = TravelRequest(
        destination="Paris",
        duration_days=3,
        budget=300,
        must_have=["anime stores"],
    )
    profile = UserProfile(user_id="test")
    prepared = PreparedPlanContext(
        request=request,
        profile=profile,
        loaded_memory=profile,
        activities=[
            Activity(name="Junku", category="shopping"),
            Activity(name="Album Comics", category="shopping"),
        ],
        weather={},
        workflow_steps=[],
        activity_evaluation={},
        memory_context=[],
        agentic_tool_workflow={},
        place_queries=[],
        query_planning={},
        constraints={},
        questions=[],
    )

    monkeypatch.setattr(
        "app.orchestrator.search_places_with_metadata",
        lambda **_: (
            [
                Activity(name="Junku", category="shopping"),
                Activity(name="Manga Story", category="shopping"),
            ],
            {"query_count": 1, "cache_hits": 0, "queries": []},
        ),
    )
    monkeypatch.setattr(
        "app.orchestrator.evaluate_activities",
        lambda **kwargs: (kwargs["activities"], {"evaluations": [], "removed": []}),
    )

    expanded = expand_interactive_plan(
        prepared,
        "mehr anime shops",
        {"already_visited_names": ["Junku"], "exclude_names": []},
    )

    names = [activity.name for activity in expanded.activities]
    assert "Junku" not in names
    assert "Manga Story" in names
    assert "Album Comics" in names
