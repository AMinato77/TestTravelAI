from __future__ import annotations

from app.agents.query_planning_agent import PlaceQuery, _augment_queries_from_memory
from app.models.preference_source import PreferenceSource
from app.models.travel_request import TravelRequest


def test_gmail_memory_can_add_soft_query_when_request_has_no_must_haves():
    request = TravelRequest(
        destination="Madrid",
        duration_days=3,
        budget=600,
        travel_style="balanced",
        use_profile_memory=False,
    )
    memory = [
        PreferenceSource(
            source_type="email_newsletter",
            name="gmail_newsletter_signals",
            text=(
                "Gmail-derived travel preference memory after relevance classification.\n"
                "Reliable travel preference patterns:\n"
                "- Recurring interests inferred from newsletters: nature, beach holidays, relaxation, slow travel.\n"
                "- Travel style signal from newsletters: relaxed.\n"
                "Soft query directions for future trips:\n"
                "- scenic outdoor experiences if relevant\n"
                "- relaxed local experiences\n"
            ),
        )
    ]

    queries, usage = _augment_queries_from_memory(
        [PlaceQuery(query="Madrid tapas bars La Latina", reason="Base query.")],
        request,
        memory,
        max_queries=6,
    )

    assert any("scenic outdoor" in query.query.lower() or "parks" in query.query.lower() for query in queries)
    assert any(item.get("effect") for item in usage)
