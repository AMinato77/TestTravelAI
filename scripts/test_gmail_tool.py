from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.tools.gmail_tool import (
    GmailNewsletterMessage,
    _apply_ai_classification,
    _classify_message_fallback,
    _gmail_queries,
    _message_to_signal,
    _signals_to_preference_text,
)
from app.agents.preference_agent import extract_preferences
from app.models.preference_source import PreferenceSource
from app.models.travel_request import TravelRequest
from app.tools import openai_runtime


def main() -> None:
    message = {
        "labelIds": ["CATEGORY_PROMOTIONS"],
        "snippet": "Weekend flight deals to Barcelona, Madrid and Rome.",
        "payload": {
            "headers": [
                {"name": "From", "value": "Travel Deals <news@example.com>"},
                {"name": "Subject", "value": "Cheap city trip inspiration"},
                {"name": "Date", "value": "Mon, 01 Jan 2026 12:00:00 +0000"},
                {"name": "List-ID", "value": "<travel.example.com>"},
                {"name": "List-Unsubscribe", "value": "<mailto:unsubscribe@example.com>"},
            ]
        },
    }
    signal = _message_to_signal(message)
    assert isinstance(signal, GmailNewsletterMessage)
    signal = _classify_message_fallback(signal)
    assert signal.keep_as_preference
    assert signal.travel_relevance_score > 0.4
    text = _signals_to_preference_text([signal])
    assert "Gmail newsletter preference signals after relevance classification." in text
    assert "Travel Deals" in text
    assert "Evidence summary" in text
    assert "full email bodies are not stored" in text

    science_message = GmailNewsletterMessage(
        sender="Nature Briefing <briefing@nature.com>",
        subject="How Venus flytraps snap shut so fast",
        date="Fri, 12 Jun 2026 16:43:47 +0000",
        labels=["INBOX"],
        snippet="What matters in science. Today we learn how Venus flytraps snap shut.",
    )
    science_message = _classify_message_fallback(science_message)
    assert not science_message.keep_as_preference
    assert "wissenschaftlich" in science_message.ignore_reason

    landreise = GmailNewsletterMessage(
        sender="LandReise.de Team <service@landreise.de>",
        subject="Ich willkommen auf LandReise.de - jetzt erste Reisetipps entdecken!",
        date="Mon, 8 Jun 2026 21:03:55 +0000",
        labels=["INBOX"],
        snippet="Schoen, dass du mit uns auf Landreise gehst. Jetzt Landurlaub finden.",
    )
    classified = _apply_ai_classification(
        [landreise],
        {
            "messages": [
                {
                    "index": 1,
                    "keep": True,
                    "travel_relevance_score": 0.6,
                    "signal_strength": "medium",
                    "interest_tags": [],
                    "budget_signal": "unknown",
                    "travel_style_signal": "adventure",
                    "avoid": [],
                    "summary": "The email highlights land vacations and Baumhaushotel options.",
                    "reason": "travel newsletter",
                }
            ]
        },
    )
    assert classified[0].keep_as_preference
    assert classified[0].inferred_interest_tags

    openai_runtime.ai_provider = lambda: "demo"
    profile = extract_preferences(
        request=TravelRequest(
            destination="Rome",
            must_have=["local food"],
            interest_tags=["food"],
            travel_style="balanced",
        ),
        budget_preference="medium",
        preference_sources=[
            PreferenceSource(
                source_type="email_newsletter",
                name="gmail_newsletter_signals",
                text="Interest tags: nature\nTravel style signal: relaxed\n",
            )
        ],
    )
    assert "local food" in profile.preference_notes

    queries = _gmail_queries("", 30)
    assert "newer_than:30d" in queries[0]
    assert any("list-unsubscribe" in query for query in queries)
    print("gmail_tool signal extraction ok")


if __name__ == "__main__":
    main()
