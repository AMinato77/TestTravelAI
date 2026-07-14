from __future__ import annotations

from app.tools.gmail_tool import (
    GmailNewsletterMessage,
    _apply_ai_classification,
    _build_gmail_memory_source,
    _classify_message_fallback,
    _signals_to_preference_text,
)


def test_science_newsletter_with_place_words_is_not_travel_preference():
    message = GmailNewsletterMessage(
        sender="Nature Briefing <briefing@example.com>",
        subject="Mysterious space balls on Australia beach probably came from rocket launches",
        date="Mon, 13 Jul 2026",
        snippet="Researchers say the objects are linked to rocket launches, not travel.",
        labels=["CATEGORY_UPDATES"],
    )

    classified = _classify_message_fallback(message)

    assert classified.keep_as_preference is False
    assert classified.inferred_interest_tags == []
    assert "reise" in classified.ignore_reason.lower() or "travel" in classified.ignore_reason.lower()


def test_ai_classification_cannot_keep_science_newsletter_with_beach_word():
    message = GmailNewsletterMessage(
        sender="Nature Briefing <briefing@example.com>",
        subject="Mysterious space balls on Australia beach probably came from rocket launches",
        date="Mon, 13 Jul 2026",
        snippet="Researchers say the objects are linked to rocket launches.",
        labels=["CATEGORY_UPDATES"],
    )

    classified = _apply_ai_classification(
        [message],
        {
            "messages": [
                {
                    "index": 1,
                    "keep": True,
                    "travel_relevance_score": 0.9,
                    "signal_strength": "strong",
                    "interest_tags": ["nature"],
                    "budget_signal": "unknown",
                    "travel_style_signal": "unknown",
                    "avoid": [],
                    "summary": "Nature and beach interest inferred.",
                    "reason": "Contains Australia beach.",
                }
            ]
        },
    )[0]

    classified = _classify_message_fallback(classified) if classified.keep_as_preference else classified

    assert classified.keep_as_preference is False


def test_vacation_newsletters_create_aggregated_query_compatible_memory():
    signals = [
        GmailNewsletterMessage(
            sender="Sonne und Strand PLUS <vorteile@plus.sonneundstrand.de>",
            subject="Diesen Urlaub vergessen die Kinder nie",
            date="Sat, 11 Jul 2026",
            snippet="Ferienhaus mit Pool am Strand fuer den Sommerurlaub.",
            labels=["CATEGORY_PROMOTIONS"],
            keep_as_preference=True,
            travel_relevance_score=0.85,
            signal_strength="strong",
            inferred_interest_tags=["family travel", "vacation home", "beach"],
            budget_signal="unknown",
            travel_style_signal="relaxed",
            preference_summary="Family vacation homes near the coast appear repeatedly.",
        ),
        GmailNewsletterMessage(
            sender="Sonne und Strand PLUS <vorteile@plus.sonneundstrand.de>",
            subject="Sommerurlaub ohne Stress",
            date="Sun, 12 Jul 2026",
            snippet="Rabatt fuer entspannte Ferienhaeuser in Daenemark.",
            labels=["CATEGORY_PROMOTIONS"],
            keep_as_preference=True,
            travel_relevance_score=0.75,
            signal_strength="strong",
            inferred_interest_tags=["vacation home", "discount", "beach"],
            budget_signal="low",
            travel_style_signal="relaxed",
            preference_summary="Relaxed coastal vacation-home offers with discounts.",
        ),
    ]

    text = _signals_to_preference_text(signals)

    assert "Reliable travel preference patterns" in text
    assert "vacation homes and coastal stays" in text
    assert "relaxed local experiences" in text
    assert "good value local experiences" in text
    assert "Use this as soft planning context only" in text


def test_single_weak_newsletter_signal_is_not_saved_as_memory_source():
    signal = GmailNewsletterMessage(
        sender="Generic Travel Deals <newsletter@example.com>",
        subject="Travel ideas this week",
        date="Mon, 13 Jul 2026",
        snippet="Some broad destination inspiration without a repeated preference pattern.",
        labels=["CATEGORY_PROMOTIONS"],
        keep_as_preference=True,
        travel_relevance_score=0.56,
        signal_strength="weak",
        inferred_interest_tags=[],
        budget_signal="unknown",
        travel_style_signal="unknown",
        preference_summary="A broad travel newsletter was opened.",
    )

    source = _build_gmail_memory_source([signal])

    assert source is None


def test_repeated_vacation_newsletters_are_saved_as_memory_source():
    signals = [
        GmailNewsletterMessage(
            sender="Sonne und Strand PLUS <vorteile@plus.sonneundstrand.de>",
            subject="Ferienhaus am Strand",
            date="Sat, 11 Jul 2026",
            snippet="Ferienhaus mit Pool am Strand fuer den Sommerurlaub.",
            labels=["CATEGORY_PROMOTIONS"],
            keep_as_preference=True,
            travel_relevance_score=0.85,
            signal_strength="strong",
            inferred_interest_tags=["vacation home", "beach"],
            budget_signal="unknown",
            travel_style_signal="relaxed",
            preference_summary="Family vacation homes near the coast appear repeatedly.",
        ),
        GmailNewsletterMessage(
            sender="Sonne und Strand PLUS <vorteile@plus.sonneundstrand.de>",
            subject="Sommerurlaub ohne Stress",
            date="Sun, 12 Jul 2026",
            snippet="Rabatt fuer entspannte Ferienhaeuser in Daenemark.",
            labels=["CATEGORY_PROMOTIONS"],
            keep_as_preference=True,
            travel_relevance_score=0.75,
            signal_strength="strong",
            inferred_interest_tags=["vacation home", "beach"],
            budget_signal="low",
            travel_style_signal="relaxed",
            preference_summary="Relaxed coastal vacation-home offers with discounts.",
        ),
    ]

    source = _build_gmail_memory_source(signals)

    assert source is not None
    assert source.source_type == "email_newsletter"
    assert "Reliable travel preference patterns" in source.text
