from __future__ import annotations

from datetime import date

import pytest

from app.calendar.calendar_adapter import CalendarAdapterError, build_calendar_preview
from app.calendar import calendar_agent
from app.calendar.calendar_auth import CALENDAR_SCOPES
from app.calendar.calendar_models import CalendarInfo
from app.calendar.calendar_service import create_trip_calendar, list_user_calendars, sync_preview_to_calendar
from app.calendar.google_calendar_client import event_to_google_body


def sample_plan() -> dict:
    return {
        "itinerary": {
            "destination": "Madrid",
            "currency": "EUR",
            "total_cost": 90,
            "days": [
                {
                    "day": 1,
                    "activities": [
                        {
                            "name": "Casa Alberto",
                            "category": "food",
                            "description": (
                                "Matched must-have: typical Spanish cuisine | "
                                "Address: C. de las Huertas, 18, Madrid, Spain | "
                                "Rating: 4.4/5 | Reviews: 5248 | "
                                "Website: https://www.casaalberto.es/ | "
                                "Google Maps: https://maps.google.com/?cid=1"
                            ),
                            "cost": 35,
                            "duration_hours": 1.5,
                        }
                    ],
                },
                {
                    "day": 2,
                    "activities": [
                        {
                            "name": "Bernabeu",
                            "category": "sport",
                            "description": (
                                "Matched must-have: watch a football match in Madrid | "
                                "Address: Av. de Concha Espina, 1, Madrid, Spain | "
                                "Google Maps: https://maps.google.com/?cid=2"
                            ),
                            "cost": 55,
                            "duration_hours": 2,
                        }
                    ],
                },
            ],
        },
        "request": {
            "destination": "Madrid",
            "duration_days": 2,
            "budget": 350,
            "must_have": ["typical Spanish cuisine", "watch a football match in Madrid"],
            "travel_style": "balanced",
        },
        "validation": {"ok": True, "issues": []},
    }


class FakeCalendarClient:
    def __init__(self, fail_ids: set[str] | None = None):
        self.fail_ids = fail_ids or set()
        self.inserted = []

    def list_calendars(self):
        return [CalendarInfo(calendar_id="primary", summary="Persönlich", primary=True, writable=True)]

    def create_calendar(self, summary: str, timezone: str):
        return CalendarInfo(calendar_id="created", summary=summary, primary=False, writable=True)

    def insert_event(self, calendar_id, event, plan_hash):
        if event.activity_id in self.fail_ids:
            raise RuntimeError("API unavailable")
        self.inserted.append((calendar_id, event.activity_id, plan_hash))
        return {"id": f"event-{event.activity_id}", "htmlLink": f"https://calendar/{event.activity_id}"}


def test_final_plan_to_calendar_preview_dates_timezone():
    preview = build_calendar_preview(sample_plan(), date(2026, 7, 12), use_ai=False)
    assert preview.destination == "Madrid"
    assert preview.timezone == "Europe/Madrid"
    assert len(preview.events) == 2
    first, second = preview.events
    assert first.date == "2026-07-12"
    assert second.date == "2026-07-13"
    assert first.start_datetime.startswith("2026-07-12T09:00:00")
    assert second.start_datetime.startswith("2026-07-13T09:00:00")
    assert first.location == "C. de las Huertas, 18, Madrid, Spain"
    assert first.maps_url.startswith("https://maps.google.com")


def test_calendar_copy_does_not_change_fixed_values():
    preview = build_calendar_preview(sample_plan(), "2026-07-12", use_ai=False)
    event = preview.events[0]
    assert event.start_time == "09:00"
    assert event.end_time == "10:30"
    assert event.cost_label == "35 EUR"
    assert event.location == "C. de las Huertas, 18, Madrid, Spain"


def test_google_event_body_contains_extended_properties():
    preview = build_calendar_preview(sample_plan(), "2026-07-12", use_ai=False)
    body = event_to_google_body(preview.events[0], preview.plan_hash)
    assert body["summary"]
    assert body["start"]["dateTime"].startswith("2026-07-12T09:00:00")
    assert body["end"]["dateTime"].startswith("2026-07-12T10:30:00")
    assert body["extendedProperties"]["private"]["travelai_plan_hash"] == preview.plan_hash
    assert "Google Maps:" in body["description"]


def test_calendar_selection_and_create():
    client = FakeCalendarClient()
    calendars = list_user_calendars("user", client=client)
    assert calendars[0].calendar_id == "primary"
    created = create_trip_calendar("user", "Madrid Reise 2026", "Europe/Madrid", client=client)
    assert created.calendar_id == "created"


def test_calendar_uses_full_scope_for_trip_calendar_creation():
    assert CALENDAR_SCOPES == ["https://www.googleapis.com/auth/calendar"]


def test_calendar_agent_returns_ai_copy_list(monkeypatch):
    monkeypatch.setattr(calendar_agent, "demo_fallback_enabled", lambda: False)
    monkeypatch.setattr(
        calendar_agent,
        "generate_json",
        lambda **_: {
            "events": [
                {
                    "activity_id": "activity-1",
                    "title": "Kulinarischer Start",
                    "description": "Starte entspannt mit regionaler Küche.",
                    "reminder_minutes": 20,
                }
            ]
        },
    )
    result = calendar_agent.generate_calendar_copy(
        [
            {
                "activity_id": "activity-1",
                "name": "Casa Alberto",
                "category": "Restaurant",
                "destination": "Madrid",
                "reason": "Traditionelle Küche.",
                "cost_label": "35 EUR",
            }
        ]
    )
    assert len(result) == 1
    assert result[0].title == "Kulinarischer Start"
    assert result[0].reminder_minutes == 20


def test_partial_failed_batch_and_duplicate_guard():
    preview = build_calendar_preview(sample_plan(), "2026-07-12", use_ai=False)
    client = FakeCalendarClient(fail_ids={preview.events[1].activity_id})
    result = sync_preview_to_calendar("user", "primary", preview, client=client)
    assert len(result.successes) == 1
    assert len(result.failures) == 1

    synced = {item.activity_id for item in result.successes}
    retry_client = FakeCalendarClient()
    retry = sync_preview_to_calendar("user", "primary", preview, already_synced_activity_ids=synced, client=retry_client)
    assert len(retry_client.inserted) == 1
    assert retry_client.inserted[0][1] == preview.events[1].activity_id


def test_missing_travel_date():
    with pytest.raises(CalendarAdapterError):
        build_calendar_preview(sample_plan(), "", use_ai=False)


def test_umlauts_survive_preview():
    plan = sample_plan()
    plan["itinerary"]["days"][0]["activities"][0]["name"] = "Café München"
    preview = build_calendar_preview(plan, "2026-07-12", use_ai=False)
    assert "Café München" in preview.events[0].source_activity_name
