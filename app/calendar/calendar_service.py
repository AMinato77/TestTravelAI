from __future__ import annotations

from datetime import date

from app.calendar.calendar_adapter import build_calendar_preview
from app.calendar.calendar_auth import CalendarIntegrationError
from app.calendar.calendar_models import (
    CalendarInfo,
    CalendarPreview,
    CalendarSyncFailure,
    CalendarSyncResult,
    CalendarSyncSuccess,
)
from app.calendar.google_calendar_client import GoogleCalendarClient


def prepare_calendar_preview(plan, start_date: date | str, use_ai: bool = True) -> CalendarPreview:
    return build_calendar_preview(plan, start_date=start_date, use_ai=use_ai)


def list_user_calendars(user_id: str, client: GoogleCalendarClient | None = None) -> list[CalendarInfo]:
    calendar_client = client or GoogleCalendarClient(user_id)
    return calendar_client.list_calendars()


def create_trip_calendar(user_id: str, summary: str, timezone: str, client: GoogleCalendarClient | None = None) -> CalendarInfo:
    calendar_client = client or GoogleCalendarClient(user_id)
    return calendar_client.create_calendar(summary=summary, timezone=timezone)


def sync_preview_to_calendar(
    user_id: str,
    calendar_id: str,
    preview: CalendarPreview,
    already_synced_activity_ids: set[str] | None = None,
    client: GoogleCalendarClient | None = None,
) -> CalendarSyncResult:
    if not calendar_id:
        raise CalendarIntegrationError("Bitte einen Zielkalender auswählen.")
    synced = set(already_synced_activity_ids or set())
    calendar_client = client or GoogleCalendarClient(user_id)
    successes: list[CalendarSyncSuccess] = []
    failures: list[CalendarSyncFailure] = []
    for event in preview.events:
        if event.activity_id in synced:
            continue
        try:
            created = calendar_client.insert_event(calendar_id=calendar_id, event=event, plan_hash=preview.plan_hash)
            successes.append(
                CalendarSyncSuccess(
                    activity_id=event.activity_id,
                    event_id=str(created.get("id") or ""),
                    html_link=str(created.get("htmlLink") or ""),
                )
            )
        except Exception as exc:
            failures.append(CalendarSyncFailure(activity_id=event.activity_id, title=event.title, error=str(exc)))
    return CalendarSyncResult(successes=successes, failures=failures)
