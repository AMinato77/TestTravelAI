from __future__ import annotations

from app.calendar.calendar_auth import CalendarIntegrationError, calendar_credentials
from app.calendar.calendar_models import CalendarEventDraft, CalendarInfo


class GoogleCalendarClient:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self._service = None

    @property
    def service(self):
        if self._service is None:
            try:
                from googleapiclient.discovery import build
            except ImportError as exc:
                raise CalendarIntegrationError("Google Calendar-Abhängigkeiten fehlen.") from exc
            self._service = build(
                "calendar",
                "v3",
                credentials=calendar_credentials(self.user_id, allow_oauth=True),
                cache_discovery=False,
            )
        return self._service

    def list_calendars(self) -> list[CalendarInfo]:
        try:
            result = self.service.calendarList().list().execute()
        except Exception as exc:
            raise CalendarIntegrationError(f"Kalender konnten nicht geladen werden: {_safe_error(exc)}") from exc
        calendars: list[CalendarInfo] = []
        for item in result.get("items", []) or []:
            access_role = str(item.get("accessRole") or "")
            calendars.append(
                CalendarInfo(
                    calendar_id=str(item.get("id") or ""),
                    summary=str(item.get("summary") or "Kalender"),
                    primary=bool(item.get("primary")),
                    writable=access_role in {"owner", "writer"},
                )
            )
        return [calendar for calendar in calendars if calendar.calendar_id]

    def create_calendar(self, summary: str, timezone: str) -> CalendarInfo:
        body = {"summary": summary, "timeZone": timezone}
        try:
            created = self.service.calendars().insert(body=body).execute()
        except Exception as exc:
            raise CalendarIntegrationError(f"Kalender konnte nicht erstellt werden: {_safe_error(exc)}") from exc
        return CalendarInfo(
            calendar_id=str(created.get("id") or ""),
            summary=str(created.get("summary") or summary),
            primary=False,
            writable=True,
        )

    def insert_event(self, calendar_id: str, event: CalendarEventDraft, plan_hash: str) -> dict:
        body = event_to_google_body(event, plan_hash)
        try:
            return self.service.events().insert(calendarId=calendar_id, body=body).execute()
        except Exception as exc:
            raise CalendarIntegrationError(_safe_error(exc)) from exc


def event_to_google_body(event: CalendarEventDraft, plan_hash: str) -> dict:
    description_parts = [event.description]
    if event.cost_label:
        description_parts.append(f"Geschätzte Kosten: {event.cost_label}")
    if event.maps_url:
        description_parts.append(f"Google Maps: {event.maps_url}")
    if event.website:
        description_parts.append(f"Website: {event.website}")

    return {
        "summary": event.title,
        "location": event.location,
        "description": "\n\n".join(part for part in description_parts if part),
        "start": {"dateTime": event.start_datetime, "timeZone": event.timezone},
        "end": {"dateTime": event.end_datetime, "timeZone": event.timezone},
        "reminders": {
            "useDefault": False,
            "overrides": [{"method": "popup", "minutes": int(event.reminder_minutes)}],
        },
        "extendedProperties": {
            "private": {
                "travelai_plan_hash": plan_hash,
                "travelai_activity_id": event.activity_id,
                "travelai_plan_id": plan_hash[:16],
            }
        },
    }


def _safe_error(exc: Exception) -> str:
    text = str(exc)
    lowered = text.lower()
    if "accessnotconfigured" in lowered or "api has not been used" in lowered:
        return "Google Calendar API ist für dieses Google-Cloud-Projekt vermutlich nicht aktiviert."
    if "insufficient" in lowered or "scope" in lowered:
        return "Google Calendar-Zugriff fehlt oder der OAuth-Scope wurde noch nicht freigegeben."
    if "forbidden" in lowered or "not authorized" in lowered:
        return "Auf diesen Kalender kann nicht geschrieben werden."
    if "rate" in lowered or "quota" in lowered:
        return "Google Calendar Rate Limit erreicht. Bitte später erneut versuchen."
    if "timeout" in lowered:
        return "Google Calendar hat zu lange nicht geantwortet."
    return text[:400]
