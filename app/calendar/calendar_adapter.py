from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from app.calendar.calendar_agent import generate_calendar_copy, generate_fallback_calendar_copy
from app.calendar.calendar_models import CalendarEventDraft, CalendarPreview
from app.export.export_context import build_pdf_context
from app.export.export_service import calculate_plan_hash


DESTINATION_TIMEZONES = {
    "madrid": "Europe/Madrid",
    "barcelona": "Europe/Madrid",
    "paris": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "hamburg": "Europe/Berlin",
    "münchen": "Europe/Berlin",
    "munich": "Europe/Berlin",
    "rome": "Europe/Rome",
    "rom": "Europe/Rome",
    "athens": "Europe/Athens",
    "athen": "Europe/Athens",
    "prague": "Europe/Prague",
    "prag": "Europe/Prague",
    "london": "Europe/London",
    "new york": "America/New_York",
    "miami": "America/New_York",
}


class CalendarAdapterError(RuntimeError):
    pass


def build_calendar_preview(plan: Any, start_date: date | str, use_ai: bool = True) -> CalendarPreview:
    parsed_start = _parse_start_date(start_date)
    context = build_pdf_context(plan)
    destination = context["destination"]
    timezone = _timezone_for(destination)
    plan_hash = calculate_plan_hash(plan)

    activity_payloads = []
    fixed_rows = []
    for day in context.get("days") or []:
        day_number = int(day.get("number") or 1)
        event_date = parsed_start + timedelta(days=day_number - 1)
        for index, activity in enumerate(day.get("activities") or [], start=1):
            activity_id = f"day-{day_number}-activity-{index}"
            start_dt = _combine(event_date, activity.get("start_time"), timezone)
            end_dt = _combine(event_date, activity.get("end_time"), timezone)
            if end_dt <= start_dt:
                raise CalendarAdapterError(f"Ungültige Zeit bei {activity.get('name') or 'Aktivität'}.")
            fixed = {
                "activity_id": activity_id,
                "name": activity.get("name", ""),
                "category": activity.get("category", ""),
                "destination": destination,
                "date": event_date.isoformat(),
                "start_time": activity.get("start_time", ""),
                "end_time": activity.get("end_time", ""),
                "start_datetime": start_dt.isoformat(),
                "end_datetime": end_dt.isoformat(),
                "timezone": timezone,
                "location": activity.get("address", ""),
                "maps_url": activity.get("maps_url", ""),
                "website": activity.get("website", ""),
                "cost_label": activity.get("cost_label", ""),
                "reason": activity.get("reason", ""),
                "day_number": day_number,
            }
            fixed_rows.append(fixed)
            activity_payloads.append(
                {
                    "activity_id": activity_id,
                    "name": fixed["name"],
                    "category": fixed["category"],
                    "destination": destination,
                    "reason": fixed["reason"],
                    "cost_label": fixed["cost_label"],
                }
            )

    if not fixed_rows:
        raise CalendarAdapterError("Der Reiseplan enthält keine Aktivitäten für den Kalender.")

    copies = generate_calendar_copy(activity_payloads) if use_ai else generate_fallback_calendar_copy(activity_payloads)
    copy_by_id = {item.activity_id: item for item in copies}
    events = []
    for fixed in fixed_rows:
        event_copy = copy_by_id.get(fixed["activity_id"])
        if event_copy is None:
            event_copy = generate_fallback_calendar_copy([fixed])[0]
        events.append(
            CalendarEventDraft(
                activity_id=fixed["activity_id"],
                title=event_copy.title,
                description=event_copy.description,
                date=fixed["date"],
                start_time=fixed["start_time"],
                end_time=fixed["end_time"],
                start_datetime=fixed["start_datetime"],
                end_datetime=fixed["end_datetime"],
                timezone=fixed["timezone"],
                location=fixed["location"],
                maps_url=fixed["maps_url"],
                website=fixed["website"],
                cost_label=fixed["cost_label"],
                reminder_minutes=event_copy.reminder_minutes,
                day_number=fixed["day_number"],
                source_activity_name=fixed["name"],
            )
        )
    return CalendarPreview(destination=destination, start_date=parsed_start.isoformat(), timezone=timezone, plan_hash=plan_hash, events=events)


def _parse_start_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    if not value:
        raise CalendarAdapterError("Bitte ein Startdatum für die Reise angeben.")
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError as exc:
        raise CalendarAdapterError("Startdatum muss im Format YYYY-MM-DD vorliegen.") from exc


def _combine(day: date, time_value: str, timezone: str) -> datetime:
    if not time_value:
        raise CalendarAdapterError("Eine Aktivität hat keine Uhrzeit.")
    try:
        hour, minute = [int(part) for part in str(time_value).split(":", 1)]
    except ValueError as exc:
        raise CalendarAdapterError(f"Ungültige Uhrzeit: {time_value}") from exc
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=ZoneInfo(timezone))


def _timezone_for(destination: str) -> str:
    lowered = str(destination or "").lower()
    for key, timezone in DESTINATION_TIMEZONES.items():
        if key in lowered:
            return timezone
    return "Europe/Berlin"


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return _to_plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value
