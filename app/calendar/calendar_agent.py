from __future__ import annotations

from app.calendar.calendar_models import CalendarEventCopy
from app.tools.openai_runtime import demo_fallback_enabled, generate_json


def generate_calendar_copy(activity_payloads: list[dict]) -> list[CalendarEventCopy]:
    if not activity_payloads or demo_fallback_enabled():
        return [_fallback_copy(activity) for activity in activity_payloads]
    try:
        data = generate_json(
            system_prompt=(
                "Du bist ein Calendar Delivery Agent für TravelAI. "
                "Du darfst nur nutzerfreundliche Kalendereintrag-Titel, kurze Beschreibungen "
                "und eine Erinnerung in Minuten formulieren. "
                "Du darfst niemals Datum, Uhrzeit, Dauer, Reihenfolge, Adresse, Maps-Link, Kosten "
                "oder Reiseziel ändern. Gib striktes JSON mit key events zurück. "
                "Jedes Event: activity_id, title, description, reminder_minutes."
            ),
            payload={"activities": activity_payloads},
            model_env="OPENAI_CALENDAR_AGENT_MODEL",
        )
        rows = data.get("events") if isinstance(data.get("events"), list) else []
        by_id = {str(row.get("activity_id")): row for row in rows if isinstance(row, dict)}
        result: list[CalendarEventCopy] = []
        for activity in activity_payloads:
            activity_id = str(activity["activity_id"])
            row = by_id.get(activity_id) or {}
            result.append(_coerce_copy(activity, row))
        return result
    except Exception:
        return [_fallback_copy(activity) for activity in activity_payloads]


def generate_fallback_calendar_copy(activity_payloads: list[dict]) -> list[CalendarEventCopy]:
    return [_fallback_copy(activity) for activity in activity_payloads]


def _coerce_copy(activity: dict, row: dict) -> CalendarEventCopy:
    fallback = _fallback_copy(activity)
    title = _clean(row.get("title")) or fallback.title
    description = _clean(row.get("description")) or fallback.description
    try:
        reminder = int(row.get("reminder_minutes", fallback.reminder_minutes))
    except (TypeError, ValueError):
        reminder = fallback.reminder_minutes
    reminder = min(max(reminder, 0), 24 * 60)
    return CalendarEventCopy(
        activity_id=str(activity["activity_id"]),
        title=title[:90],
        description=description[:900],
        reminder_minutes=reminder,
    )


def _fallback_copy(activity: dict) -> CalendarEventCopy:
    name = _clean(activity.get("name")) or "Reisestopp"
    category = _clean(activity.get("category"))
    destination = _clean(activity.get("destination")) or "deiner Reise"
    title = _title_for(name, category)
    description = _clean(activity.get("reason"))
    if not description:
        description = f"Dieser Stopp ergänzt deinen Reiseplan in {destination} und passt gut in den Tagesablauf."
    return CalendarEventCopy(
        activity_id=str(activity["activity_id"]),
        title=title,
        description=description,
        reminder_minutes=30,
    )


def _title_for(name: str, category: str) -> str:
    lower = category.lower()
    if "restaurant" in lower or "tapas" in lower:
        return f"Kulinarischer Stopp bei {name}"
    if "fußball" in lower:
        return f"Fußballerlebnis: {name}"
    if "park" in lower:
        return f"Entspannte Pause im {name}"
    if "architektur" in lower:
        return f"Architektur entdecken: {name}"
    return name


def _clean(value) -> str:
    return " ".join(str(value or "").replace("\x00", "").split())
