from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CalendarEventCopy:
    activity_id: str
    title: str
    description: str
    reminder_minutes: int = 30


@dataclass(frozen=True)
class CalendarEventDraft:
    activity_id: str
    title: str
    description: str
    date: str
    start_time: str
    end_time: str
    start_datetime: str
    end_datetime: str
    timezone: str
    location: str = ""
    maps_url: str = ""
    website: str = ""
    cost_label: str = ""
    reminder_minutes: int = 30
    day_number: int = 1
    source_activity_name: str = ""


@dataclass(frozen=True)
class CalendarPreview:
    destination: str
    start_date: str
    timezone: str
    plan_hash: str
    events: list[CalendarEventDraft] = field(default_factory=list)


@dataclass(frozen=True)
class CalendarInfo:
    calendar_id: str
    summary: str
    primary: bool = False
    writable: bool = True


@dataclass(frozen=True)
class CalendarSyncSuccess:
    activity_id: str
    event_id: str
    html_link: str = ""


@dataclass(frozen=True)
class CalendarSyncFailure:
    activity_id: str
    title: str
    error: str


@dataclass(frozen=True)
class CalendarSyncResult:
    successes: list[CalendarSyncSuccess]
    failures: list[CalendarSyncFailure]

    @property
    def ok(self) -> bool:
        return not self.failures
