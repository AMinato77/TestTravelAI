from __future__ import annotations


CALENDAR_STAGE_IDLE = "CALENDAR_IDLE"
CALENDAR_STAGE_PREVIEW = "CALENDAR_PREVIEW"
CALENDAR_STAGE_SYNCED = "CALENDAR_SYNCED"


def sync_key(plan_hash: str, calendar_id: str) -> str:
    return f"{plan_hash}:{calendar_id}"
