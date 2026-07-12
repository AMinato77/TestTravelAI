from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from app.export.discord_client import send_itinerary_to_discord
from app.export.export_context import build_pdf_context, validate_export_context
from app.export.pdf_generator import generate_itinerary_pdf


@dataclass(frozen=True)
class ExportResult:
    pdf_bytes: bytes
    filename: str
    plan_hash: str


class ExportValidationError(RuntimeError):
    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("Der Reiseplan ist noch nicht bereit für den PDF-Export: " + "; ".join(issues))


def calculate_plan_hash(plan: Any) -> str:
    serialized = json.dumps(_to_jsonable(plan), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def create_trip_export(plan: Any) -> ExportResult:
    plan_hash = calculate_plan_hash(plan)
    context = build_pdf_context(plan)
    issues = validate_export_context(context)
    if issues:
        raise ExportValidationError(issues)
    filename = _filename_for_destination(context.get("destination") or "reise")
    pdf_bytes = generate_itinerary_pdf(context)
    return ExportResult(pdf_bytes=pdf_bytes, filename=filename, plan_hash=plan_hash)


def deliver_trip_to_discord(plan: Any, export: ExportResult | None = None) -> ExportResult:
    trip_export = export or create_trip_export(plan)
    send_itinerary_to_discord(plan=plan, pdf_bytes=trip_export.pdf_bytes, filename=trip_export.filename)
    return trip_export


def _filename_for_destination(destination: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(destination).strip().lower()).strip("_")
    return f"reiseplan_{cleaned or 'reise'}.pdf"


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _to_jsonable(value.to_dict())
    return str(value)
