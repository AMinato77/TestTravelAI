from __future__ import annotations

from app.models.activity import Activity
from app.models.itinerary import Itinerary, ItineraryDay, ValidationIssue, ValidationResult


def activity_to_dict(activity: Activity) -> dict:
    return {
        "name": activity.name,
        "category": activity.category,
        "description": activity.description,
        "cost": activity.cost,
        "duration_hours": activity.duration_hours,
        "indoor": activity.indoor,
        "latitude": activity.latitude,
        "longitude": activity.longitude,
        "distance_m": activity.distance_m,
        "source": activity.source,
    }


def activity_from_dict(data: dict) -> Activity:
    return Activity(
        name=str(data.get("name") or ""),
        category=str(data.get("category") or "place"),
        description=str(data.get("description") or ""),
        cost=float(data.get("cost") or 0),
        duration_hours=float(data.get("duration_hours") or data.get("duration_h") or 1.5),
        indoor=bool(data.get("indoor")),
        latitude=_optional_float(data.get("latitude")),
        longitude=_optional_float(data.get("longitude")),
        distance_m=_optional_float(data.get("distance_m")),
        source=str(data.get("source") or "local"),
    )


def itinerary_to_dict(itinerary: Itinerary) -> dict:
    return {
        "destination": itinerary.destination,
        "currency": itinerary.currency,
        "total_cost": itinerary.total_cost,
        "days": [
            {
                "day": day.day,
                "total_cost": day.total_cost,
                "total_duration_hours": day.total_duration_hours,
                "notes": day.notes,
                "activities": [activity_to_dict(activity) for activity in day.activities],
            }
            for day in itinerary.days
        ],
    }


def itinerary_from_dict(data: dict) -> Itinerary:
    return Itinerary(
        destination=str(data.get("destination") or ""),
        currency=str(data.get("currency") or "EUR"),
        days=[
            ItineraryDay(
                day=int(day.get("day") or index + 1),
                activities=[activity_from_dict(activity) for activity in day.get("activities", [])],
                notes=[str(note) for note in day.get("notes", []) if str(note).strip()],
            )
            for index, day in enumerate(data.get("days", []))
            if isinstance(day, dict)
        ],
    )


def validation_to_dict(validation: ValidationResult) -> dict:
    return {
        "ok": validation.ok,
        "error_count": validation.error_count,
        "warning_count": validation.warning_count,
        "issues": [
            {
                "severity": issue.severity,
                "message": issue.message,
                "day": issue.day,
                "issue_type": issue.issue_type,
                "activity": issue.activity,
            }
            for issue in validation.issues
        ],
    }


def validation_from_dict(data: dict) -> ValidationResult:
    issues = [
        ValidationIssue(
            severity=str(issue.get("severity") or "warning"),
            message=str(issue.get("message") or ""),
            day=issue.get("day"),
            issue_type=str(issue.get("issue_type") or "general"),
            activity=issue.get("activity"),
        )
        for issue in data.get("issues", [])
        if isinstance(issue, dict)
    ]
    return ValidationResult(
        ok=bool(data.get("ok")) if "ok" in data else not issues,
        issues=issues,
        error_count=int(data.get("error_count") or sum(1 for issue in issues if issue.severity == "error")),
        warning_count=int(data.get("warning_count") or sum(1 for issue in issues if issue.severity == "warning")),
    )


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
