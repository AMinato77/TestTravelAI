from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from datetime import date
from typing import Any

from app.export import travel_copy_adapter as copy


COUNTRY_ALIASES = {
    "madrid": {"spain", "spanien", "españa", "espana"},
    "barcelona": {"spain", "spanien", "españa", "espana"},
    "paris": {"france", "frankreich"},
    "berlin": {"germany", "deutschland"},
    "hamburg": {"germany", "deutschland"},
    "munich": {"germany", "deutschland"},
    "münchen": {"germany", "deutschland"},
    "rome": {"italy", "italien"},
    "rom": {"italy", "italien"},
    "athens": {"greece", "griechenland"},
    "athen": {"greece", "griechenland"},
    "prague": {"czechia", "czech republic", "tschechien"},
    "prag": {"czechia", "czech republic", "tschechien"},
}
KNOWN_COUNTRIES = set().union(*COUNTRY_ALIASES.values()) | {
    "pakistan",
    "india",
    "usa",
    "united states",
    "united kingdom",
    "uk",
    "france",
    "germany",
    "italy",
    "spain",
    "greece",
    "czechia",
}


def build_pdf_context(plan: Any) -> dict[str, Any]:
    data = _to_plain(plan)
    itinerary = data.get("itinerary", data)
    validation = data.get("validation") or {}
    request = data.get("request") or data.get("parsed_request") or {}
    tool_workflow = data.get("agentic_tool_workflow") or {}

    days = itinerary.get("days") or []
    currency = itinerary.get("currency") or request.get("currency") or "EUR"
    total_cost = _number(itinerary.get("total_cost"))
    maximum_budget = _number(request.get("budget") or data.get("budget"))
    destination = _clean(itinerary.get("destination") or request.get("destination") or data.get("destination") or "Reise")

    rendered_days = []
    maps_links = []
    category_totals: dict[str, float] = {}
    for day in days:
        rendered_day = _render_day(day, currency, destination)
        rendered_days.append(rendered_day)
        maps_links.extend(link for link in rendered_day["maps_links"] if link.get("url"))
        for activity in rendered_day["activities"]:
            category = activity["category"]
            category_totals[category] = category_totals.get(category, 0) + (activity.get("estimated_cost") or 0)

    ok = bool(validation.get("ok", not validation.get("issues")))
    highlights = _clean_list(request.get("must_have") or data.get("must_have"), copy.highlight_text)
    not_planned = _clean_list(request.get("avoid") or data.get("avoid"), copy.avoid_text)
    summary = copy.clean_public_text((data.get("explanation") or {}).get("summary") or data.get("summary") or "")

    return {
        "destination": destination,
        "destination_initial": destination[:1].upper(),
        "duration_days": len(rendered_days) or _int(request.get("duration_days") or data.get("duration_days"), 0),
        "created_date": date.today().strftime("%d.%m.%Y"),
        "travel_style": _style_label(request.get("travel_style") or data.get("travel_style") or "balanced"),
        "currency": currency,
        "maximum_budget": maximum_budget,
        "maximum_budget_label": _format_money(maximum_budget, currency) if maximum_budget is not None else "",
        "estimated_total_cost": total_cost,
        "estimated_total_cost_label": _format_money(total_cost, currency) if total_cost is not None else "",
        "budget_story": copy.planning_text(ok, total_cost, maximum_budget, currency),
        "weather_summary": _weather_summary(data, tool_workflow),
        "highlights": highlights,
        "not_planned": not_planned,
        "validation": _validation_context(validation),
        "planning_note": copy.planning_text(ok, total_cost, maximum_budget, currency),
        "summary": _truncate(summary, 420),
        "days": rendered_days,
        "maps_links": maps_links,
        "notes": copy.final_notes(total_cost, maximum_budget, currency),
        "category_totals": [
            {"category": category, "amount": amount, "amount_label": _format_money(amount, currency)}
            for category, amount in sorted(category_totals.items(), key=lambda item: item[1], reverse=True)
            if amount > 0
        ],
        "plan_stats": copy.plan_stats(rendered_days, total_cost, maximum_budget, currency),
    }


def discord_summary(plan: Any) -> str:
    context = build_pdf_context(plan)
    lines = [
        f"**Reiseplan: {context['destination']}**",
        f"Dauer: {context['duration_days']} Tag(e)",
    ]
    if context.get("estimated_total_cost_label"):
        lines.append(f"Geplante Ausgaben: {context['estimated_total_cost_label']}")
    lines.append("")
    for day in context["days"][:4]:
        names = ", ".join(activity["name"] for activity in day["activities"][:4])
        if names:
            lines.append(f"Tag {day['number']}: {names}")
    lines.append("")
    lines.append("Der vollständige Reiseplan befindet sich in der PDF.")
    return "\n".join(lines)


def validate_export_context(context: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    destination = _clean(context.get("destination"))
    expected_countries = _expected_countries(destination)
    seen_names: set[str] = set()

    if not context.get("days"):
        errors.append("Der Reiseplan enthält keine Reisetage.")

    for day in context.get("days") or []:
        activities = day.get("activities") or []
        if not activities:
            errors.append(f"Tag {day.get('number', '?')} enthält keine Aktivitäten.")
        for activity in activities:
            name = _clean(activity.get("name"))
            if not name:
                errors.append(f"Tag {day.get('number', '?')} enthält eine Aktivität ohne Namen.")
                continue
            key = name.lower()
            if key in seen_names:
                errors.append(f"Die Aktivität '{name}' kommt mehrfach im Plan vor.")
            seen_names.add(key)

            maps_url = _clean(activity.get("maps_url"))
            if maps_url and not maps_url.startswith(("http://", "https://")):
                errors.append(f"Die Aktivität '{name}' hat keinen gültigen Google-Maps-Link.")

            address = _clean(activity.get("address"))
            mismatched_country = _destination_country_mismatch(address, expected_countries)
            if mismatched_country:
                errors.append(
                    f"'{name}' scheint nicht zum Reiseziel {destination} zu passen "
                    f"(Adresse enthält {mismatched_country})."
                )

    validation = context.get("validation") or {}
    if not validation.get("ok", True):
        errors.append("Der Reiseplan enthält noch kritische Hinweise.")
    return errors


def _render_day(day: dict[str, Any], currency: str, destination: str) -> dict[str, Any]:
    activities = day.get("activities") or []
    current_minutes = 9 * 60
    rendered_activities = []
    maps_links = []
    for index, activity in enumerate(activities, start=1):
        duration = max(0.5, _number(activity.get("duration_hours"), 1.5) or 1.5)
        start_time = _clock(current_minutes)
        current_minutes += int(duration * 60)
        end_time = _clock(current_minutes)
        current_minutes += 30

        details = _description_details(activity.get("description") or "")
        maps_url = details.get("google_maps") or ""
        cost = _number(activity.get("cost"), 0)
        category = copy.category_label(activity, details)
        rendered = {
            "index": index,
            "start_time": start_time,
            "end_time": end_time,
            "time_label": f"{start_time}–{end_time}",
            "name": _clean(activity.get("name") or "Stopp"),
            "category": category,
            "category_key": copy.category_key(category),
            "duration_hours": duration,
            "duration_label": _format_duration(duration),
            "estimated_cost": cost,
            "cost_label": _format_money(cost, currency) if cost is not None else "",
            "currency": currency,
            "rating": _format_rating(details.get("rating", "")),
            "reviews": _format_reviews(details.get("reviews", "")),
            "address": details.get("address", ""),
            "maps_url": maps_url,
            "website": details.get("website", ""),
            "reason": copy.activity_description(activity, details, destination),
            "visual_label": category,
        }
        rendered_activities.append(rendered)
        if maps_url:
            maps_links.append({"name": rendered["name"], "url": maps_url})

    total_cost = _number(day.get("total_cost"), sum(item["estimated_cost"] or 0 for item in rendered_activities))
    total_duration = _number(day.get("total_duration_hours"), sum(item["duration_hours"] for item in rendered_activities))
    number = _int(day.get("day"), 1)
    return {
        "number": number,
        "title": copy.day_title(number, rendered_activities, destination),
        "activities": rendered_activities,
        "total_cost": total_cost,
        "total_cost_label": _format_money(total_cost, currency) if total_cost is not None else "",
        "total_duration_hours": total_duration,
        "total_duration_label": _format_duration(total_duration) if total_duration is not None else "",
        "notes": _clean_list(day.get("notes"), copy.clean_public_text),
        "maps_links": maps_links,
    }


def _description_details(description: str) -> dict[str, str]:
    details = {
        "address": _extract_between(description, "Address:"),
        "rating": _extract_between(description, "Rating:"),
        "reviews": _extract_between(description, "Reviews:"),
        "matched_must_have": _extract_between(description, "Matched must-have:"),
        "website": _extract_url(description, "Website:"),
        "google_maps": _extract_url(description, "Google Maps:"),
    }
    return {key: _clean(value) for key, value in details.items() if _clean(value)}


def _extract_between(text: str, marker: str) -> str:
    if marker not in text:
        return ""
    return text.split(marker, 1)[1].split("|", 1)[0].strip()


def _extract_url(text: str, marker: str) -> str:
    value = _extract_between(text, marker)
    return value if value.startswith(("http://", "https://")) else ""


def _validation_context(validation: dict[str, Any]) -> dict[str, Any]:
    issues = validation.get("issues") or []
    return {
        "ok": bool(validation.get("ok", not issues)),
        "error_count": _int(validation.get("error_count"), 0),
        "warning_count": _int(validation.get("warning_count"), 0),
        "issues": [
            {
                "severity": _clean(issue.get("severity") or "warning"),
                "message": copy.clean_public_text(issue.get("message") or ""),
                "issue_type": _clean(issue.get("issue_type") or ""),
            }
            for issue in issues
            if isinstance(issue, dict) and _clean(issue.get("message") or "")
        ],
    }


def _weather_summary(data: dict[str, Any], tool_workflow: dict[str, Any]) -> str:
    weather = data.get("weather_summary") or {}
    if isinstance(weather, dict) and weather.get("summary"):
        return _clean(weather["summary"])
    if isinstance(weather, str):
        return _clean(weather)
    for call in tool_workflow.get("tool_calls") or []:
        result = call.get("result") or {}
        if isinstance(result, dict) and result.get("summary") and "weather" in str(call.get("tool", "")).lower():
            return _clean(result["summary"])
    return ""


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return _to_plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _to_plain(value.to_dict())
    return str(value)


def _clean(value: Any) -> str:
    text = " ".join(str(value or "").replace("\x00", "").split())
    return "" if text.lower() in {"none", "null", "nan"} else text


def _clean_list(values: Any, formatter) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    result = []
    seen = set()
    for value in values:
        cleaned = formatter(_clean(value))
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clock(minutes: int) -> str:
    hour = (minutes // 60) % 24
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"


def _style_label(style: str) -> str:
    labels = {
        "balanced": "Ausgewogen",
        "relaxed": "Entspannt",
        "adventure": "Abenteuer",
        "luxury": "Luxus",
        "budget": "Günstig",
    }
    return labels.get(_clean(style).lower(), _clean(style) or "Ausgewogen")


def _format_money(value: float | None, currency: str) -> str:
    if value is None:
        return ""
    return f"{value:,.0f} {currency}".replace(",", ".")


def _format_duration(hours: float | None) -> str:
    if hours is None:
        return ""
    if abs(hours - 1) < 0.01:
        return "1 Stunde"
    formatted = f"{hours:.1f}".replace(".", ",").rstrip("0").rstrip(",")
    return f"{formatted} Stunden"


def _format_rating(value: str) -> str:
    return _clean(value).replace("/5", "")


def _format_reviews(value: str) -> str:
    cleaned = _clean(value)
    return f"{cleaned} Bewertungen" if cleaned and "bewertung" not in cleaned.lower() else cleaned


def _truncate(text: str, limit: int) -> str:
    cleaned = _clean(text)
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"


def _expected_countries(destination: str) -> set[str]:
    normalized = destination.lower()
    for city, countries in COUNTRY_ALIASES.items():
        if city in normalized:
            return countries
    return set()


def _destination_country_mismatch(address: str, expected_countries: set[str]) -> str:
    if not address or not expected_countries:
        return ""
    lowered = address.lower()
    mentioned = [country for country in KNOWN_COUNTRIES if re.search(rf"\b{re.escape(country)}\b", lowered)]
    for country in mentioned:
        if country not in expected_countries:
            return country.title()
    return ""
