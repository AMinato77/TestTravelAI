from __future__ import annotations

import re

from app.models.itinerary import Itinerary, ValidationIssue
from app.services.destination_normalizer import destination_matches_text, normalize_destination
from app.services.wish_matching import activity_covers_wish, activity_text


def semantic_issues(itinerary: Itinerary, constraints: dict | None = None) -> list[ValidationIssue]:
    """Validate request fulfillment beyond budget/weather mechanics."""
    constraints = constraints or {}
    issues: list[ValidationIssue] = []
    must_have = _clean_terms(constraints.get("must_have") or [])
    avoid = _clean_terms(constraints.get("avoid") or [])
    destination = normalize_destination(str(constraints.get("destination") or itinerary.destination))
    activities = [activity for day in itinerary.days for activity in day.activities]

    for wish in must_have:
        if not any(_activity_covers_wish(activity, wish) for activity in activities):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    issue_type="must_have_gap",
                    message=f"Must-have wish is not clearly covered: {wish}.",
                )
            )

    for day in itinerary.days:
        for activity in day.activities:
            text = _activity_text(activity)
            if destination and not _activity_matches_destination(activity, destination):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        issue_type="destination_mismatch",
                        activity=activity.name,
                        day=day.day,
                        message=f"Activity appears to be outside the destination {destination}.",
                    )
                )
            for term in avoid:
                if _text_conflicts_with_avoid(text, term):
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            issue_type="semantic_avoid_conflict",
                            activity=activity.name,
                            day=day.day,
                            message=f"Activity appears to conflict with avoid request: {term}.",
                        )
                    )

    return issues


def _activity_text(activity) -> str:
    return activity_text(activity)


def _activity_covers_wish(activity, wish: str) -> bool:
    return activity_covers_wish(activity, wish)


def _activity_matches_destination(activity, destination: str) -> bool:
    address = _description_field(activity.description, "Address")
    if address:
        return destination_matches_text(destination, address)
    return destination_matches_text(destination, activity.description)


def _description_field(description: str, label: str) -> str:
    marker = f"{label}:"
    for part in str(description or "").split("|"):
        cleaned = part.strip()
        if cleaned.lower().startswith(marker.lower()):
            return cleaned.split(":", 1)[1].strip()
    return ""


def _matching_description(description: str) -> str:
    kept: list[str] = []
    blocked_labels = {"matched query", "matched must-have", "google maps", "website"}
    for part in str(description or "").split("|"):
        cleaned = part.strip()
        label = cleaned.split(":", 1)[0].strip().lower() if ":" in cleaned else ""
        if label in blocked_labels:
            continue
        if cleaned.lower().startswith(("http://", "https://")):
            continue
        kept.append(cleaned)
    return " | ".join(kept)


def _matched_must_have_covers(description: str, wish: str) -> bool:
    matched = _description_field(description, "Matched must-have")
    wanted = " ".join(str(wish or "").lower().split())
    if not matched or not wanted:
        return False
    return any(" ".join(part.lower().split()) == wanted for part in matched.split(","))


def _text_matches_wish(text: str, wish: str) -> bool:
    tokens = _content_tokens(wish)
    if not tokens:
        return False
    exact = " ".join(tokens)
    if exact and exact in text:
        return True
    matches = sum(1 for token in tokens if _token_in_text(token, text))
    threshold = 1 if len(tokens) == 1 else max(2, round(len(tokens) * 0.55))
    return matches >= threshold


def _text_conflicts_with_avoid(text: str, avoid_term: str) -> bool:
    normalized = " ".join(str(avoid_term).lower().split())
    if not normalized:
        return False
    if normalized in text:
        return True
    tokens = _content_tokens(normalized)
    if not tokens:
        return False
    matches = sum(1 for token in tokens if _token_in_text(token, text))
    if len(tokens) <= 2:
        return matches == len(tokens)
    return matches / len(tokens) >= 0.8


def _token_in_text(token: str, text: str) -> bool:
    if token in text:
        return True
    variants = {token}
    if token.endswith("ies") and len(token) > 4:
        variants.add(f"{token[:-3]}y")
    if token.endswith("es") and len(token) > 4:
        variants.add(token[:-2])
    if token.endswith("s") and len(token) > 3:
        variants.add(token[:-1])
    return any(variant and variant in text for variant in variants)


def _clean_terms(values) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value).strip().lower().split())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _content_tokens(text: str) -> list[str]:
    stop_words = {
        "the", "and", "for", "with", "from", "und", "oder", "mit", "von", "fuer", "für",
        "eine", "einen", "der", "die", "das", "zu", "in", "im", "am", "an", "places",
        "things", "activity", "activities", "visit", "trip", "travel",
    }
    return [
        token
        for token in re.findall(r"[a-z0-9äöüß]+", text.lower())
        if len(token) > 2 and token not in stop_words
    ]
