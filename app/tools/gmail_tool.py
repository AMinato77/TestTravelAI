from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.models.preference_source import PreferenceSource
from app.services.interest_taxonomy import normalize_interests, taxonomy_payload
from app.tools.openai_runtime import demo_fallback_enabled, generate_json


GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
DEFAULT_CREDENTIALS_FILE = Path("data/gmail_credentials.json")
DEFAULT_TOKEN_DIR = Path("data/gmail_tokens")
NEWSLETTER_QUERY_TERMS = [
    "newsletter",
    "travel",
    "reise",
    "trip",
    "flight",
    "hotel",
    "deal",
    "angebot",
    "event",
    "guide",
]
NEWSLETTER_LABELS = ["CATEGORY_PROMOTIONS", "CATEGORY_UPDATES", "CATEGORY_FORUMS"]
MAX_SOURCE_TEXT_CHARS = 8000


class GmailIntegrationError(RuntimeError):
    pass


@dataclass(slots=True)
class GmailNewsletterMessage:
    sender: str
    subject: str
    date: str
    snippet: str
    labels: list[str]
    list_id: str = ""
    list_unsubscribe: str = ""
    keep_as_preference: bool = False
    travel_relevance_score: float = 0.0
    signal_strength: str = "none"
    inferred_interests: list[str] = field(default_factory=list)
    budget_signal: str = ""
    travel_style_signal: str = ""
    avoid: list[str] = field(default_factory=list)
    preference_summary: str = ""
    ignore_reason: str = ""


def gmail_credentials_path() -> Path:
    return Path(os.getenv("GMAIL_CREDENTIALS_FILE", str(DEFAULT_CREDENTIALS_FILE)))


def gmail_token_dir() -> Path:
    return Path(os.getenv("GMAIL_TOKEN_DIR", str(DEFAULT_TOKEN_DIR)))


def gmail_token_path(user_id: str) -> Path:
    return gmail_token_dir() / f"{_safe_user_id(user_id)}.json"


def gmail_credentials_available() -> bool:
    return gmail_credentials_path().exists()


def gmail_user_connected(user_id: str) -> bool:
    return gmail_token_path(user_id).exists()


def save_gmail_credentials_file(uploaded_bytes: bytes) -> Path:
    return save_gmail_credentials(uploaded_bytes)


def save_gmail_credentials(uploaded_bytes: bytes) -> Path:
    if not uploaded_bytes:
        raise GmailIntegrationError("Keine Gmail OAuth-Credentials hochgeladen.")
    try:
        data = json.loads(uploaded_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GmailIntegrationError("Gmail Credentials muessen eine gueltige JSON-Datei sein.") from exc
    _validate_gmail_client_config(data)

    path = gmail_credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def connect_gmail_user(user_id: str) -> Path:
    creds_path = gmail_credentials_path()
    if not creds_path.exists():
        raise GmailIntegrationError(
            "Gmail OAuth-Credentials fehlen. Lade zuerst die client_secret JSON-Datei hoch."
        )
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise GmailIntegrationError(
            "Gmail-Abhaengigkeiten fehlen. Installiere google-api-python-client, "
            "google-auth und google-auth-oauthlib."
        ) from exc

    try:
        flow = InstalledAppFlow.from_client_config(_load_gmail_client_config(), scopes=[GMAIL_READONLY_SCOPE])
        credentials = flow.run_local_server(port=0, prompt="consent")
    except GmailIntegrationError:
        raise
    except Exception as exc:
        raise GmailIntegrationError(f"Gmail OAuth-Flow konnte nicht gestartet werden: {exc}") from exc
    token_path = gmail_token_path(user_id)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(credentials.to_json(), encoding="utf-8")
    return token_path


def get_gmail_account_email(user_id: str) -> str:
    service = _gmail_service(user_id, allow_oauth=True)
    try:
        profile = service.users().getProfile(userId="me").execute()
    except Exception as exc:
        raise GmailIntegrationError(f"Gmail-Kontoinfo konnte nicht geladen werden: {exc}") from exc
    return str(profile.get("emailAddress") or "")


def build_gmail_preference_source(
    user_id: str,
    max_messages: int = 20,
    lookback_days: int = 365,
    max_results: int | None = None,
    query: str = "",
) -> tuple[list[PreferenceSource], list[GmailNewsletterMessage]]:
    if max_results is not None:
        max_messages = max_results
    messages = fetch_gmail_newsletter_signals(
        user_id=user_id,
        max_messages=max_messages,
        lookback_days=lookback_days,
        query=query,
    )
    if not messages:
        return [], []
    classified_messages = _classify_newsletter_messages(messages)
    kept_messages = [message for message in classified_messages if message.keep_as_preference]
    if not kept_messages:
        return [], classified_messages
    source = PreferenceSource(
        source_type="email_newsletter",
        name="gmail_newsletter_signals",
        text=_signals_to_preference_text(kept_messages),
    )
    return [source], classified_messages


def fetch_gmail_newsletter_signals(
    user_id: str,
    max_messages: int = 20,
    lookback_days: int = 365,
    query: str = "",
) -> list[GmailNewsletterMessage]:
    service = _gmail_service(user_id, allow_oauth=True)
    message_ids = _search_newsletter_messages(
        service,
        max_results=max_messages,
        lookback_days=lookback_days,
        query=query,
    )
    signals: list[GmailNewsletterMessage] = []
    seen: set[str] = set()
    for message_id in message_ids:
        message = _get_message(service, message_id)
        signal = _message_to_signal(message)
        if not signal or not _is_newsletter_signal(signal):
            continue
        key = f"{signal.sender}|{signal.subject}|{signal.date}".lower()
        if key in seen:
            continue
        seen.add(key)
        signals.append(signal)
        if len(signals) >= max_messages:
            break
    return signals


def _gmail_service(user_id: str, allow_oauth: bool = False):
    token_path = gmail_token_path(user_id)
    creds_path = gmail_credentials_path()
    if not creds_path.exists():
        raise GmailIntegrationError("Gmail OAuth-Credentials fehlen.")
    if not token_path.exists():
        if not allow_oauth:
            raise GmailIntegrationError("Dieser User ist noch nicht mit Gmail verbunden.")
        connect_gmail_user(user_id)
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise GmailIntegrationError(
            "Gmail-Abhaengigkeiten fehlen. Installiere google-api-python-client, "
            "google-auth und google-auth-oauthlib."
        ) from exc

    credentials = Credentials.from_authorized_user_file(str(token_path), scopes=[GMAIL_READONLY_SCOPE])
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        token_path.write_text(credentials.to_json(), encoding="utf-8")
    if not credentials.valid:
        try:
            flow = InstalledAppFlow.from_client_config(_load_gmail_client_config(), scopes=[GMAIL_READONLY_SCOPE])
            credentials = flow.run_local_server(port=0, prompt="consent")
        except GmailIntegrationError:
            raise
        except Exception as exc:
            raise GmailIntegrationError(f"Gmail OAuth-Token konnte nicht erneuert werden: {exc}") from exc
        token_path.write_text(credentials.to_json(), encoding="utf-8")
    return build("gmail", "v1", credentials=credentials, cache_discovery=False)


def _load_gmail_client_config() -> dict:
    path = gmail_credentials_path()
    if not path.exists():
        raise GmailIntegrationError("Gmail OAuth-Credentials fehlen.")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise GmailIntegrationError(
            "Gmail Credentials JSON kann nicht gelesen werden. "
            "Lade die Google OAuth Desktop-App JSON-Datei erneut hoch."
        ) from exc
    _validate_gmail_client_config(data)
    return data


def _validate_gmail_client_config(data: object) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("installed"), dict):
        raise GmailIntegrationError("OAuth-Datei muss ein Google OAuth Desktop-App Client JSON enthalten.")
    installed = data["installed"]
    required = ["client_id", "client_secret", "auth_uri", "token_uri", "redirect_uris"]
    missing = [key for key in required if not installed.get(key)]
    if missing:
        raise GmailIntegrationError(
            "Gmail OAuth-Credentials sind unvollstaendig. Fehlende Felder: " + ", ".join(missing)
        )


def _search_newsletter_messages(service, max_results: int, lookback_days: int, query: str) -> list[str]:
    max_results = max(1, min(int(max_results or 20), 50))
    lookback_days = max(1, min(int(lookback_days or 365), 3650))
    queries = _gmail_queries(query=query, lookback_days=lookback_days)
    ids: list[str] = []
    seen: set[str] = set()
    for gmail_query in queries:
        try:
            result = (
                service.users()
                .messages()
                .list(userId="me", q=gmail_query, maxResults=max_results)
                .execute()
            )
        except Exception as exc:
            raise GmailIntegrationError(f"Gmail-Suche fehlgeschlagen: {exc}") from exc
        for item in result.get("messages", []) or []:
            message_id = item.get("id")
            if not message_id or message_id in seen:
                continue
            seen.add(message_id)
            ids.append(message_id)
            if len(ids) >= max_results * 2:
                return ids
    return ids


def _gmail_queries(query: str, lookback_days: int) -> list[str]:
    explicit = query.strip()
    if explicit:
        return [f"({explicit}) newer_than:{lookback_days}d"]
    queries = [
        f'category:{label.split("_", 1)[1].lower()} newer_than:{lookback_days}d'
        for label in NEWSLETTER_LABELS
    ]
    queries.extend(f'newer_than:{lookback_days}d {term}' for term in NEWSLETTER_QUERY_TERMS)
    queries.append(f'newer_than:{lookback_days}d (list-unsubscribe OR list-id)')
    return queries


def _get_message(service, message_id: str) -> dict:
    try:
        return (
            service.users()
            .messages()
            .get(
                userId="me",
                id=message_id,
                format="metadata",
                metadataHeaders=[
                    "From",
                    "Subject",
                    "Date",
                    "List-ID",
                    "List-Unsubscribe",
                    "Precedence",
                ],
            )
            .execute()
        )
    except Exception as exc:
        raise GmailIntegrationError(f"Gmail-Mail konnte nicht geladen werden: {exc}") from exc


def _message_to_signal(message: dict) -> GmailNewsletterMessage | None:
    headers = _headers_dict(message)
    sender = headers.get("from", "")
    subject = headers.get("subject", "")
    if not sender and not subject:
        return None
    return GmailNewsletterMessage(
        sender=_clean_text(sender),
        subject=_clean_text(subject),
        date=_clean_text(headers.get("date", "")),
        snippet=_clean_text(message.get("snippet", "")),
        labels=[str(label) for label in message.get("labelIds", []) or []],
        list_id=_clean_text(headers.get("list-id", "")),
        list_unsubscribe=_clean_text(headers.get("list-unsubscribe", "")),
    )


def _headers_dict(message: dict) -> dict[str, str]:
    payload = message.get("payload") or {}
    headers = payload.get("headers") or []
    return {str(item.get("name", "")).lower(): str(item.get("value", "")) for item in headers}


def _is_newsletter_signal(signal: GmailNewsletterMessage) -> bool:
    label_text = " ".join(signal.labels).lower()
    text = f"{signal.sender} {signal.subject} {signal.snippet} {signal.list_id} {signal.list_unsubscribe}".lower()
    if signal.list_id or signal.list_unsubscribe:
        return True
    if any(label.lower() in label_text for label in NEWSLETTER_LABELS):
        return True
    return any(term in text for term in NEWSLETTER_QUERY_TERMS)


def _classify_newsletter_messages(messages: list[GmailNewsletterMessage]) -> list[GmailNewsletterMessage]:
    if not messages:
        return []
    if demo_fallback_enabled():
        return [_classify_message_fallback(message) for message in messages]
    try:
        data = generate_json(
            system_prompt=(
                "You are a privacy-safe Gmail Preference Signal Classifier for a travel planner. "
                "You receive only email metadata and snippets, never full email bodies. "
                "Decide whether each newsletter contains a real travel preference signal. "
                "Ignore generic science newsletters, account confirmations, welcome emails, unsubscribe/admin emails, "
                "and generic shop or coupon emails unless they clearly express travel behavior. "
                "Do not infer nature from a brand name like 'Nature Briefing' unless the content is about travel/outdoor trips. "
                "Do not infer shopping from a shop sender, coupon, or newsletter signup alone. "
                "Keep only signals that can improve travel planning. "
                "Return strict JSON with key messages. Each item must contain: index, keep, travel_relevance_score "
                "from 0 to 1, signal_strength as none/weak/medium/strong, interests using only allowed_interests, "
                "budget_signal as low/medium/high/unknown, travel_style_signal as relaxed/adventure/luxury/budget/balanced/unknown, "
                "avoid as list, summary as one concise evidence-based sentence, and reason."
            ),
            payload={
                "messages": [
                    {
                        "index": index,
                        "sender": message.sender,
                        "subject": message.subject,
                        "date": message.date,
                        "labels": message.labels,
                        "snippet": message.snippet[:600],
                    }
                    for index, message in enumerate(messages, start=1)
                ],
                **taxonomy_payload(),
            },
            model_env="OPENAI_GMAIL_SIGNAL_MODEL",
        )
        return _apply_ai_classification(messages, data)
    except Exception:
        return [_classify_message_fallback(message) for message in messages]


def _apply_ai_classification(messages: list[GmailNewsletterMessage], data: dict) -> list[GmailNewsletterMessage]:
    rows = data.get("messages") if isinstance(data.get("messages"), list) else []
    by_index: dict[int, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            index = int(row.get("index"))
        except (TypeError, ValueError):
            continue
        by_index[index] = row

    classified: list[GmailNewsletterMessage] = []
    for index, message in enumerate(messages, start=1):
        row = by_index.get(index)
        if not row:
            classified.append(_classify_message_fallback(message))
            continue
        fallback = _classify_message_fallback(_copy_message(message))
        score = _score(row.get("travel_relevance_score"))
        interests = normalize_interests(row.get("interests") or []) or fallback.inferred_interests
        summary = _clean_text(str(row.get("summary") or ""))
        reason = _clean_text(str(row.get("reason") or ""))
        keep = bool(row.get("keep")) and score >= 0.45 and bool(summary)
        if not fallback.keep_as_preference and _is_admin_or_noise_reason(fallback.ignore_reason):
            keep = False
        message.keep_as_preference = keep
        message.travel_relevance_score = score
        message.signal_strength = _normalize_signal_strength(row.get("signal_strength"), score)
        message.inferred_interests = interests
        message.budget_signal = _normalize_choice(row.get("budget_signal"), {"low", "medium", "high", "unknown"}, "unknown")
        message.travel_style_signal = _normalize_choice(
            row.get("travel_style_signal"),
            {"relaxed", "adventure", "luxury", "budget", "balanced", "unknown"},
            "unknown",
        )
        message.avoid = [str(item).strip().lower() for item in row.get("avoid") or [] if str(item).strip()]
        message.preference_summary = summary if keep else ""
        message.ignore_reason = "" if keep else fallback.ignore_reason or reason or "AI classifier did not find a reliable travel preference signal."
        classified.append(message)
    return classified


def _copy_message(message: GmailNewsletterMessage) -> GmailNewsletterMessage:
    return GmailNewsletterMessage(
        sender=message.sender,
        subject=message.subject,
        date=message.date,
        snippet=message.snippet,
        labels=list(message.labels),
        list_id=message.list_id,
        list_unsubscribe=message.list_unsubscribe,
    )


def _is_admin_or_noise_reason(reason: str) -> bool:
    lower = reason.lower()
    return any(
        term in lower
        for term in [
            "anmeldung",
            "bestaetigung",
            "begrüessung",
            "begrussung",
            "administrative",
            "wissenschaftlich",
            "nicht wie ein reise",
            "kein ausreichend klares",
        ]
    )


def _classify_message_fallback(message: GmailNewsletterMessage) -> GmailNewsletterMessage:
    text = f"{message.sender} {message.subject} {message.snippet}".lower()
    admin_terms = [
        "confirm",
        "confirmation",
        "bestätige",
        "bestaetige",
        "anmeldung",
        "welcome",
        "willkommen",
        "unsubscribe",
        "verify",
        "email-adresse",
        "e-mail-adresse",
    ]
    non_travel_science_terms = [
        "nature briefing",
        "human bones",
        "fossil",
        "fossilized",
        "venus flytrap",
        "squirrels",
        "science",
        "scientists",
    ]
    travel_terms = [
        "reise",
        "travel",
        "trip",
        "urlaub",
        "landurlaub",
        "hotel",
        "flight",
        "flug",
        "city break",
        "strand",
        "beach",
        "sommerurlaub",
        "destination",
        "reisentipps",
        "guide",
    ]
    if any(term in text for term in non_travel_science_terms) and not any(term in text for term in travel_terms):
        return _ignored_message(message, "Newsletter wirkt wissenschaftlich/inhaltlich, aber nicht wie ein Reisepräferenzsignal.")
    if any(term in text for term in admin_terms) and not any(term in text for term in ["hotel", "urlaub", "reiseangebot", "travel deal"]):
        return _ignored_message(message, "Newsletter ist primaer Anmeldung, Bestaetigung oder administrative Begruessung.")
    if not any(term in text for term in travel_terms):
        return _ignored_message(message, "Kein ausreichend klares Reisepräferenzsignal gefunden.")

    interests: list[str] = []
    if any(term in text for term in ["restaurant", "food", "cuisine", "essen", "tapas", "street food"]):
        interests.append("food")
    if any(
        term in text
        for term in [
            "beach",
            "strand",
            "hiking",
            "wandern",
            "outdoor",
            "park",
            "garten",
            "landurlaub",
            "baumhaushotel",
            "landreise",
            "natururlaub",
        ]
    ):
        interests.append("nature")
    if any(term in text for term in ["museum", "gallery", "festival", "culture", "kultur", "kunst"]):
        interests.append("culture")
    if any(term in text for term in ["hidden gem", "geheimtipp", "local", "lokal", "viertel"]):
        interests.extend(["hidden gems", "local spots"])
    if any(term in text for term in ["shopping street", "mall", "einkaufen", "shopping"]):
        interests.append("shopping")

    budget_signal = "unknown"
    if any(term in text for term in ["deal", "angebot", "rabatt", "discount", "%", "gutschein", "cheap", "günstig", "guenstig"]):
        budget_signal = "low"
    travel_style_signal = "unknown"
    if any(term in text for term in ["ohne stress", "entspannt", "relaxed", "landurlaub", "autopacken", "auto packen"]):
        travel_style_signal = "relaxed"

    message.keep_as_preference = True
    message.travel_relevance_score = 0.65 if interests or budget_signal != "unknown" or travel_style_signal != "unknown" else 0.5
    message.signal_strength = "medium" if message.travel_relevance_score >= 0.6 else "weak"
    message.inferred_interests = normalize_interests(interests)
    message.budget_signal = budget_signal
    message.travel_style_signal = travel_style_signal
    message.preference_summary = _build_fallback_summary(message)
    message.ignore_reason = ""
    return message


def _ignored_message(message: GmailNewsletterMessage, reason: str) -> GmailNewsletterMessage:
    message.keep_as_preference = False
    message.travel_relevance_score = 0.0
    message.signal_strength = "none"
    message.inferred_interests = []
    message.budget_signal = "unknown"
    message.travel_style_signal = "unknown"
    message.preference_summary = ""
    message.ignore_reason = reason
    return message


def _build_fallback_summary(message: GmailNewsletterMessage) -> str:
    parts = []
    if message.inferred_interests:
        parts.append(f"possible interests: {', '.join(message.inferred_interests)}")
    if message.budget_signal != "unknown":
        parts.append(f"budget signal: {message.budget_signal}")
    if message.travel_style_signal != "unknown":
        parts.append(f"travel style signal: {message.travel_style_signal}")
    evidence = message.subject or message.snippet[:120] or message.sender
    return f"{'; '.join(parts) or 'travel newsletter signal'} based on '{evidence}'."


def _signals_to_preference_text(signals: list[GmailNewsletterMessage]) -> str:
    lines = [
        "Gmail newsletter preference signals after relevance classification.",
        "Only metadata, snippets, and extracted travel preference summaries are included; full email bodies are not stored.",
        f"Kept travel-relevant signal count: {len(signals)}",
        "",
    ]
    for index, signal in enumerate(signals, start=1):
        lines.extend(
            [
                f"{index}. Sender: {signal.sender or 'unknown'}",
                f"Subject: {signal.subject or 'unknown'}",
                f"Date: {signal.date or 'unknown'}",
                f"Travel relevance: {signal.travel_relevance_score:.2f} ({signal.signal_strength})",
                f"Interests: {', '.join(signal.inferred_interests) or 'none'}",
                f"Budget signal: {signal.budget_signal or 'unknown'}",
                f"Travel style signal: {signal.travel_style_signal or 'unknown'}",
                f"Avoid: {', '.join(signal.avoid) or 'none'}",
                f"Evidence summary: {signal.preference_summary or 'none'}",
                "",
            ]
        )
    return "\n".join(lines)[:MAX_SOURCE_TEXT_CHARS]


def _infer_weak_signal_tags(signal: GmailNewsletterMessage) -> list[str]:
    if signal.inferred_interests:
        return signal.inferred_interests
    return _classify_message_fallback(signal).inferred_interests


def _score(value) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _normalize_signal_strength(value, score: float) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"none", "weak", "medium", "strong"}:
        return normalized
    if score >= 0.8:
        return "strong"
    if score >= 0.55:
        return "medium"
    if score >= 0.35:
        return "weak"
    return "none"


def _normalize_choice(value, allowed: set[str], fallback: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else fallback


def _clean_text(value: str) -> str:
    decoded = _decode_rfc2047(value)
    decoded = re.sub(r"\s+", " ", decoded)
    return decoded.strip()


def _decode_rfc2047(value: str) -> str:
    try:
        import email.header

        parts = email.header.decode_header(value)
        decoded: list[str] = []
        for text, charset in parts:
            if isinstance(text, bytes):
                decoded.append(text.decode(charset or "utf-8", errors="ignore"))
            else:
                decoded.append(text)
        return "".join(decoded)
    except Exception:
        return value


def _safe_user_id(user_id: str) -> str:
    safe = "".join(char for char in user_id if char.isalnum() or char in ("-", "_")).strip()
    return safe or "demo_user_1"


GmailNewsletterSignal = GmailNewsletterMessage


def decode_pubsub_message(data: str) -> dict:
    """Small helper for future Gmail push notifications; unused by Streamlit."""
    raw = base64.urlsafe_b64decode(data.encode("utf-8"))
    return json.loads(raw.decode("utf-8"))
