from __future__ import annotations

import json
import os
from pathlib import Path


CALENDAR_EVENTS_SCOPE = "https://www.googleapis.com/auth/calendar.events"
CALENDAR_MANAGE_SCOPE = "https://www.googleapis.com/auth/calendar"
CALENDAR_SCOPES = [CALENDAR_MANAGE_SCOPE]
DEFAULT_CREDENTIALS_FILE = Path("data/gmail_credentials.json")
DEFAULT_TOKEN_DIR = Path("data/calendar_tokens")


class CalendarIntegrationError(RuntimeError):
    pass


def calendar_credentials_path() -> Path:
    return Path(os.getenv("CALENDAR_CREDENTIALS_FILE", os.getenv("GMAIL_CREDENTIALS_FILE", str(DEFAULT_CREDENTIALS_FILE))))


def calendar_token_dir() -> Path:
    return Path(os.getenv("CALENDAR_TOKEN_DIR", str(DEFAULT_TOKEN_DIR)))


def calendar_token_path(user_id: str) -> Path:
    return calendar_token_dir() / f"{_safe_user_id(user_id)}.json"


def calendar_credentials_available() -> bool:
    return calendar_credentials_path().exists()


def calendar_user_connected(user_id: str) -> bool:
    return calendar_token_path(user_id).exists()


def reset_calendar_user(user_id: str) -> None:
    token_path = calendar_token_path(user_id)
    if token_path.exists():
        token_path.unlink()


def connect_calendar_user(user_id: str) -> Path:
    if not calendar_credentials_available():
        raise CalendarIntegrationError("Google OAuth-Credentials fehlen. Lade zuerst die OAuth Client JSON-Datei hoch.")
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise CalendarIntegrationError("Google Calendar-Abhängigkeiten fehlen.") from exc

    try:
        flow = InstalledAppFlow.from_client_config(_load_client_config(), scopes=CALENDAR_SCOPES)
        credentials = flow.run_local_server(port=0, prompt="consent")
    except Exception as exc:
        raise CalendarIntegrationError(f"Google Calendar-Verbindung konnte nicht gestartet werden: {exc}") from exc

    token_path = calendar_token_path(user_id)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(credentials.to_json(), encoding="utf-8")
    return token_path


def calendar_credentials(user_id: str, allow_oauth: bool = False):
    token_path = calendar_token_path(user_id)
    if not calendar_credentials_available():
        raise CalendarIntegrationError("Google OAuth-Credentials fehlen.")
    if not token_path.exists():
        if not allow_oauth:
            raise CalendarIntegrationError("Google Calendar ist für dieses Profil noch nicht verbunden.")
        connect_calendar_user(user_id)

    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError as exc:
        raise CalendarIntegrationError("Google Calendar-Abhängigkeiten fehlen.") from exc

    credentials = Credentials.from_authorized_user_file(str(token_path), scopes=CALENDAR_SCOPES)
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        token_path.write_text(credentials.to_json(), encoding="utf-8")
    if not credentials.valid:
        if not allow_oauth:
            raise CalendarIntegrationError("Google Calendar-Zugriff ist abgelaufen. Bitte erneut verbinden.")
        connect_calendar_user(user_id)
        credentials = Credentials.from_authorized_user_file(str(token_path), scopes=CALENDAR_SCOPES)
    return credentials


def _load_client_config() -> dict:
    path = calendar_credentials_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalendarIntegrationError("OAuth Client JSON konnte nicht gelesen werden.") from exc
    _validate_client_config(data)
    return data


def _validate_client_config(data: object) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("installed"), dict):
        raise CalendarIntegrationError("OAuth-Datei muss ein Google OAuth Desktop-App Client JSON enthalten.")
    installed = data["installed"]
    missing = [key for key in ["client_id", "client_secret", "auth_uri", "token_uri", "redirect_uris"] if not installed.get(key)]
    if missing:
        raise CalendarIntegrationError("OAuth-Credentials sind unvollständig. Fehlende Felder: " + ", ".join(missing))


def _safe_user_id(user_id: str) -> str:
    safe = "".join(char for char in str(user_id) if char.isalnum() or char in ("-", "_")).strip()
    return safe or "demo_user_1"
