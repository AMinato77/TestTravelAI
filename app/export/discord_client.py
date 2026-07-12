from __future__ import annotations

import os
from typing import Any

import requests

from app.export.export_context import discord_summary


class DiscordDeliveryError(RuntimeError):
    pass


def send_itinerary_to_discord(*, plan: Any, pdf_bytes: bytes, filename: str) -> None:
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        raise DiscordDeliveryError("DISCORD_WEBHOOK_URL ist nicht konfiguriert.")
    if not pdf_bytes:
        raise DiscordDeliveryError("PDF-Anhang ist leer.")

    try:
        response = requests.post(
            webhook_url,
            data={"content": discord_summary(plan)},
            files={"files[0]": (filename, pdf_bytes, "application/pdf")},
            timeout=30,
        )
    except requests.Timeout as exc:
        raise DiscordDeliveryError("Discord-Versand ist wegen Timeout fehlgeschlagen.") from exc
    except requests.RequestException as exc:
        raise DiscordDeliveryError("Discord-Versand ist fehlgeschlagen. Bitte Webhook und Netzwerk pruefen.") from exc

    if response.status_code not in {200, 204}:
        raise DiscordDeliveryError(
            f"Discord-Versand fehlgeschlagen: HTTP {response.status_code}."
        )
