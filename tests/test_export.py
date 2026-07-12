from __future__ import annotations

import pytest
import requests
import re
from html import unescape

from app.export import discord_client, export_service, pdf_generator
from app.export.discord_client import DiscordDeliveryError
from app.export.export_context import build_pdf_context, validate_export_context


BANNED_VISIBLE_TERMS = [
    "validation",
    "validiert",
    "validierter",
    "quality score",
    "required experience",
    "required experiences",
    "avoid",
    "matched",
    "candidate",
    "query",
    "memory",
    "budget utilization",
    "FOOD",
    "TRIP",
    "nature experiences",
    "Planprüfung",
    "Qualitätsprüfung",
    "Bestanden",
]


def sample_plan() -> dict:
    return {
        "itinerary": {
            "destination": "Madrid",
            "currency": "EUR",
            "total_cost": 100,
            "days": [
                {
                    "day": 1,
                    "total_cost": 100,
                    "total_duration_hours": 3.5,
                    "notes": ["Reservierung empfohlen."],
                    "activities": [
                        {
                            "name": "Mercado de San Miguel",
                            "category": "shopping",
                            "description": (
                                "Category: shopping | Matched query: Madrid food markets | "
                                "Matched must-have: typical Spanish cuisine | Address: Plaza de San Miguel, Madrid, Spain | "
                                "Rating: 4.4/5 | Reviews: 12000 | Website: https://example.com | "
                                "Google Maps: https://maps.google.com/?cid=123"
                            ),
                            "cost": 20,
                            "duration_hours": 2,
                        }
                    ],
                }
            ],
        },
        "validation": {"ok": True, "error_count": 0, "warning_count": 0, "issues": []},
        "request": {
            "destination": "Madrid",
            "duration_days": 1,
            "budget": 350,
            "must_have": ["typical Spanish cuisine"],
            "avoid": ["clubs"],
            "travel_style": "balanced",
        },
        "weather_summary": {"summary": "Madrid: Sonnig."},
        "explanation": {"summary": "Ein kompakter Madrid-Plan."},
    }


class FakeBrowser:
    def new_page(self, viewport=None):
        self.viewport = viewport
        return FakePage()

    def close(self):
        self.closed = True


class FakeChromium:
    def launch(self, headless=True):
        self.headless = headless
        return FakeBrowser()


class FakePlaywright:
    def __init__(self):
        self.chromium = FakeChromium()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakePage:
    def set_content(self, html, wait_until=None):
        self.html = html
        self.wait_until = wait_until

    def emulate_media(self, media):
        self.media = media

    def pdf(self, **kwargs):
        self.pdf_kwargs = kwargs
        return b"%PDF-PLAYWRIGHT" + (b"0" * 10000)


def test_build_pdf_context_full_data():
    context = build_pdf_context(sample_plan())
    assert context["destination"] == "Madrid"
    assert context["highlights"] == ["authentische lokale Küche"]
    assert context["not_planned"] == ["Nachtclubs und lautes Partyprogramm"]
    activity = context["days"][0]["activities"][0]
    assert activity["address"] == "Plaza de San Miguel, Madrid, Spain"
    assert activity["maps_url"].startswith("https://maps.google.com")
    assert "Tapas" in activity["reason"] or "Spezialitäten" in activity["reason"]
    assert activity["cost_label"] == "20 EUR"


def test_build_pdf_context_missing_optional_data():
    context = build_pdf_context({"itinerary": {"destination": "Rome", "days": []}, "validation": {"ok": True}})
    assert context["destination"] == "Rome"
    assert context["weather_summary"] == ""
    assert context["highlights"] == []


def test_build_html_preserves_unicode_and_public_copy():
    context = build_pdf_context(sample_plan())
    context["destination"] = "München, Paris & Café - 50 €"
    html = pdf_generator.build_html(context)
    assert "München, Paris &amp; Café - 50 €" in html
    assert "Dein persönlicher Reiseplan" in html
    assert "Darauf darfst du dich freuen" in html
    assert "Bewusst nicht eingeplant" in html
    assert "<meta charset=\"UTF-8\">" in html
    assert "pdf_css" not in html


def test_rendered_html_has_no_internal_terms():
    html = pdf_generator.build_html(build_pdf_context(sample_plan()))
    visible_html = re.sub(r"<style.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    visible_text = unescape(re.sub(r"<[^>]+>", " ", visible_html))
    html_lower = " ".join(visible_text.split()).lower()
    for term in BANNED_VISIBLE_TERMS:
        assert term.lower() not in html_lower


def test_generate_pdf_with_fake_playwright(monkeypatch):
    monkeypatch.setattr(pdf_generator, "sync_playwright", lambda: FakePlaywright())
    pdf = pdf_generator.generate_itinerary_pdf(build_pdf_context(sample_plan()))
    assert pdf.startswith(b"%PDF-PLAYWRIGHT")


def test_generate_pdf_missing_chromium_raises(monkeypatch):
    class BrokenChromium:
        def launch(self, headless=True):
            raise pdf_generator.PlaywrightError("Executable doesn't exist at chromium.exe")

    class BrokenPlaywright(FakePlaywright):
        def __init__(self):
            self.chromium = BrokenChromium()

    monkeypatch.setattr(pdf_generator, "sync_playwright", lambda: BrokenPlaywright())
    with pytest.raises(pdf_generator.PdfGenerationError) as exc:
        pdf_generator.generate_itinerary_pdf(build_pdf_context(sample_plan()))
    assert "playwright install chromium" in str(exc.value)


def test_generate_pdf_rejects_tiny_output(monkeypatch):
    class TinyPage(FakePage):
        def pdf(self, **kwargs):
            return b"%PDF"

    class TinyBrowser(FakeBrowser):
        def new_page(self, viewport=None):
            return TinyPage()

    class TinyChromium(FakeChromium):
        def launch(self, headless=True):
            return TinyBrowser()

    class TinyPlaywright(FakePlaywright):
        def __init__(self):
            self.chromium = TinyChromium()

    monkeypatch.setattr(pdf_generator, "sync_playwright", lambda: TinyPlaywright())
    with pytest.raises(pdf_generator.PdfGenerationError):
        pdf_generator.generate_itinerary_pdf(build_pdf_context(sample_plan()))


def test_image_to_data_uri(tmp_path):
    image = tmp_path / "logo.png"
    image.write_bytes(b"png-data")
    data_uri = pdf_generator.image_to_data_uri(image)
    assert data_uri == "data:image/png;base64,cG5nLWRhdGE="


def test_export_validation_blocks_wrong_country():
    context = build_pdf_context(sample_plan())
    context["days"][0]["activities"][0]["address"] = "Lahore, Pakistan"
    issues = validate_export_context(context)
    assert any("Pakistan" in issue for issue in issues)


def test_export_service_blocks_invalid_context(monkeypatch):
    plan = sample_plan()
    plan["itinerary"]["days"][0]["activities"][0]["description"] = (
        "Address: Lahore, Pakistan | Google Maps: https://maps.google.com/?cid=123"
    )
    monkeypatch.setattr(export_service, "generate_itinerary_pdf", lambda context: b"%PDF-PLAYWRIGHT")
    with pytest.raises(export_service.ExportValidationError):
        export_service.create_trip_export(plan)


def test_filename_and_hash(monkeypatch):
    monkeypatch.setattr(export_service, "generate_itinerary_pdf", lambda context: b"%PDF-PLAYWRIGHT")
    export = export_service.create_trip_export(sample_plan())
    assert export.filename == "reiseplan_madrid.pdf"
    assert export.pdf_bytes == b"%PDF-PLAYWRIGHT"
    assert export.plan_hash == export_service.calculate_plan_hash(sample_plan())


def test_plan_hash_changes_when_plan_changes():
    first = sample_plan()
    second = sample_plan()
    second["itinerary"]["total_cost"] = 120
    assert export_service.calculate_plan_hash(first) != export_service.calculate_plan_hash(second)


def test_discord_missing_webhook(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    with pytest.raises(DiscordDeliveryError):
        discord_client.send_itinerary_to_discord(plan=sample_plan(), pdf_bytes=b"%PDF", filename="plan.pdf")


def test_discord_success(monkeypatch):
    calls = {}

    class Response:
        status_code = 204
        text = ""

    def fake_post(url, data, files, timeout):
        calls["url"] = url
        calls["data"] = data
        calls["files"] = files
        calls["timeout"] = timeout
        return Response()

    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/test")
    monkeypatch.setattr(discord_client.requests, "post", fake_post)
    discord_client.send_itinerary_to_discord(plan=sample_plan(), pdf_bytes=b"%PDF", filename="plan.pdf")
    assert "Madrid" in calls["data"]["content"]
    assert calls["files"]["files[0]"][0] == "plan.pdf"


def test_discord_http_error(monkeypatch):
    class Response:
        status_code = 404
        text = "unknown webhook"

    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/secret")
    monkeypatch.setattr(discord_client.requests, "post", lambda *args, **kwargs: Response())
    with pytest.raises(DiscordDeliveryError) as exc:
        discord_client.send_itinerary_to_discord(plan=sample_plan(), pdf_bytes=b"%PDF", filename="plan.pdf")
    assert "secret" not in str(exc.value)


def test_discord_timeout(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.Timeout()

    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/test")
    monkeypatch.setattr(discord_client.requests, "post", fake_post)
    with pytest.raises(DiscordDeliveryError):
        discord_client.send_itinerary_to_discord(plan=sample_plan(), pdf_bytes=b"%PDF", filename="plan.pdf")
