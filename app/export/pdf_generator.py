from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright


BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
PDF_CSS_PATH = STATIC_DIR / "pdf.css"


class PdfGenerationError(RuntimeError):
    pass


def build_html(context: dict[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("itinerary.html")
    render_context = dict(context)
    render_context["pdf_css"] = PDF_CSS_PATH.read_text(encoding="utf-8") if PDF_CSS_PATH.exists() else ""
    return template.render(**render_context)


def image_to_data_uri(path: Path) -> str:
    mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def generate_itinerary_pdf(context: dict[str, Any]) -> bytes:
    html_content = build_html(context)

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1240, "height": 1754})
                page.set_content(html_content, wait_until="networkidle")
                page.emulate_media(media="print")
                pdf_bytes = page.pdf(
                    format="A4",
                    print_background=True,
                    prefer_css_page_size=True,
                    margin={
                        "top": "0mm",
                        "right": "0mm",
                        "bottom": "0mm",
                        "left": "0mm",
                    },
                )
                _validate_pdf_bytes(pdf_bytes)
                return pdf_bytes
            finally:
                browser.close()
    except PdfGenerationError:
        raise
    except Exception as exc:
        raise PdfGenerationError(_pdf_error_message(exc)) from exc


def _pdf_error_message(exc: Exception) -> str:
    message = str(exc).lower()
    if (
        isinstance(exc, PlaywrightError)
        and ("executable doesn't exist" in message or "browser" in message or "chromium" in message)
    ):
        return (
            "Chromium fuer den PDF-Export wurde nicht gefunden. "
            "Fuehre 'python -m playwright install chromium' aus."
        )
    return "Die PDF konnte mit Chromium nicht erzeugt werden. Pruefe Playwright und Chromium."


def _validate_pdf_bytes(pdf_bytes: bytes) -> None:
    if not pdf_bytes.startswith(b"%PDF"):
        raise PdfGenerationError("Der PDF-Export hat keine gültige PDF-Datei erzeugt.")
    if len(pdf_bytes) < 10_000:
        raise PdfGenerationError("Der PDF-Export ist ungewöhnlich klein und wurde vorsorglich verworfen.")
