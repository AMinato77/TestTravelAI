from __future__ import annotations

import html
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=True)

from app.agents.request_agent import parse_travel_request
from app.export.discord_client import DiscordDeliveryError
from app.export.export_service import (
    ExportValidationError,
    calculate_plan_hash,
    create_trip_export,
    deliver_trip_to_discord,
)
from app.export.pdf_generator import PdfGenerationError
from app.models.preference_source import PreferenceSource
from app.models.travel_request import TravelRequest
from app.orchestrator import (
    PreparedPlanContext,
    TravelPlanResult,
    expand_interactive_plan,
    finalize_interactive_plan,
    prepare_interactive_plan,
    revise_travel_plan,
)
from app.rag.memory_retrieval import delete_user_memory_sources
from app.rag.user_memory import create_user_profile, list_user_ids, load_user_profile
from app.services.serialization import itinerary_to_dict, validation_to_dict
from app.tools.gmail_tool import (
    GmailIntegrationError,
    build_gmail_preference_source,
    get_gmail_account_email,
    gmail_credentials_available,
    save_gmail_credentials_file,
)
from app.tools.openai_runtime import MissingLocalAIError, MissingOpenAIKeyError, TransientAIProviderError, ai_provider


APP_TITLE = "TravelAI"
APP_SUBTITLE = "Agentischer Reiseplaner mit konkreten Suchqueries, Memory und interaktiver Anpassung"

DEFAULT_USER_ID = "demo_user_1"
DEFAULT_REQUEST = (
    "Ich will 2 Tage nach Paris, typische franzoesische Kueche und Anime-Laeden, "
    "aber kein Sport und keine Touristenfallen."
)
PDF_EXPORT_CACHE_VERSION = "travel-copy-v3"

STYLE_CHOICES = [
    ("Ausgewogen", "balanced"),
    ("Entspannt", "relaxed"),
    ("Abenteuer", "adventure"),
    ("Luxus", "luxury"),
    ("Guenstig", "budget"),
]
BUDGET_CHOICES = [("Niedrig", "low"), ("Mittel", "medium"), ("Hoch", "high")]
DESTINATION_SCOPE_CHOICES = [("Stadt", "city"), ("Land", "country"), ("Region", "region"), ("Offen", "open")]
INTEREST_TAG_CHOICES = [
    ("Essen", "food"),
    ("Streetfood", "street food"),
    ("Anime", "anime"),
    ("Gaming", "gaming"),
    ("Kultur", "culture"),
    ("Geschichte", "history"),
    ("Lokale Orte", "local spots"),
    ("Natur", "nature"),
    ("Shopping", "shopping"),
    ("Sport", "sport"),
    ("Architektur", "architecture"),
    ("Fotografie", "photography"),
    ("Technik", "technology"),
    ("Nightlife", "nightlife"),
]
AVOID_TAG_CHOICES = [
    "Sport",
    "Restaurants",
    "Touristenfallen",
    "Museen",
    "Shopping",
    "Clubs",
    "Nightlife",
    "Volle Orte",
    "Stressiger Plan",
]


def _friendly_exception(exc: Exception) -> str:
    text = str(exc)
    lower = text.lower()
    if isinstance(exc, TransientAIProviderError) or ("cloudflare" in lower and "520" in lower):
        return (
            "OpenAI hatte gerade einen temporaeren Serverfehler (Cloudflare 520). "
            "Bitte in etwa einer Minute erneut versuchen. Es wurden keine Demo-Daten verwendet."
        )
    if "timeout" in lower or "timed out" in lower:
        return "Der KI-Aufruf hat zu lange gedauert. Bitte erneut versuchen; die bisherigen Profildaten bleiben erhalten."
    if isinstance(exc, ExportValidationError):
        return text
    if len(text) > 500:
        return text[:500] + "..."
    return text
MAIN_VIEWS = ["Briefing", "Kandidaten", "Reiseplan", "Memory", "Technik"]


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="AI", layout="wide", initial_sidebar_state="expanded")
    _apply_styles()
    _init_state()

    current_profile = load_user_profile(st.session_state.user_id)
    sidebar_state = _render_sidebar(current_profile)

    _render_header()
    result = st.session_state.get("last_result")
    parsed_request = st.session_state.get("last_parsed_request")

    briefing_tab, candidates_tab, plan_tab, memory_tab, tech_tab = st.tabs(MAIN_VIEWS)
    with briefing_tab:
        _render_ai_view(current_profile, sidebar_state, result)
    with candidates_tab:
        _render_candidates_view()
    with plan_tab:
        _render_plan_view(result, parsed_request)
    with memory_tab:
        _render_memory_view(current_profile)
    with tech_tab:
        _render_tech_view(result, parsed_request, sidebar_state, current_profile)


def _render_header() -> None:
    st.markdown(
        f"""
        <div class="app-header">
          <div>
            <div class="eyebrow">{APP_TITLE}</div>
            <h1>Reiseplaner</h1>
            <p>{APP_SUBTITLE}</p>
          </div>
          <div class="header-badges">
            {_pill(ai_provider().upper(), "accent")}
            {_pill(st.session_state.user_id, "muted")}
            {_pill("Query Workflow", "success")}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(profile) -> dict[str, Any]:
    st.sidebar.markdown(
        f"<div class='sidebar-title'>Herzlich Willkommen</div>"
        f"<div class='sidebar-user'>{html.escape(st.session_state.user_id)}</div>",
        unsafe_allow_html=True,
    )

    user_ids = list_user_ids()
    if not user_ids:
        create_user_profile(DEFAULT_USER_ID)
        user_ids = [DEFAULT_USER_ID]

    if st.session_state.user_id not in user_ids:
        st.session_state.user_id = user_ids[0]

    selected_user = st.sidebar.selectbox(
        "Gespeicherte Profile",
        sorted(user_ids),
        index=sorted(user_ids).index(st.session_state.user_id),
        key="profile_select",
    )
    if selected_user != st.session_state.user_id:
        st.session_state.user_id = selected_user
        st.rerun()

    with st.sidebar.expander("Neues Profil", expanded=False):
        new_user_id = st.text_input("User ID", placeholder="z. B. tokyo_user")
        if st.button("Profil anlegen", use_container_width=True):
            cleaned = _safe_user_id(new_user_id)
            if cleaned:
                create_user_profile(cleaned)
                st.session_state.user_id = cleaned
                st.rerun()
            st.warning("Bitte eine gueltige User ID eingeben.")

    st.sidebar.divider()
    st.sidebar.markdown("### Profil")
    _sidebar_list("Praeferenznotizen", getattr(profile, "preference_notes", []))
    _sidebar_list("Tags", getattr(profile, "interest_tags", []))
    _sidebar_list("Meiden", getattr(profile, "avoid", []))
    _sidebar_list("Ziele", getattr(profile, "past_destinations", []))

    st.sidebar.divider()
    st.sidebar.markdown("### Quellen")
    uploaded_files = st.sidebar.file_uploader(
        "Notizen / Chat-Exports",
        type=["txt", "md", "json", "csv"],
        accept_multiple_files=True,
        key="source_uploads",
    )
    travel_ratings = st.sidebar.text_area(
        "Reisebewertungen",
        placeholder="Barcelona: 9/10, Essen war super.\nParis: 5/10, zu touristisch.",
        height=96,
        key="travel_ratings",
    )

    st.sidebar.divider()
    st.sidebar.markdown("### Gmail")
    creds_ok = gmail_credentials_available()
    st.sidebar.caption(f"Credentials: {'ok' if creds_ok else 'fehlt'}")
    with st.sidebar.expander("OAuth", expanded=False):
        if not creds_ok:
            credentials = st.file_uploader("OAuth Client JSON", type=["json"], key="gmail_credentials_upload")
            if credentials and st.button("Credentials speichern", use_container_width=True):
                try:
                    save_gmail_credentials_file(credentials.getvalue())
                    st.success("Credentials gespeichert.")
                    st.rerun()
                except GmailIntegrationError as exc:
                    st.error(_friendly_exception(exc))

        gmail_limit = st.number_input("Max. Mails", min_value=1, max_value=50, value=20, step=1)
        gmail_days = st.number_input("Lookback Tage", min_value=1, max_value=3650, value=365, step=30)
        if st.button("Gmail laden", disabled=not creds_ok, use_container_width=True):
            try:
                account_email = get_gmail_account_email(st.session_state.user_id)
                sources, messages = build_gmail_preference_source(
                    user_id=st.session_state.user_id,
                    max_messages=int(gmail_limit),
                    lookback_days=int(gmail_days),
                )
                st.session_state.gmail_account_email = account_email
                st.session_state.gmail_sources = sources
                st.session_state.gmail_messages = messages
                st.success(f"{len(messages)} Mail(s), {len(sources)} Quelle(n)")
                st.rerun()
            except GmailIntegrationError as exc:
                st.error(_friendly_exception(exc))
        if st.button("Gmail-Memory loeschen", use_container_width=True):
            deleted = delete_user_memory_sources(st.session_state.user_id, source_type="email_newsletter")
            st.session_state.gmail_sources = []
            st.session_state.gmail_messages = []
            st.session_state.gmail_account_email = ""
            st.success(f"{deleted} Gmail-Memory-Chunk(s) geloescht.")
            st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("### Debug")
    if st.sidebar.button("Letztes Ergebnis loeschen", use_container_width=True):
        st.session_state.last_result = None
        st.session_state.last_parsed_request = None
        st.session_state.last_inputs = {}
        st.session_state.plan_versions = []
        st.session_state.prepared_context = None
        st.session_state.pending_conflict_result = None
        st.session_state.pending_conflict_decisions = {}
        st.session_state.planning_stage = "INPUT"
        st.rerun()

    return {
        "uploaded_files": uploaded_files or [],
        "travel_ratings": travel_ratings,
        "gmail_sources": st.session_state.get("gmail_sources", []),
        "gmail_messages": st.session_state.get("gmail_messages", []),
        "gmail_account_email": st.session_state.get("gmail_account_email", ""),
    }


def _render_ai_view(profile, sidebar_state: dict[str, Any], result: TravelPlanResult | None) -> None:
    st.markdown("### Reisebriefing")
    with st.form("travel_form", border=False, clear_on_submit=False):
        request_text = st.text_area(
            "Beschreibe deine Reise konkret",
            value=st.session_state.last_inputs.get("request_text", DEFAULT_REQUEST),
            placeholder=DEFAULT_REQUEST,
            height=130,
        )

        col_a, col_b, col_c = st.columns([1.5, 0.8, 0.9])
        with col_a:
            destination = st.text_input("Reiseziel", value=st.session_state.last_inputs.get("destination", ""))
        with col_b:
            days = st.number_input("Tage", min_value=1, max_value=14, value=int(st.session_state.last_inputs.get("days", 3)))
        with col_c:
            budget = st.number_input(
                "Budget",
                min_value=0.0,
                max_value=100_000.0,
                value=float(st.session_state.last_inputs.get("budget", 600.0)),
                step=50.0,
            )

        col_d, col_e, col_f = st.columns(3)
        with col_d:
            style_label = st.selectbox("Reisestil", [label for label, _ in STYLE_CHOICES])
        with col_e:
            budget_label = st.selectbox("Budgetpraeferenz", [label for label, _ in BUDGET_CHOICES], index=1)
        with col_f:
            scope_label = st.selectbox("Zielart", [label for label, _ in DESTINATION_SCOPE_CHOICES])

        col_must, col_avoid = st.columns(2)
        with col_must:
            must_have_text = st.text_area(
                "Muss enthalten sein",
                value=st.session_state.last_inputs.get("must_have_text", ""),
                placeholder="z. B. One Piece Shops, Formel-1-Orte, typische lokale Kueche",
                height=95,
            )
        with col_avoid:
            avoid_text = st.text_area(
                "Vermeiden",
                value=st.session_state.last_inputs.get("avoid_text", ""),
                placeholder="z. B. Sport, Touristenfallen, Clubs, lange Warteschlangen",
                height=95,
            )

        tag_labels = st.multiselect(
            "Optionale Interessen-Tags",
            [label for label, _ in INTEREST_TAG_CHOICES],
            default=_labels_for_values(st.session_state.last_inputs.get("interest_tags", []), INTEREST_TAG_CHOICES),
        )
        avoid_labels = st.multiselect("Optionale Avoid-Tags", AVOID_TAG_CHOICES)
        recommend_destination = st.toggle("Reiseziel empfehlen lassen", value=False)

        submitted = st.form_submit_button("Reise vorbereiten", type="primary", use_container_width=True)

    if submitted:
        _run_prepare_plan(
            profile=profile,
            sidebar_state=sidebar_state,
            request_text=request_text,
            destination=destination,
            days=int(days),
            budget=float(budget),
            travel_style=_value_for_label(style_label, STYLE_CHOICES),
            budget_preference=_value_for_label(budget_label, BUDGET_CHOICES),
            destination_scope=_value_for_label(scope_label, DESTINATION_SCOPE_CHOICES),
            must_have_text=must_have_text,
            avoid_text=avoid_text,
            interest_tags=_values_for_labels(tag_labels, INTEREST_TAG_CHOICES),
            avoid_tags=avoid_labels,
            recommend_destination=recommend_destination,
        )

    prepared = st.session_state.get("prepared_context")
    st.markdown("### Aktueller Stand")
    if st.session_state.get("planning_stage") == "CONFLICT_REVIEW" and st.session_state.get("pending_conflict_result"):
        st.markdown(
            _info_panel(
                "Rueckfrage offen",
                "Der vorlaeufige Plan braucht eine Budgetentscheidung. Oeffne den Kandidaten-Tab, um festzulegen, wie TravelAI weiter planen soll.",
            ),
            unsafe_allow_html=True,
        )
    elif prepared:
        _render_prepared_status(prepared)
    elif result:
        _render_ai_summary(result)
    else:
        _render_empty_state(profile, sidebar_state)


def _render_candidates_view() -> None:
    prepared = st.session_state.get("prepared_context")
    if st.session_state.get("planning_stage") == "CONFLICT_REVIEW" and st.session_state.get("pending_conflict_result"):
        _render_budget_conflict_review()
        return
    if not prepared:
        st.markdown("### Kandidaten")
        st.markdown(
            _info_panel(
                "Noch keine Kandidaten",
                "Bereite zuerst im Briefing-Tab eine Reise vor. Danach erscheinen hier echte Google-Places-Kandidaten und Rueckfragen.",
            ),
            unsafe_allow_html=True,
        )
        return
    _render_interactive_planning(prepared)


def _render_memory_view(profile) -> None:
    st.markdown("### Memory & Profil")
    prepared = st.session_state.get("prepared_context")
    result = st.session_state.get("last_result")
    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric("Profil", getattr(profile, "user_id", st.session_state.user_id))
    col_2.metric("Tags", len(getattr(profile, "interest_tags", []) or []))
    col_3.metric("Praeferenzen", len(getattr(profile, "preference_notes", []) or []))
    col_4.metric("Vergangene Ziele", len(getattr(profile, "past_destinations", []) or []))

    st.markdown("#### Profilzusammenfassung")
    profile_bits = []
    if getattr(profile, "interest_tags", None):
        profile_bits.append("Interessen: " + ", ".join(str(item) for item in profile.interest_tags[:10]))
    if getattr(profile, "preference_notes", None):
        profile_bits.append("Praeferenzen: " + " | ".join(str(item) for item in profile.preference_notes[:4]))
    if getattr(profile, "avoid", None):
        profile_bits.append("Meiden: " + ", ".join(str(item) for item in profile.avoid[:8]))
    if getattr(profile, "past_destinations", None):
        profile_bits.append("Bisherige Ziele: " + ", ".join(str(item) for item in profile.past_destinations[:8]))
    st.markdown(_bullets_panel("Was TravelAI ueber dieses Profil weiss", profile_bits or ["Noch keine belastbare Profil-Memory."]), unsafe_allow_html=True)

    if prepared:
        _render_memory_influence(prepared)
    elif result:
        memory = ((result.query_planning or {}).get("memory_usage") or [])
        ignored = ((result.query_planning or {}).get("ignored_memories") or [])
        if memory or ignored:
            st.markdown("#### Memory-Nutzung im letzten Plan")
            for item in memory[:6]:
                st.markdown(_memory_item_card(item.get("memory", ""), item.get("effect", ""), "success"), unsafe_allow_html=True)
            for item in ignored[:4]:
                st.markdown(_memory_item_card(item.get("memory", ""), item.get("reason", ""), "muted"), unsafe_allow_html=True)
        else:
            st.caption("Im letzten fertigen Plan wurde keine explizite Memory-Nutzung gemeldet.")
    else:
        st.caption("Sobald eine Reise vorbereitet ist, erscheint hier, wie Chroma/Memory die Planung beeinflusst.")


def _render_prepared_status(prepared: PreparedPlanContext) -> None:
    request = prepared.request
    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric("Ziel", request.destination or "-")
    col_2.metric("Tage", request.duration_days)
    col_3.metric("Budget", _format_currency(request.budget))
    col_4.metric("Kandidaten", len(prepared.activities))
    wishes = request.must_have or ["Plan wird aus Profil und Anfrage abgeleitet"]
    st.markdown(
        _bullets_panel(
            "Naechster Schritt",
            [
                "Recherche ist abgeschlossen.",
                "Oeffne den Kandidaten-Tab, markiere passende Orte und erstelle danach den finalen Plan.",
                "Erkannte Planungsbasis: " + ", ".join(str(item) for item in wishes[:5]),
            ],
        ),
        unsafe_allow_html=True,
    )


def _run_prepare_plan(
    profile,
    sidebar_state: dict[str, Any],
    request_text: str,
    destination: str,
    days: int,
    budget: float,
    travel_style: str,
    budget_preference: str,
    destination_scope: str,
    must_have_text: str,
    avoid_text: str,
    interest_tags: list[str],
    avoid_tags: list[str],
    recommend_destination: bool,
    ) -> None:
    fallback = TravelRequest(
        destination=destination,
        destination_scope=destination_scope,
        needs_destination_recommendation=recommend_destination,
        duration_days=days,
        budget=budget,
        must_have=_parse_list(must_have_text),
        avoid=[*_parse_list(avoid_text), *avoid_tags],
        interest_tags=interest_tags,
        query_hints=[],
        travel_style=travel_style,
    )
    briefing_parts = [request_text]
    if must_have_text.strip():
        briefing_parts.append(f"Must-have: {must_have_text}")
    if avoid_text.strip() or avoid_tags:
        avoid_briefing = ", ".join([avoid_text.strip(), *avoid_tags]).strip(" ,")
        if avoid_briefing:
            briefing_parts.append(f"Vermeiden: {avoid_briefing}")
    briefing = "\n".join(part for part in briefing_parts if part.strip())
    try:
        parsed = parse_travel_request(briefing, fallback)
        effective_must_have = _merge_unique(_parse_list(must_have_text), parsed.must_have)
        effective_avoid = _merge_unique(_parse_list(avoid_text), avoid_tags, parsed.avoid)
        effective_tags = _merge_unique(interest_tags, parsed.interest_tags)
        effective_query_hints = _merge_unique(parsed.query_hints, [f"{parsed.destination} {item}" for item in effective_must_have if parsed.destination])
        preference_sources = _build_preference_sources(
            sidebar_state.get("uploaded_files") or [],
            sidebar_state.get("travel_ratings") or "",
            "",
        )
        preference_sources.extend(sidebar_state.get("gmail_sources") or [])

        with st.spinner("TravelAI recherchiert echte Orte und bereitet Rueckfragen vor..."):
            prepared = prepare_interactive_plan(
                user_id=st.session_state.user_id,
                destination=parsed.destination,
                days=parsed.duration_days,
                budget=parsed.budget,
                travel_style=parsed.travel_style,
                budget_preference=budget_preference,
                feedback=None,
                preference_sources=preference_sources,
                manual_avoid=effective_avoid,
                destination_scope=parsed.destination_scope,
                needs_destination_recommendation=parsed.needs_destination_recommendation,
                must_have=effective_must_have,
                interest_tags=effective_tags,
                query_hints=effective_query_hints,
                use_profile_memory=parsed.use_profile_memory,
            )
    except (MissingOpenAIKeyError, MissingLocalAIError) as exc:
        st.error(_friendly_exception(exc))
        return
    except Exception as exc:
        st.error(f"Planung fehlgeschlagen: {_friendly_exception(exc)}")
        return

    st.session_state.prepared_context = prepared
    st.session_state.planning_stage = "PREVIEW"
    st.session_state.last_result = None
    st.session_state.last_parsed_request = parsed
    st.session_state.last_inputs = {
        "request_text": request_text,
        "destination": parsed.destination,
        "days": parsed.duration_days,
        "budget": parsed.budget,
        "travel_style": parsed.travel_style,
        "budget_preference": budget_preference,
        "destination_scope": parsed.destination_scope,
        "must_have_text": must_have_text,
        "avoid_text": avoid_text,
        "must_have": effective_must_have,
        "avoid": effective_avoid,
        "interest_tags": effective_tags,
        "query_hints": effective_query_hints,
        "use_profile_memory": parsed.use_profile_memory,
    }
    st.session_state.plan_versions = []
    st.session_state.pending_conflict_result = None
    st.session_state.pending_conflict_decisions = {}
    st.rerun()


def _render_interactive_planning(prepared: PreparedPlanContext) -> None:
    request = prepared.request
    st.markdown("### Kandidaten auswählen")
    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric("Ziel", request.destination or "-")
    col_2.metric("Tage", request.duration_days)
    col_3.metric("Budget", _format_currency(request.budget))
    col_4.metric("Kandidaten", len(prepared.activities))

    st.markdown(
        _info_panel(
            "Recherche abgeschlossen",
            "TravelAI hat Anfrage, Memory, Wetter und echte Google-Places-Daten vorbereitet. Markiere jetzt, was in den Plan soll.",
        ),
        unsafe_allow_html=True,
    )

    if request.must_have:
        st.markdown("**Erkannte Wuensche**")
        st.markdown(_render_tags(request.must_have, "accent"), unsafe_allow_html=True)
    if prepared.weather:
        st.caption(str(prepared.weather.get("summary") or "Wetterdaten geladen."))
    if getattr(request, "use_profile_memory", False) or (prepared.query_planning or {}).get("memory_usage"):
        st.caption("Memory wurde beruecksichtigt. Details findest du im Memory-Tab.")

    st.markdown("#### Rueckfragen")
    answers: dict[str, str] = {}
    if prepared.questions:
        for question in prepared.questions:
            options = question.get("options") or []
            labels = [str(option.get("label")) for option in options]
            default_index = 0
            selected = st.radio(
                str(question.get("question") or question.get("title") or "Auswahl"),
                labels,
                index=default_index,
                horizontal=True,
                key=f"interactive_question_{question.get('id')}",
            )
            selected_option = next((option for option in options if option.get("label") == selected), options[0] if options else {})
            if selected_option.get("note"):
                st.caption(str(selected_option.get("note")))
            if selected_option.get("value"):
                answers[str(question.get("id"))] = str(selected_option.get("value"))
            if question.get("context"):
                st.caption("Betrifft: " + ", ".join(str(item) for item in question.get("context") or []))
    else:
        st.caption("Keine zwingende Rueckfrage erkannt. Du kannst trotzdem Kandidaten markieren.")

    st.markdown("#### Orte und Erlebnisse")
    candidate_actions: dict[str, str] = {}
    for activity in prepared.activities[:12]:
        candidate_actions[activity.name] = _render_candidate_card(activity)
    if len(prepared.activities) > 12:
        st.caption(f"{len(prepared.activities) - 12} weitere Kandidaten bleiben im Pool, werden hier aber nicht alle angezeigt.")

    with st.container(border=True):
        st.markdown("**Fehlt noch etwas?**")
        expansion_text = st.text_input(
            "Schreibe, was TravelAI noch suchen soll",
            placeholder="z. B. Mir fehlt noch mehr Architektur, bitte 2-3 passende Orte suchen.",
            key="interactive_expansion_text",
        )
        if st.button("Weitere Kandidaten suchen", disabled=not expansion_text.strip(), use_container_width=True):
            decisions = _collect_interactive_decisions(candidate_actions, answers)
            try:
                removed_before_search = len(decisions.get("exclude_names", [])) + len(decisions.get("already_visited_names", []))
                with st.spinner("TravelAI sucht gezielt weitere echte Places-Kandidaten..."):
                    expanded = expand_interactive_plan(prepared, expansion_text, decisions)
                st.session_state.prepared_context = expanded
                added = max(0, len(expanded.activities) - len(prepared.activities) + removed_before_search)
                if added > 0:
                    st.success(f"{added} neue Kandidat(en) hinzugefuegt.")
                elif removed_before_search > 0:
                    st.success(f"{removed_before_search} markierte Kandidat(en) entfernt.")
                else:
                    st.warning("Keine neuen Kandidaten gefunden. Du kannst die Suchbeschreibung konkreter formulieren.")
                st.rerun()
            except Exception as exc:
                st.error(f"Nachsuche fehlgeschlagen: {_friendly_exception(exc)}")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Finalen Plan mit meinen Entscheidungen erstellen", type="primary", use_container_width=True):
            decisions = _collect_interactive_decisions(candidate_actions, answers)
            try:
                with st.spinner("Planning Agent erstellt finalen Plan, Validation prueft danach..."):
                    result = finalize_interactive_plan(prepared, decisions)
                _store_or_review_final_result(result, decisions)
                st.rerun()
            except Exception as exc:
                st.error(f"Finalisierung fehlgeschlagen: {_friendly_exception(exc)}")
    with col_b:
        if st.button("Neue Recherche starten", use_container_width=True):
            st.session_state.prepared_context = None
            st.session_state.planning_stage = "INPUT"
            st.session_state.pending_conflict_result = None
            st.session_state.pending_conflict_decisions = {}
            st.rerun()


def _render_candidate_card(activity) -> str:
    with st.container(border=True):
        meta = []
        if activity.cost:
            meta.append(_format_currency(activity.cost))
        if activity.duration_hours:
            meta.append(f"{activity.duration_hours:g} h")
        st.markdown(
            f"""
            <div class="candidate-card-head">
              <div>
                <div class="candidate-title">{html.escape(activity.name)}</div>
                <div class="candidate-meta">{_pill(_category_label(activity.category), "accent")}</div>
              </div>
              <div class="candidate-price">{html.escape(" · ".join(meta) if meta else "flexibel")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        details = _place_details(activity.description or "")
        if details:
            st.markdown(_render_tags(details[:3], "muted"), unsafe_allow_html=True)
        if activity.description:
            with st.expander("Details", expanded=False):
                st.caption(_compact_description(activity.description, 420))
        return st.radio(
            "Entscheidung",
            ["Neutral", "Unbedingt einplanen", "Kenne ich schon", "Mehr davon", "Nicht mein Stil"],
            horizontal=True,
            key=f"candidate_action_{_safe_widget_key(activity.name)}",
            label_visibility="collapsed",
        )


def _render_memory_influence(prepared: PreparedPlanContext) -> None:
    request = prepared.request
    query_planning = prepared.query_planning or {}
    memory_usage = query_planning.get("memory_usage") or []
    ignored_memories = query_planning.get("ignored_memories") or []
    memory_context = prepared.memory_context or []

    with st.expander("Warum dieser Plan zu dir passt", expanded=bool(getattr(request, "use_profile_memory", False) or memory_usage)):
        col_1, col_2 = st.columns(2)
        with col_1:
            st.markdown("**Aus deiner Anfrage**")
            if request.must_have:
                st.markdown(_render_tags(request.must_have, "accent"), unsafe_allow_html=True)
            else:
                st.caption("Keine expliziten Muss-Wuensche erkannt; die Planung wird aus Anfrage und Profil abgeleitet.")
        with col_2:
            st.markdown("**Aus deinem Profil / Chroma**")
            if memory_context:
                for source in memory_context[:4]:
                    preview = " ".join(str(source.text).split())[:180]
                    st.caption(f"{source.name}: {preview}")
            else:
                st.caption("Keine passende Memory gefunden.")

        if memory_usage:
            st.markdown("**Genutzte Erinnerungen**")
            for item in memory_usage[:5]:
                memory = str(item.get("memory") or "").strip()
                effect = str(item.get("effect") or "").strip()
                confidence = item.get("confidence")
                suffix = f" (Confidence: {confidence})" if confidence not in (None, "") else ""
                st.markdown(_memory_item_card(memory, f"{effect}{suffix}", "success"), unsafe_allow_html=True)
        elif getattr(request, "use_profile_memory", False):
            st.caption("Profil-Memory wurde angefragt; der Query Planner hat keine explizite Zusatzentscheidung gemeldet.")

        if ignored_memories:
            st.markdown("**Bewusst nicht genutzt**")
            for item in ignored_memories[:5]:
                memory = str(item.get("memory") or "").strip()
                reason = str(item.get("reason") or "").strip()
                st.markdown(_memory_item_card(memory, reason, "muted"), unsafe_allow_html=True)


def _collect_interactive_decisions(candidate_actions: dict[str, str], answers: dict[str, str]) -> dict:
    decisions = {
        "answers": answers,
        "include_names": [],
        "exclude_names": [],
        "already_visited_names": [],
        "more_like_names": [],
    }
    for name, action in candidate_actions.items():
        if action == "Unbedingt einplanen":
            decisions["include_names"].append(name)
        elif action == "Kenne ich schon":
            decisions["already_visited_names"].append(name)
        elif action == "Mehr davon":
            decisions["more_like_names"].append(name)
        elif action == "Nicht mein Stil":
            decisions["exclude_names"].append(name)
    return decisions


def _render_budget_conflict_review() -> None:
    result: TravelPlanResult | None = st.session_state.get("pending_conflict_result")
    prepared: PreparedPlanContext | None = st.session_state.get("prepared_context")
    decisions = st.session_state.get("pending_conflict_decisions") or {}
    if not result or not prepared:
        st.session_state.planning_stage = "PREVIEW" if prepared else "INPUT"
        st.session_state.pending_conflict_result = None
        st.session_state.pending_conflict_decisions = {}
        st.rerun()
        return

    budget = float(prepared.request.budget or 0)
    total = float(result.itinerary.total_cost or 0)
    over = max(0.0, total - budget)
    included = _decision_name_set(decisions, "include_names")
    included_cost = sum(activity.cost for _day, _index, activity in _planned_activity_rows(result) if _activity_key(activity.name) in included)

    st.markdown("### Budget-Konflikt")
    st.markdown(
        _info_panel(
            "Rueckfrage vor dem finalen Plan",
            "Deine fest ausgewaehlten Aktivitaeten werden respektiert. Dadurch kann der vorlaeufige Plan aber Budget oder Tagesumfang verletzen. Entscheide jetzt, wie TravelAI weiter planen soll.",
        ),
        unsafe_allow_html=True,
    )
    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric("Budget", _format_currency(budget, result.itinerary.currency))
    col_2.metric("Vorlaeufig geplant", _format_currency(total, result.itinerary.currency))
    col_3.metric("Ueberschreitung", _format_currency(over, result.itinerary.currency))
    col_4.metric("Fest markiert", _format_currency(included_cost, result.itinerary.currency))

    st.warning(
        "Der Plan ueberschreitet das Budget, weil mehrere Aktivitaeten als 'Unbedingt einplanen' markiert sind. "
        "Diese bleiben geschuetzt, bis du selbst etwas anderes entscheidest."
    )

    rows = _planned_activity_rows(result)
    with st.container(border=True):
        st.markdown("**Aktivitaeten im vorlaeufigen Plan**")
        for day, index, activity in rows:
            key = _activity_key(activity.name)
            protected = key in included
            cols = st.columns([0.12, 1.2, 0.45, 0.45])
            keep = cols[0].checkbox(
                "behalten",
                value=True,
                key=f"budget_keep_{day}_{index}_{_safe_widget_key(activity.name)}",
                label_visibility="collapsed",
            )
            title = f"**{html.escape(activity.name)}**"
            if protected:
                title += " " + _pill("Unbedingt", "accent")
            cols[1].markdown(title, unsafe_allow_html=True)
            cols[1].caption(_category_label(activity.category))
            cols[2].caption(_format_currency(activity.cost, result.itinerary.currency))
            cols[3].caption(f"Tag {day}")
            st.session_state[f"budget_keep_value_{day}_{index}_{_safe_widget_key(activity.name)}"] = keep

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Trotzdem uebernehmen", type="primary", use_container_width=True):
            _complete_final_plan(result, label="Budget bewusst ueberschritten")
            st.rerun()

    with col_b:
        disabled = included_cost > budget > 0
        if st.button("Budget automatisch einhalten", disabled=disabled, use_container_width=True):
            reduced_decisions = _decisions_for_auto_budget_reduction(result, decisions, budget)
            try:
                with st.spinner("TravelAI reduziert nur nicht-priorisierte Aktivitaeten und validiert neu..."):
                    reduced = finalize_interactive_plan(prepared, reduced_decisions)
                _store_or_review_final_result(reduced, reduced_decisions)
                st.rerun()
            except Exception as exc:
                st.error(f"Budget-Reduktion fehlgeschlagen: {_friendly_exception(exc)}")
        if disabled:
            st.caption("Automatisch nicht moeglich: Die fest markierten Aktivitaeten liegen bereits ueber dem Budget.")

    with col_c:
        if st.button("Meine Auswahl neu planen", use_container_width=True):
            revised = _decisions_from_budget_selection(result, decisions)
            try:
                with st.spinner("TravelAI plant mit deiner Budget-Auswahl neu..."):
                    replanned = finalize_interactive_plan(prepared, revised)
                _store_or_review_final_result(replanned, revised)
                st.rerun()
            except Exception as exc:
                st.error(f"Neuplanung fehlgeschlagen: {_friendly_exception(exc)}")

    if st.button("Zurueck zur Kandidaten-Auswahl", use_container_width=True):
        st.session_state.planning_stage = "PREVIEW"
        st.session_state.pending_conflict_result = None
        st.session_state.pending_conflict_decisions = {}
        st.rerun()


def _store_or_review_final_result(result: TravelPlanResult, decisions: dict) -> None:
    if _has_budget_exceeded(result):
        st.session_state.pending_conflict_result = result
        st.session_state.pending_conflict_decisions = decisions
        st.session_state.planning_stage = "CONFLICT_REVIEW"
        st.session_state.pending_main_view = "KI"
        return
    _complete_final_plan(result, label="Interaktiver Erstplan")


def _complete_final_plan(result: TravelPlanResult, label: str = "Interaktiver Erstplan") -> None:
    st.session_state.last_result = result
    st.session_state.plan_versions = [{"version": 1, "label": label, "feedback": ""}]
    st.session_state.prepared_context = None
    st.session_state.pending_conflict_result = None
    st.session_state.pending_conflict_decisions = {}
    _clear_export_cache()
    st.session_state.planning_stage = "COMPLETED"
    st.session_state.pending_main_view = "Reiseplan"


def _has_budget_exceeded(result: TravelPlanResult) -> bool:
    return any(issue.issue_type == "budget_exceeded" for issue in result.validation.issues)


def _planned_activity_rows(result: TravelPlanResult) -> list[tuple[int, int, Any]]:
    rows: list[tuple[int, int, Any]] = []
    for day in result.itinerary.days:
        for index, activity in enumerate(day.activities):
            rows.append((day.day, index, activity))
    return rows


def _decisions_for_auto_budget_reduction(result: TravelPlanResult, decisions: dict, budget: float) -> dict:
    revised = _copy_decisions(decisions)
    included = _decision_name_set(revised, "include_names")
    total = float(result.itinerary.total_cost or 0)
    removable = [
        activity
        for _day, _index, activity in _planned_activity_rows(result)
        if _activity_key(activity.name) not in included
    ]
    removable.sort(key=lambda activity: (activity.cost, activity.duration_hours), reverse=True)
    for activity in removable:
        if total <= budget:
            break
        revised["exclude_names"] = _merge_unique(revised.get("exclude_names") or [], [activity.name])
        total -= float(activity.cost or 0)
    return _remove_excluded_from_positive_decisions(revised)


def _decisions_from_budget_selection(result: TravelPlanResult, decisions: dict) -> dict:
    revised = _copy_decisions(decisions)
    removed: list[str] = []
    for day, index, activity in _planned_activity_rows(result):
        keep = st.session_state.get(f"budget_keep_value_{day}_{index}_{_safe_widget_key(activity.name)}", True)
        if not keep:
            removed.append(activity.name)
    if removed:
        revised["exclude_names"] = _merge_unique(revised.get("exclude_names") or [], removed)
    return _remove_excluded_from_positive_decisions(revised)


def _remove_excluded_from_positive_decisions(decisions: dict) -> dict:
    blocked = _decision_name_set(decisions, "exclude_names") | _decision_name_set(decisions, "already_visited_names")
    for key in ("include_names", "more_like_names"):
        decisions[key] = [name for name in decisions.get(key, []) if _activity_key(name) not in blocked]
    return decisions


def _copy_decisions(decisions: dict) -> dict:
    return {
        "answers": dict(decisions.get("answers") or {}) if isinstance(decisions, dict) else {},
        "include_names": list((decisions or {}).get("include_names") or []),
        "exclude_names": list((decisions or {}).get("exclude_names") or []),
        "already_visited_names": list((decisions or {}).get("already_visited_names") or []),
        "more_like_names": list((decisions or {}).get("more_like_names") or []),
    }


def _decision_name_set(decisions: dict, key: str) -> set[str]:
    return {_activity_key(name) for name in (decisions or {}).get(key, []) if str(name).strip()}


def _activity_key(name: str) -> str:
    return " ".join(str(name or "").strip().lower().split())


def _render_empty_state(profile, sidebar_state: dict[str, Any]) -> None:
    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric("Profil", getattr(profile, "user_id", st.session_state.user_id))
    col_2.metric("Memory-Tags", len(getattr(profile, "interest_tags", [])))
    col_3.metric("Quellen", len(sidebar_state.get("uploaded_files") or []) + len(sidebar_state.get("gmail_sources") or []))
    col_4.metric("Provider", ai_provider())
    st.markdown(
        _info_panel(
            "Noch kein Plan erzeugt",
            "Beschreibe konkret, was du erleben willst. TravelAI erzeugt daraus Must-haves, Avoids und echte Google-Places-Suchqueries.",
        ),
        unsafe_allow_html=True,
    )


def _render_ai_summary(result: TravelPlanResult) -> None:
    itinerary = result.itinerary
    points = [
        f"Ziel: {itinerary.destination}",
        f"Tage: {len(itinerary.days)}",
        f"Aktivitaeten: {sum(len(day.activities) for day in itinerary.days)}",
        "Validierung ok" if result.validation.ok else f"{len(result.validation.issues)} Validierungshinweis(e)",
    ]
    if result.place_queries:
        points.append(f"{len(result.place_queries)} konkrete Places-Queries")
    st.markdown(_bullets_panel("Kurzfassung", points), unsafe_allow_html=True)


def _render_plan_view(result: TravelPlanResult | None, parsed_request: TravelRequest | None) -> None:
    st.markdown("### Finaler Reiseplan")
    if not result or not parsed_request:
        st.markdown(_info_panel("Noch kein Reiseplan vorhanden.", "Erstelle zuerst im KI-Tab einen Plan."), unsafe_allow_html=True)
        return

    summary = (result.explanation or {}).get("summary") or f"Plan fuer {result.itinerary.destination}."
    st.markdown(_info_panel(f"Reise nach {result.itinerary.destination}", summary), unsafe_allow_html=True)
    total_activities = sum(len(day.activities) for day in result.itinerary.days)
    total_hours = sum(float(day.total_duration_hours or 0) for day in result.itinerary.days)
    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric("Tage", len(result.itinerary.days))
    col_2.metric("Aktivitaeten", total_activities)
    col_3.metric("Dauer", f"{total_hours:g} h")
    col_4.metric("Budget", _format_currency(result.itinerary.total_cost, result.itinerary.currency))
    _render_wish_coverage(result.validation)
    _render_itinerary(result)
    _render_export_panel(result)
    _render_revision_panel(result, parsed_request)


def _render_wish_coverage(validation) -> None:
    semantic = _semantic_summary(validation)
    st.markdown("#### Abdeckung deiner Wuensche")
    col_1, col_2, col_3 = st.columns(3)
    col_1.metric("Status", "ok" if semantic["ok"] else "pruefen")
    col_2.metric("Offene Wuensche", len(semantic["missing"]))
    col_3.metric("Avoid-Konflikte", len(semantic["avoid"]))
    if semantic["missing"]:
        st.warning("Offen: " + " | ".join(semantic["missing"]))
    if semantic["avoid"]:
        st.error("Konflikte: " + " | ".join(semantic["avoid"]))


def _render_itinerary(result: TravelPlanResult) -> None:
    for day in result.itinerary.days:
        with st.expander(
            f"Tag {day.day} · {_format_currency(day.total_cost, result.itinerary.currency)} · {day.total_duration_hours:g} h",
            expanded=True,
        ):
            timeline_items = []
            for index, activity in enumerate(day.activities, start=1):
                meta = []
                if activity.duration_hours:
                    meta.append(f"{activity.duration_hours:g} h")
                if activity.cost:
                    meta.append(_format_currency(activity.cost, result.itinerary.currency))
                timeline_items.append(
                    {
                        "index": index,
                        "name": activity.name,
                        "category": _category_label(activity.category),
                        "meta": " · ".join(meta),
                        "details": _compact_description(activity.description, 180) if activity.description else "",
                    }
                )
            st.markdown(_timeline_html(timeline_items), unsafe_allow_html=True)
            if day.notes:
                with st.expander("Hinweise", expanded=False):
                    st.markdown(_bullet_list(day.notes))


def _render_export_panel(result: TravelPlanResult) -> None:
    st.divider()
    st.markdown("### Export")
    if not result.validation.ok:
        st.info("Export wird angeboten, sobald der finale Plan die Validation besteht.")
        return

    try:
        export = _get_cached_trip_export(result)
    except (PdfGenerationError, Exception) as exc:
        st.error(f"PDF konnte nicht vorbereitet werden: {_friendly_exception(exc)}")
        return

    col_1, col_2 = st.columns(2)
    with col_1:
        st.download_button(
            "PDF herunterladen",
            data=export.pdf_bytes,
            file_name=export.filename,
            mime="application/pdf",
            use_container_width=True,
        )
    with col_2:
        if st.button("An Discord senden", type="primary", use_container_width=True):
            try:
                with st.spinner("Reiseplan wird an Discord gesendet..."):
                    deliver_trip_to_discord(result, export=export)
                st.success("Der Reiseplan wurde an Discord gesendet.")
            except DiscordDeliveryError as exc:
                st.error(_friendly_exception(exc))
            except Exception as exc:
                st.error(f"Discord-Versand fehlgeschlagen: {_friendly_exception(exc)}")


def _get_cached_trip_export(result: TravelPlanResult):
    plan_hash = f"{PDF_EXPORT_CACHE_VERSION}:{calculate_plan_hash(result)}"
    cached_hash = st.session_state.get("trip_export_hash")
    cached_export = st.session_state.get("trip_export")
    if cached_hash == plan_hash and cached_export is not None:
        return cached_export
    export = create_trip_export(result)
    st.session_state.trip_export = export
    st.session_state.trip_export_hash = plan_hash
    return export


def _clear_export_cache() -> None:
    st.session_state.trip_export = None
    st.session_state.trip_export_hash = ""


def _get_cached_sample_trip_export():
    plan = _sample_pdf_plan()
    plan_hash = f"{PDF_EXPORT_CACHE_VERSION}:sample:{calculate_plan_hash(plan)}"
    cached_hash = st.session_state.get("sample_trip_export_hash")
    cached_export = st.session_state.get("sample_trip_export")
    if cached_hash == plan_hash and cached_export is not None:
        return cached_export
    export = create_trip_export(plan)
    st.session_state.sample_trip_export = export
    st.session_state.sample_trip_export_hash = plan_hash
    return export


def _sample_pdf_plan() -> dict[str, Any]:
    return {
        "itinerary": {
            "destination": "Madrid",
            "currency": "EUR",
            "total_cost": 285,
            "days": [
                {
                    "day": 1,
                    "total_cost": 135,
                    "total_duration_hours": 6.5,
                    "notes": ["Für Casa Alberto und das Fußballmuseum ist eine Reservierung sinnvoll."],
                    "activities": [
                        _sample_activity(
                            "Casa Alberto",
                            "food",
                            "typical Spanish cuisine",
                            "C. de las Huertas, 18, Centro, 28012 Madrid, Spain",
                            "4.4/5",
                            "5248",
                            "https://www.casaalberto.es/",
                            "https://maps.google.com/?cid=5012004565998211204",
                            35,
                            1.5,
                        ),
                        _sample_activity(
                            "Legends: The Home of Football",
                            "culture",
                            "watch a football match in Madrid",
                            "Cra de S. Jeronimo, 2, Centro, 28014 Madrid, Spain",
                            "4.8/5",
                            "6117",
                            "https://www.legendsmuseo.com/",
                            "https://maps.google.com/?cid=12684119845016445530",
                            25,
                            2,
                        ),
                        _sample_activity(
                            "Mercado de San Miguel",
                            "food",
                            "enjoy local Spanish food in Madrid",
                            "Pl. de San Miguel, s/n, Centro, 28005 Madrid, Spain",
                            "4.4/5",
                            "142000",
                            "https://mercadodesanmiguel.es/",
                            "https://maps.google.com/?cid=123456789",
                            40,
                            1.5,
                        ),
                        _sample_activity(
                            "Templo de Debod",
                            "culture",
                            "architecture experiences",
                            "C. de Ferraz, 1, Moncloa-Aravaca, 28008 Madrid, Spain",
                            "4.4/5",
                            "57000",
                            "",
                            "https://maps.google.com/?cid=987654321",
                            0,
                            1.5,
                        ),
                    ],
                },
                {
                    "day": 2,
                    "total_cost": 150,
                    "total_duration_hours": 6,
                    "notes": ["Stadiontour vormittags planen, danach bleibt genug Zeit für Tapas und Spaziergang."],
                    "activities": [
                        _sample_activity(
                            "Bernabeu",
                            "sport",
                            "watch a football match in Madrid",
                            "Av. de Concha Espina, 1, Chamartin, 28036 Madrid, Spain",
                            "4.7/5",
                            "233",
                            "https://bernabeu.realmadrid.com/",
                            "https://maps.google.com/?cid=12177723301084993928",
                            35,
                            2,
                        ),
                        _sample_activity(
                            "El Retiro Park",
                            "nature",
                            "nature experiences",
                            "Plaza de la Independencia, 7, Retiro, 28001 Madrid, Spain",
                            "4.8/5",
                            "180000",
                            "",
                            "https://maps.google.com/?cid=1020304050",
                            0,
                            1.5,
                        ),
                        _sample_activity(
                            "La Mi Venta",
                            "food",
                            "typical Spanish cuisine",
                            "Pl. de la Marina Espanola, 7, Centro, 28013 Madrid, Spain",
                            "4.7/5",
                            "7688",
                            "https://www.lamiventa.com/",
                            "https://maps.google.com/?cid=9750852188347724557",
                            40,
                            1.5,
                        ),
                        _sample_activity(
                            "Rooftop Circulo de Bellas Artes",
                            "activity",
                            "architecture experiences",
                            "C. de Alcala, 42, Centro, 28014 Madrid, Spain",
                            "4.4/5",
                            "12000",
                            "https://www.circulobellasartes.com/",
                            "https://maps.google.com/?cid=2030405060",
                            75,
                            1,
                        ),
                    ],
                },
            ],
        },
        "validation": {"ok": True, "error_count": 0, "warning_count": 0, "issues": []},
        "request": {
            "destination": "Madrid",
            "duration_days": 2,
            "budget": 350,
            "must_have": ["typical Spanish cuisine", "watch a football match in Madrid", "architecture experiences"],
            "avoid": ["clubs"],
            "travel_style": "balanced",
        },
        "weather_summary": {"summary": "Madrid: sonnig, warm und mit geringer Regenwahrscheinlichkeit."},
        "explanation": {
            "summary": (
                "Der Beispielplan kombiniert klassische Madrider Küche, Fußballkultur, "
                "Architektur und entspannte Stadterlebnisse in zwei gut gefüllten Tagen."
            )
        },
        "agentic_quality_review": {
            "summary": "Das geplante Budget wird sinnvoll genutzt und die Qualitätsprüfung wurde bestanden."
        },
    }


def _sample_activity(
    name: str,
    category: str,
    must_have: str,
    address: str,
    rating: str,
    reviews: str,
    website: str,
    maps_url: str,
    cost: float,
    duration_hours: float,
) -> dict[str, Any]:
    parts = [
        f"Category: {category}",
        "Matched query: PDF test export",
        f"Matched must-have: {must_have}",
        f"Address: {address}",
        f"Rating: {rating}",
        f"Reviews: {reviews}",
    ]
    if website:
        parts.append(f"Website: {website}")
    parts.append(f"Google Maps: {maps_url}")
    return {
        "name": name,
        "category": category,
        "description": " | ".join(parts),
        "cost": cost,
        "duration_hours": duration_hours,
        "source": "pdf_test_fixture",
    }


def _render_revision_panel(result: TravelPlanResult, parsed_request: TravelRequest) -> None:
    st.divider()
    st.markdown("### Plan wie im Reisebuero anpassen")
    versions = st.session_state.get("plan_versions", [])
    if versions:
        st.caption(" · ".join(f"Version {item['version']}: {item['label']}" for item in versions[-5:]))

    seed = st.session_state.pop("revision_seed", "")
    feedback = st.text_area(
        "Was soll geaendert werden?",
        value=seed,
        placeholder="z. B. Das Restaurant kenne ich schon, bitte ersetzen. Oder: Mehr Anime-Laeden. Oder: Tag 2 ist zu voll.",
        height=92,
        key="revision_feedback_input",
    )
    col_1, col_2, col_3 = st.columns(3)
    with col_1:
        if st.button("Restaurant ersetzen", use_container_width=True):
            st.session_state.revision_seed = "Das Restaurant kenne ich schon, bitte durch eine aehnliche Alternative ersetzen."
            st.rerun()
    with col_2:
        if st.button("Mehr davon", use_container_width=True):
            st.session_state.revision_seed = "Bitte mehr davon einbauen, ohne den Plan zu voll zu machen."
            st.rerun()
    with col_3:
        if st.button("Weniger stressig", use_container_width=True):
            st.session_state.revision_seed = "Der Plan ist zu stressig. Bitte mache ihn entspannter."
            st.rerun()

    if st.button("Plan anpassen", type="primary", disabled=not feedback.strip(), use_container_width=True):
        try:
            with st.spinner("Revision Agent passt den Plan an..."):
                revised = revise_travel_plan(
                    previous_result=result,
                    feedback=feedback,
                    original_inputs={
                        "destination": parsed_request.destination,
                        "days": parsed_request.duration_days,
                        "budget": parsed_request.budget,
                        "must_have": parsed_request.must_have,
                        "avoid": parsed_request.avoid,
                        "interest_tags": parsed_request.interest_tags,
                        "query_hints": parsed_request.query_hints,
                        "travel_style": parsed_request.travel_style,
                        "use_profile_memory": parsed_request.use_profile_memory,
                    },
                )
            st.session_state.last_result = revised
            st.session_state.plan_versions.append(
                {"version": len(st.session_state.plan_versions) + 1, "label": "Anpassung", "feedback": feedback}
            )
            _clear_export_cache()
            st.success("Plan wurde angepasst.")
            st.rerun()
        except Exception as exc:
            st.error(f"Anpassung fehlgeschlagen: {_friendly_exception(exc)}")


def _render_tech_view(
    result: TravelPlanResult | None,
    parsed_request: TravelRequest | None,
    sidebar_state: dict[str, Any],
    profile,
) -> None:
    st.markdown("### Technik")
    col_1, col_2, col_3 = st.columns(3)
    col_1.metric("AI Provider", ai_provider())
    col_2.metric("Profil", st.session_state.user_id)
    col_3.metric("Versionen", len(st.session_state.get("plan_versions", [])))

    payload = {
        "environment": {
            "AI_PROVIDER": os.getenv("AI_PROVIDER", ""),
            "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
            "GOOGLE_PLACES_API_KEY": bool(os.getenv("GOOGLE_PLACES_API_KEY")),
            "WEATHER_API_KEY": bool(os.getenv("WEATHER_API_KEY")),
        },
        "last_inputs": st.session_state.get("last_inputs", {}),
        "parsed_request": _request_to_dict(parsed_request) if parsed_request else {},
        "profile": _to_jsonable(profile),
        "gmail_sources": [_source_to_dict(source) for source in sidebar_state.get("gmail_sources", [])],
        "plan_versions": st.session_state.get("plan_versions", []),
    }
    prepared = st.session_state.get("prepared_context")
    if prepared:
        payload["prepared_context"] = {
            "request": _request_to_dict(prepared.request),
            "candidate_count": len(prepared.activities),
            "questions": prepared.questions,
            "tool_workflow": prepared.agentic_tool_workflow,
        }
    if result:
        payload["result"] = _result_to_dict(result)

    st.download_button(
        "JSON export",
        data=json.dumps(payload, ensure_ascii=False, indent=2),
        file_name="travelai_debug.json",
        mime="application/json",
        use_container_width=True,
    )

    st.divider()
    st.markdown("#### PDF-Testexport")
    st.caption("Erzeugt eine feste Beispielreise direkt aus dem Export-System, ohne KI-, Places- oder Wetteraufrufe.")
    try:
        test_export = _get_cached_sample_trip_export()
        st.download_button(
            "Test-PDF herunterladen",
            data=test_export.pdf_bytes,
            file_name=test_export.filename,
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as exc:
        st.error(f"Test-PDF konnte nicht erzeugt werden: {_friendly_exception(exc)}")

    with st.expander("Umgebung", expanded=True):
        st.json(payload["environment"])
        with st.expander("Parsed Request", expanded=True):
            st.json(payload["parsed_request"])
    if prepared:
        with st.expander("Prepared Interactive Context", expanded=True):
            st.json(payload["prepared_context"])
    if result:
        with st.expander("Query Planning", expanded=True):
            st.json(
                {
                    "summary": result.query_planning,
                    "queries": [
                        {
                            "query": query.query,
                            "reason": query.reason,
                            "source": query.source,
                            "must_have": query.must_have,
                        }
                        for query in result.place_queries
                    ],
                }
            )
        with st.expander("Workflow Steps", expanded=False):
            for index, step in enumerate(result.workflow_steps, start=1):
                st.write(f"{index}. {step}")
        with st.expander("Validation", expanded=False):
            st.json(validation_to_dict(result.validation))
        with st.expander("Itinerary", expanded=False):
            st.json(itinerary_to_dict(result.itinerary))
        with st.expander("Activity Evaluation", expanded=False):
            st.json(result.activity_evaluation)
        with st.expander("Agentic / Quality", expanded=False):
            st.json({"tool_workflow": result.agentic_tool_workflow, "quality_review": result.agentic_quality_review})
        with st.expander("Revision", expanded=False):
            st.json(result.revision or {})
        with st.expander("Costs", expanded=False):
            st.json(result.cost_report)


def _build_preference_sources(uploaded_files, travel_ratings: str, feedback: str) -> list[PreferenceSource]:
    sources: list[PreferenceSource] = []
    for uploaded_file in uploaded_files or []:
        raw = uploaded_file.getvalue()
        text = raw.decode("utf-8", errors="ignore")
        sources.append(PreferenceSource(source_type="upload", name=uploaded_file.name, text=text))
    if travel_ratings.strip():
        sources.append(PreferenceSource(source_type="travel_rating", name="manual_travel_ratings", text=travel_ratings))
    if feedback.strip():
        sources.append(PreferenceSource(source_type="feedback", name="current_feedback", text=feedback))
    return sources


def _init_state() -> None:
    st.session_state.setdefault("user_id", DEFAULT_USER_ID)
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_parsed_request", None)
    st.session_state.setdefault("last_inputs", {})
    st.session_state.setdefault("plan_versions", [])
    st.session_state.setdefault("prepared_context", None)
    st.session_state.setdefault("pending_conflict_result", None)
    st.session_state.setdefault("pending_conflict_decisions", {})
    st.session_state.setdefault("planning_stage", "INPUT")
    st.session_state.setdefault("main_view", "KI")
    st.session_state.setdefault("gmail_sources", [])
    st.session_state.setdefault("gmail_messages", [])
    st.session_state.setdefault("gmail_account_email", "")
    st.session_state.setdefault("trip_export", None)
    st.session_state.setdefault("trip_export_hash", "")
    st.session_state.setdefault("sample_trip_export", None)
    st.session_state.setdefault("sample_trip_export_hash", "")


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
          :root {
            --bg: #07111f;
            --surface: #0e1b2c;
            --surface-2: #13263d;
            --border: #24415f;
            --text: #edf4ff;
            --muted: #bfd0e2;
            --accent: #5da0ff;
            --good: #57c58f;
            --warn: #f0b85f;
            --bad: #ef7b7b;
          }
          .stApp, [data-testid="stAppViewContainer"], .main, .main .block-container {
            background: var(--bg);
            color: var(--text);
          }
          .main .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
          section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a1728 0%, #07111f 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
          }
          section[data-testid="stSidebar"] * { color: #d9e7fb; }
          .app-header {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #0d1c31 0%, #132947 55%, #173d67 100%);
            border: 1px solid rgba(93,160,255,0.14);
            border-radius: 12px;
          }
          .app-header h1 { margin: 0; font-size: 1.65rem; color: #f4fbfc; }
          .app-header p { margin: 0.35rem 0 0 0; color: #b9cbe0; }
          .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
            color: #8bc4ff;
            font-weight: 700;
            margin-bottom: 0.25rem;
          }
          .header-badges { display: flex; flex-wrap: wrap; gap: 0.35rem; justify-content: flex-end; }
          .tag {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            border: 1px solid transparent;
            font-size: 0.76rem;
            font-weight: 600;
            white-space: nowrap;
          }
          .tag-muted { background: var(--surface-2); border-color: var(--border); color: #d6e3f5; }
          .tag-accent { background: rgba(93,160,255,0.14); color: var(--accent); }
          .tag-success { background: rgba(21,111,59,0.12); color: var(--good); }
          .tag-warn { background: rgba(154,103,0,0.12); color: var(--warn); }
          .tag-bad { background: rgba(163,34,34,0.12); color: var(--bad); }
          .info-panel {
            padding: 1rem;
            background: linear-gradient(180deg, #112237 0%, #0c1828 100%);
            border: 1px solid var(--border);
            border-radius: 12px;
          }
          .info-title { font-weight: 700; margin-bottom: 0.25rem; }
          .info-body { color: var(--muted); }
          .sidebar-title { font-size: 1.35rem; font-weight: 800; color: #d9e7fb; }
          .sidebar-user { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.75rem; }
          div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #11243a 0%, #0c1727 100%);
            border: 1px solid rgba(95,142,196,0.42);
            border-radius: 12px;
            padding: 0.8rem 0.9rem;
          }
          .stTextInput input, .stTextArea textarea, .stNumberInput input,
          .stSelectbox div[data-baseweb="select"] > div,
          .stMultiSelect div[data-baseweb="select"] > div {
            background: #dce6f2;
            border-color: #89a6c8;
            color: #111827;
          }
          .stButton button, .stDownloadButton button {
            border-radius: 10px;
            font-weight: 600;
            background: #335f9e;
            color: #f7fbfc;
            border: 1px solid rgba(93,160,255,0.2);
          }
          .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            background: rgba(8,18,31,0.58);
            border: 1px solid rgba(95,142,196,0.18);
            border-radius: 14px;
            padding: 0.35rem;
            margin-bottom: 1rem;
          }
          .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 0.45rem 0.9rem;
            color: #bfd0e2;
            font-weight: 700;
          }
          .stTabs [aria-selected="true"] {
            background: rgba(93,160,255,0.18);
            color: #f4fbff;
          }
          div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: rgba(95,142,196,0.28) !important;
            border-radius: 14px !important;
            background: linear-gradient(180deg, rgba(16,33,54,0.72), rgba(8,18,31,0.72));
          }
          .candidate-card-head {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            margin-bottom: 0.55rem;
          }
          .candidate-title {
            color: #f3f8ff;
            font-size: 1.02rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
          }
          .candidate-meta { display: flex; flex-wrap: wrap; gap: 0.35rem; }
          .candidate-price {
            color: #d6e3f5;
            font-size: 0.85rem;
            font-weight: 700;
            white-space: nowrap;
          }
          .memory-card {
            padding: 0.75rem 0.85rem;
            border: 1px solid rgba(95,142,196,0.28);
            border-radius: 12px;
            background: rgba(10,24,40,0.82);
            margin-bottom: 0.55rem;
          }
          .memory-card.success { border-left: 4px solid var(--good); }
          .memory-card.muted { border-left: 4px solid #7088a4; }
          .memory-source {
            color: #dce9fb;
            font-weight: 700;
            margin-bottom: 0.2rem;
          }
          .memory-effect { color: #aebfd5; }
          .timeline {
            position: relative;
            padding-left: 1.15rem;
            margin: 0.2rem 0 0.6rem 0;
          }
          .timeline:before {
            content: "";
            position: absolute;
            left: 0.35rem;
            top: 0.35rem;
            bottom: 0.35rem;
            width: 2px;
            background: linear-gradient(180deg, var(--accent), rgba(93,160,255,0.08));
          }
          .timeline-item {
            position: relative;
            margin: 0 0 0.85rem 0;
            padding: 0.75rem 0.9rem;
            border: 1px solid rgba(95,142,196,0.24);
            border-radius: 12px;
            background: rgba(8,18,31,0.72);
          }
          .timeline-item:before {
            content: "";
            position: absolute;
            left: -1.03rem;
            top: 1rem;
            width: 0.7rem;
            height: 0.7rem;
            border-radius: 999px;
            background: var(--accent);
            box-shadow: 0 0 0 4px rgba(93,160,255,0.13);
          }
          .timeline-top {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: baseline;
          }
          .timeline-name { font-weight: 800; color: #f3f8ff; }
          .timeline-meta { color: #aebfd5; font-size: 0.82rem; white-space: nowrap; }
          .timeline-desc { color: #9fb2ca; margin-top: 0.35rem; font-size: 0.84rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sidebar_list(title: str, values: list[Any]) -> None:
    with st.sidebar.expander(f"{title} ({len(values or [])})", expanded=False):
        if values:
            st.markdown(_render_tags([str(value) for value in values[:12]], "muted"), unsafe_allow_html=True)
        else:
            st.caption("Keine Daten.")


def _semantic_summary(validation) -> dict:
    issues = getattr(validation, "issues", []) or []
    missing = [issue.message for issue in issues if getattr(issue, "issue_type", "") == "must_have_gap"]
    avoid = [
        issue.message
        for issue in issues
        if getattr(issue, "issue_type", "") in {"semantic_avoid_conflict", "preference_conflict"}
    ]
    return {"ok": not missing and not avoid, "missing": missing, "avoid": avoid}


def _source_to_dict(source: PreferenceSource) -> dict:
    return {"source_type": source.source_type, "name": source.name, "text": source.text}


def _request_to_dict(request: TravelRequest | None) -> dict:
    return asdict(request) if request else {}


def _result_to_dict(result: TravelPlanResult) -> dict:
    data = asdict(result)
    return _to_jsonable(data)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _to_jsonable(value.to_dict())
    return str(value)


def _parse_list(text: str) -> list[str]:
    if not str(text).strip():
        return []
    chunks = []
    for line in str(text).replace(";", "\n").splitlines():
        chunks.extend(part.strip() for part in line.split(","))
    return [chunk for chunk in chunks if chunk]


def _merge_unique(*groups: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group or []:
            cleaned = " ".join(str(value).strip().split())
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            result.append(cleaned)
    return result


def _labels_for_values(values: list[str], choices: list[tuple[str, str]]) -> list[str]:
    wanted = {str(value).strip().lower() for value in values or []}
    return [label for label, internal in choices if internal.lower() in wanted]


def _values_for_labels(labels: list[str], choices: list[tuple[str, str]]) -> list[str]:
    wanted = {str(label).strip().lower() for label in labels or []}
    return [internal for label, internal in choices if label.lower() in wanted]


def _value_for_label(label: str, choices: list[tuple[str, str]]) -> str:
    for item_label, value in choices:
        if item_label == label:
            return value
    return choices[0][1]


def _safe_user_id(value: str) -> str:
    return "".join(char for char in str(value).strip() if char.isalnum() or char in ("-", "_"))


def _safe_widget_key(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in str(value).strip().lower())
    return cleaned[:80] or "item"


def _category_label(category: str) -> str:
    labels = {
        "food": "Essen",
        "street_food": "Street Food",
        "nature": "Natur",
        "culture": "Kultur",
        "history": "Geschichte",
        "architecture": "Architektur",
        "shopping": "Shopping",
        "sport": "Sport",
        "gaming": "Gaming",
        "anime": "Anime",
        "technology": "Technik",
        "nightlife": "Nightlife",
        "local spots": "Lokale Orte",
    }
    return labels.get(str(category).strip().lower(), str(category).replace("_", " ").title())


def _compact_description(description: str, limit: int = 260) -> str:
    text = " ".join(str(description).split())
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _place_details(description: str) -> list[str]:
    details: list[str] = []
    text = str(description or "")
    for marker, label in [
        ("Rating:", "Rating"),
        ("Reviews:", "Reviews"),
        ("Address:", "Adresse"),
        ("Matched query:", "Query"),
        ("Matched must-have:", "Wunsch"),
    ]:
        if marker not in text:
            continue
        value = text.split(marker, 1)[1].split("|", 1)[0].strip()
        if value:
            details.append(f"{label}: {_compact_description(value, 48)}")
    return details


def _memory_item_card(memory: Any, effect: Any, kind: str = "success") -> str:
    return (
        f"<div class='memory-card {html.escape(kind)}'>"
        f"<div class='memory-source'>{html.escape(_compact_description(str(memory), 180))}</div>"
        f"<div class='memory-effect'>{html.escape(_compact_description(str(effect), 220))}</div>"
        "</div>"
    )


def _timeline_html(items: list[dict[str, Any]]) -> str:
    html_items = []
    for item in items:
        category = _pill(item.get("category", ""), "accent")
        meta = html.escape(str(item.get("meta") or ""))
        desc = html.escape(str(item.get("details") or ""))
        html_items.append(
            "<div class='timeline-item'>"
            "<div class='timeline-top'>"
            f"<div><span class='timeline-name'>{item.get('index')}. {html.escape(str(item.get('name') or ''))}</span> {category}</div>"
            f"<div class='timeline-meta'>{meta}</div>"
            "</div>"
            + (f"<div class='timeline-desc'>{desc}</div>" if desc else "")
            + "</div>"
        )
    return "<div class='timeline'>" + "".join(html_items) + "</div>"


def _format_currency(value: float, currency: str = "EUR") -> str:
    symbol = {"EUR": "EUR", "USD": "USD"}.get(str(currency).upper(), str(currency))
    return f"{value:,.0f} {symbol}".replace(",", ".")


def _pill(text: str, kind: str = "muted") -> str:
    return f"<span class='tag tag-{kind}'>{html.escape(str(text))}</span>"


def _render_tags(values: list[Any], kind: str = "muted") -> str:
    return "".join(_pill(str(value), kind) for value in values if str(value).strip())


def _info_panel(title: str, body: str) -> str:
    return (
        "<div class='info-panel'>"
        f"<div class='info-title'>{html.escape(title)}</div>"
        f"<div class='info-body'>{html.escape(body)}</div>"
        "</div>"
    )


def _bullet_list(values: list[Any]) -> str:
    items = [str(value).strip() for value in values if str(value).strip()]
    return "\n".join(f"- {html.escape(item)}" for item in items)


def _bullets_panel(title: str, values: list[Any]) -> str:
    bullets = "".join(f"<li>{html.escape(str(item))}</li>" for item in values if str(item).strip())
    return f"<div class='info-panel'><div class='info-title'>{html.escape(title)}</div><ul>{bullets}</ul></div>"


if __name__ == "__main__":
    main()
