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
from app.models.preference_source import PreferenceSource
from app.models.travel_request import TravelRequest
from app.orchestrator import TravelPlanResult, build_travel_plan, revise_travel_plan
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
from app.tools.openai_runtime import MissingLocalAIError, MissingOpenAIKeyError, ai_provider


APP_TITLE = "TravelAI"
APP_SUBTITLE = "Agentischer Reiseplaner mit konkreten Suchqueries, Memory und interaktiver Anpassung"

DEFAULT_USER_ID = "demo_user_1"
DEFAULT_REQUEST = (
    "Ich will 2 Tage nach Paris, typische franzoesische Kueche und Anime-Laeden, "
    "aber kein Sport und keine Touristenfallen."
)

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
MAIN_VIEWS = ["KI", "Reiseplan", "Technik"]


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="AI", layout="wide", initial_sidebar_state="expanded")
    _apply_styles()
    _init_state()

    current_profile = load_user_profile(st.session_state.user_id)
    sidebar_state = _render_sidebar(current_profile)

    _render_header()
    pending_view = st.session_state.pop("pending_main_view", None)
    if pending_view:
        st.session_state.main_view = pending_view
    selected_view = st.radio("Ansicht", MAIN_VIEWS, horizontal=True, key="main_view", label_visibility="collapsed")

    result = st.session_state.get("last_result")
    parsed_request = st.session_state.get("last_parsed_request")

    if selected_view == "KI":
        _render_ai_view(current_profile, sidebar_state, result)
    elif selected_view == "Reiseplan":
        _render_plan_view(result, parsed_request)
    else:
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
                    st.error(str(exc))

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
                st.error(str(exc))
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

        submitted = st.form_submit_button("Reiseplan erstellen", type="primary", use_container_width=True)

    if submitted:
        _run_initial_plan(
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

    if result:
        st.markdown("### Letzter Stand")
        _render_ai_summary(result)
    else:
        _render_empty_state(profile, sidebar_state)


def _run_initial_plan(
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
    briefing = "\n".join(part for part in [request_text, f"Must-have: {must_have_text}", f"Vermeiden: {avoid_text}"] if part.strip())
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

        with st.spinner("Agenten-Workflow wird ausgefuehrt..."):
            result = build_travel_plan(
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
            )
    except (MissingOpenAIKeyError, MissingLocalAIError) as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Planung fehlgeschlagen: {exc}")
        return

    st.session_state.last_result = result
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
    }
    st.session_state.plan_versions = [{"version": 1, "label": "Erstplan", "feedback": ""}]
    st.session_state.pending_main_view = "Reiseplan"
    st.rerun()


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
    st.markdown("### Reiseplan")
    if not result or not parsed_request:
        st.markdown(_info_panel("Noch kein Reiseplan vorhanden.", "Erstelle zuerst im KI-Tab einen Plan."), unsafe_allow_html=True)
        return

    summary = (result.explanation or {}).get("summary") or f"Plan fuer {result.itinerary.destination}."
    st.markdown(_info_panel(f"Reise nach {result.itinerary.destination}", summary), unsafe_allow_html=True)
    _render_wish_coverage(result.validation)
    _render_itinerary(result)
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
        with st.expander(f"Tag {day.day}", expanded=day.day == 1):
            for index, activity in enumerate(day.activities, start=1):
                meta = []
                if activity.duration_hours:
                    meta.append(f"{activity.duration_hours:g} h")
                if activity.cost:
                    meta.append(_format_currency(activity.cost, result.itinerary.currency))
                st.markdown(f"{index}. **{html.escape(activity.name)}**")
                st.caption(f"{_category_label(activity.category)}" + (f" | {' | '.join(meta)}" if meta else ""))
                if activity.description:
                    st.caption(_compact_description(activity.description))
            if day.notes:
                st.markdown("**Hinweise**")
                st.markdown(_bullet_list(day.notes))
            st.write(f"Tagessumme: {_format_currency(day.total_cost, result.itinerary.currency)}")


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
                    },
                )
            st.session_state.last_result = revised
            st.session_state.plan_versions.append(
                {"version": len(st.session_state.plan_versions) + 1, "label": "Anpassung", "feedback": feedback}
            )
            st.success("Plan wurde angepasst.")
            st.rerun()
        except Exception as exc:
            st.error(f"Anpassung fehlgeschlagen: {exc}")


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
    if result:
        payload["result"] = _result_to_dict(result)

    st.download_button(
        "JSON export",
        data=json.dumps(payload, ensure_ascii=False, indent=2),
        file_name="travelai_debug.json",
        mime="application/json",
        use_container_width=True,
    )

    with st.expander("Umgebung", expanded=True):
        st.json(payload["environment"])
    with st.expander("Parsed Request", expanded=True):
        st.json(payload["parsed_request"])
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
    st.session_state.setdefault("main_view", "KI")
    st.session_state.setdefault("gmail_sources", [])
    st.session_state.setdefault("gmail_messages", [])
    st.session_state.setdefault("gmail_account_email", "")


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
