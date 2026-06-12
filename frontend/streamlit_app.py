from __future__ import annotations

import sys
import datetime as dt
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=True)

from app.agents.request_agent import parse_travel_request
from app.models.preference_source import PreferenceSource
from app.models.travel_request import TravelRequest
from app.orchestrator import build_travel_plan
from app.rag.user_memory import create_user_profile, list_user_ids, load_user_profile
from app.tools.email_tool import fetch_email_sources
from app.tools.openai_runtime import MissingLocalAIError, MissingOpenAIKeyError, ai_provider


INTEREST_OPTIONS = [
    "food",
    "street food",
    "culture",
    "history",
    "sport",
    "nightlife",
    "gaming",
    "anime",
    "nature",
    "local spots",
    "hidden gems",
    "shopping",
    "technology",
    "photography",
    "architecture",
    "luxury",
]


def _build_preference_sources(uploaded_files, travel_ratings: str, feedback: str) -> list[PreferenceSource]:
    sources: list[PreferenceSource] = []
    for uploaded_file in uploaded_files or []:
        raw = uploaded_file.getvalue()
        text = raw.decode("utf-8", errors="ignore")
        sources.append(
            PreferenceSource(
                source_type="upload",
                name=uploaded_file.name,
                text=text,
            )
        )
    if travel_ratings.strip():
        sources.append(
            PreferenceSource(
                source_type="travel_rating",
                name="manual_travel_ratings",
                text=travel_ratings,
            )
        )
    if feedback.strip():
        sources.append(
            PreferenceSource(
                source_type="feedback",
                name="current_feedback",
                text=feedback,
            )
        )
    return sources


def _fetch_email_memory_sources(
    enabled: bool,
    host: str,
    username: str,
    password: str,
    folder: str,
    since: dt.date,
    limit: int,
) -> list[PreferenceSource]:
    if not enabled:
        return []
    return fetch_email_sources(
        host=host,
        username=username,
        password=password,
        folder=folder,
        since=since,
        limit=limit,
    )


def _show_validation(label: str, validation) -> None:
    st.markdown(f"### {label}")
    error_count = getattr(validation, "error_count", None)
    warning_count = getattr(validation, "warning_count", None)
    if error_count is None:
        error_count = sum(1 for issue in validation.issues if issue.severity == "error")
    if warning_count is None:
        warning_count = sum(1 for issue in validation.issues if issue.severity == "warning")
    st.write(f"Errors: {error_count} | Warnings: {warning_count}")
    if validation.ok:
        st.success("Plan ist valide.")
        return
    for issue in validation.issues:
        prefix = f"Tag {issue.day or '-'}"
        if issue.activity:
            prefix += f" | {issue.activity}"
        st.warning(f"{prefix}: {issue.issue_type} - {issue.message}")


def _show_items(items: list[str], empty_text: str = "Keine Daten erkannt.") -> None:
    cleaned = [item for item in items if item]
    if not cleaned:
        st.caption(empty_text)
        return
    for item in cleaned:
        st.write(f"- {item}")


def _select_user_profile() -> str:
    st.subheader("User")
    user_ids = list_user_ids()
    if not user_ids:
        create_user_profile("demo_user_1")
        user_ids = ["demo_user_1"]

    if "selected_user_id" not in st.session_state:
        st.session_state.selected_user_id = user_ids[0]

    if st.session_state.selected_user_id not in user_ids:
        st.session_state.selected_user_id = user_ids[0]

    selected_user = st.selectbox(
        "Vorhandene User",
        user_ids,
        index=user_ids.index(st.session_state.selected_user_id),
        key="user_selectbox",
    )
    st.session_state.selected_user_id = selected_user

    with st.expander("+ Neuen User erstellen", expanded=False):
        new_user_id = st.text_input("Neue User ID", placeholder="z.B. tokyo_user")
        if st.button("User erstellen", use_container_width=True):
            cleaned_user_id = _clean_user_id(new_user_id)
            if cleaned_user_id:
                create_user_profile(cleaned_user_id)
                st.session_state.selected_user_id = cleaned_user_id
                st.success(f"User '{cleaned_user_id}' wurde erstellt.")
                st.rerun()
            else:
                st.warning("Bitte eine gueltige User ID eingeben.")

    return st.session_state.selected_user_id


def _clean_user_id(value: str) -> str:
    return "".join(char for char in value.strip() if char.isalnum() or char in ("-", "_"))


def _show_request_summary(parsed_request: TravelRequest) -> None:
    st.markdown("### Erkannte Reiseanfrage")
    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric("Ziel", parsed_request.destination)
    col_2.metric("Tage", parsed_request.duration_days)
    col_3.metric("Budget", f"{parsed_request.budget:g} EUR")
    col_4.metric("Reisestil", parsed_request.travel_style)
    st.markdown("**Interessen**")
    st.write(", ".join(parsed_request.interests) or "Keine Interessen erkannt.")
    if parsed_request.avoid:
        st.markdown("**Abneigungen**")
        st.write(", ".join(parsed_request.avoid))
    with st.expander("Technische JSON-Ausgabe", expanded=False):
        st.json(
            {
                "destination": parsed_request.destination,
                "duration_days": parsed_request.duration_days,
                "budget": parsed_request.budget,
                "interests": parsed_request.interests,
                "avoid": parsed_request.avoid,
                "travel_style": parsed_request.travel_style,
            }
        )


def _show_profile_summary(profile) -> None:
    st.markdown("### Profil")
    st.write(f"**User:** {profile.user_id}")
    st.write(f"**Reisestil:** {profile.travel_style}")
    st.write(f"**Budgetpraeferenz:** {profile.budget_preference}")
    st.markdown("**Gelernte Interessen**")
    _show_items(profile.interests)
    st.markdown("**Meiden**")
    _show_items(profile.avoid)
    if profile.past_destinations:
        st.markdown("**Bisherige Ziele**")
        st.write(", ".join(profile.past_destinations))
    with st.expander("Profil als JSON", expanded=False):
        st.json(
            {
                "user_id": profile.user_id,
                "interests": profile.interests,
                "travel_style": profile.travel_style,
                "budget_preference": profile.budget_preference,
                "avoid": profile.avoid,
                "preferred_day_structure": profile.preferred_day_structure,
                "past_destinations": profile.past_destinations,
                "uploaded_sources": profile.uploaded_sources,
            }
        )


def _show_weather_summary(weather: dict) -> None:
    st.markdown("### Wetter")
    st.write(weather.get("summary", "Keine Wetterzusammenfassung vorhanden."))
    col_1, col_2 = st.columns(2)
    col_1.metric("Temperatur", f"{weather.get('temperature_c', '?')} °C")
    col_2.metric("Max. Regenchance", f"{weather.get('max_rain_chance', '?')}%")
    if weather.get("rain_expected"):
        st.warning("Regen erwartet. Indoor-Alternativen werden bevorzugt.")
    else:
        st.success("Kein relevanter Regen erwartet.")
    with st.expander("Wetter als JSON", expanded=False):
        st.json(weather)


def _show_ai_explanation(explanation: dict) -> None:
    st.markdown("## AI Explanation")
    if explanation.get("summary"):
        st.write(explanation["summary"])

    col_1, col_2 = st.columns(2)
    with col_1:
        st.markdown("### Warum dieser Plan passt")
        _show_items(explanation.get("preference_reasoning", []))
        st.markdown("### Datenquellen")
        _show_items(explanation.get("data_sources", []))
    with col_2:
        st.markdown("### Validierung")
        st.write(explanation.get("validation_result") or "Keine Validierungserklaerung vorhanden.")
        st.markdown("### Optimierung")
        st.write(explanation.get("optimization_result") or "Keine Optimierungserklaerung vorhanden.")

    caveats = explanation.get("caveats", [])
    if caveats:
        with st.expander("Hinweise"):
            _show_items(caveats)


def _show_activity_evaluation(evaluation: dict) -> None:
    removed = evaluation.get("removed") or []
    if not removed:
        st.success("Activity Evaluation Agent hat keine schwachen Kandidaten entfernt.")
        return
    with st.expander("Vom Activity Evaluation Agent entfernte Kandidaten", expanded=False):
        st.dataframe(
            [
                {
                    "name": item.get("name"),
                    "category": item.get("category"),
                    "source": item.get("source"),
                    "score": item.get("score"),
                    "reason": item.get("reason"),
                }
                for item in removed
            ],
            use_container_width=True,
            hide_index=True,
        )


def _show_memory_context(memory_context) -> None:
    st.markdown("### RAG Memory Context")
    if not memory_context:
        st.caption("Keine passenden Memory-Chunks aus ChromaDB abgerufen.")
        return
    with st.expander("Aus ChromaDB abgerufene User-Memory-Chunks", expanded=False):
        for index, source in enumerate(memory_context, start=1):
            st.markdown(f"**{index}. {source.name}** ({source.source_type})")
            facts = _profile_memory_facts(source.text) if source.source_type == "profile_snapshot" else []
            if facts:
                st.caption("Aus ChromaDB retrieved und fuer die Planung beruecksichtigt:")
                for fact in facts:
                    st.write(f"- {fact}")
            else:
                st.write(_memory_preview(source.text))


def _profile_memory_facts(text: str) -> list[str]:
    labels = {
        "Interests": "Interessen",
        "Avoid": "Meiden",
        "Travel style": "Reisestil",
        "Budget preference": "Budgetpraeferenz",
        "Past destinations": "Bisherige Ziele",
        "Feedback history": "Feedback",
        "Source notes": "Quellnotizen",
    }
    facts: list[str] = []
    for raw_line in text.splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        label = labels.get(key.strip())
        value = value.strip()
        if label and value and value.lower() != "none":
            facts.append(f"{label}: {value}")
    return facts


def _memory_preview(text: str) -> str:
    preview = text[:500].strip()
    if len(text) > 500:
        preview += "..."
    return preview


def _show_itinerary(itinerary, title: str) -> None:
    st.markdown(f"## {title}")
    for day in itinerary.days:
        with st.container(border=True):
            st.markdown(f"### Tag {day.day}")
            for activity in day.activities:
                st.markdown(
                    f"**{activity.name}**  \n"
                    f"{activity.category} | {activity.duration_hours:g}h | "
                    f"{activity.cost:g} {itinerary.currency} | "
                    f"{'Indoor' if activity.indoor else 'Outdoor'}"
                )
                if activity.description:
                    st.caption(activity.description)
            if day.notes:
                st.info(" ".join(day.notes))
            st.write(f"Tagessumme: {day.total_cost:g} {itinerary.currency}")


st.set_page_config(page_title="Adaptive AI Travel Agent", page_icon="AI", layout="wide")

st.title("Adaptive AI Travel Agent")
st.caption(f"AI Provider: {ai_provider()}")

with st.sidebar:
    st.header("Nutzerprofil")
    user_id = _select_user_profile()
    travel_style = "balanced"
    budget_preference = "medium"
    interests: list[str] = []
    feedback = ""

    memory = load_user_profile(user_id)
    with st.expander("Gespeichertes Memory", expanded=False):
        st.write(f"Reisestil: {memory.travel_style}")
        st.write(f"Budgetpraeferenz: {memory.budget_preference}")
        st.markdown("**Interessen**")
        _show_items(memory.interests)
        st.markdown("**Bisherige Ziele**")
        _show_items(memory.past_destinations)
        with st.expander("Chroma Memory Snapshot", expanded=False):
            st.json(memory.to_dict())

st.subheader("Preference Learning")
upload_col, rating_col = st.columns(2)
with upload_col:
    uploaded_files = st.file_uploader(
        "Chat-Exports oder Notizen hochladen",
        type=["txt", "md", "json", "csv"],
        accept_multiple_files=True,
    )
with rating_col:
    travel_ratings = st.text_area(
        "Reisebewertungen",
        placeholder="Barcelona: 9/10, Essen war super.\nParis: 5/10, zu touristisch.",
        height=140,
    )

with st.expander("Optional: E-Mail-Memory per IMAP verbinden", expanded=False):
    use_email_memory = st.checkbox("E-Mails live aus Postfach laden")
    email_col_1, email_col_2 = st.columns(2)
    with email_col_1:
        email_host = st.text_input("IMAP Host", placeholder="imap.gmail.com oder mail.htw-berlin.de")
        email_username = st.text_input("E-Mail Benutzername")
        email_password = st.text_input("E-Mail Passwort / App-Passwort", type="password")
    with email_col_2:
        email_folder = st.text_input("Ordner", value="INBOX")
        email_since = st.date_input("Mails ab Datum", value=dt.date.today() - dt.timedelta(days=365))
        email_limit = st.number_input("Max. Mails", min_value=1, max_value=50, value=10, step=1)
    st.caption(
        "Die App speichert nicht das Passwort. Geladene Mailtexte werden als Memory-Quelle "
        "in ChromaDB eingebettet und beim Planen per RAG verwendet."
    )

st.subheader("Reiseparameter")
request_text = st.text_area(
    "Freie Reiseanfrage",
    value="Ich will 4 Tage nach Barcelona, Budget 700 Euro, ich mag Food, Gaming, Anime und lokale Spots und will keinen stressigen Plan.",
)

generate = st.button("Reiseplan erstellen", type="primary")

if generate:
    try:
        preference_sources = _build_preference_sources(uploaded_files, travel_ratings, feedback)
        email_sources = _fetch_email_memory_sources(
            enabled=use_email_memory,
            host=email_host,
            username=email_username,
            password=email_password,
            folder=email_folder,
            since=email_since,
            limit=int(email_limit),
        )
        preference_sources.extend(email_sources)
        if email_sources:
            st.success(f"{len(email_sources)} E-Mail(s) als Memory-Quelle geladen.")
        parsed_request = parse_travel_request(
            request_text,
            TravelRequest(
                destination="Barcelona",
                duration_days=3,
                budget=600,
                interests=interests,
                avoid=[],
                travel_style=travel_style,
            ),
        )
        with st.spinner("Agenten-Workflow wird ausgefuehrt..."):
            result = build_travel_plan(
                user_id=user_id,
                destination=parsed_request.destination,
                days=parsed_request.duration_days,
                budget=parsed_request.budget,
                manual_interests=parsed_request.interests,
                travel_style=parsed_request.travel_style,
                budget_preference=budget_preference,
                feedback=feedback or None,
                preference_sources=preference_sources,
                manual_avoid=parsed_request.avoid,
            )
    except (MissingOpenAIKeyError, MissingLocalAIError) as exc:
        st.error(str(exc))
        st.stop()

    _show_request_summary(parsed_request)

    st.markdown("### Agent Workflow")
    for step in result.workflow_steps:
        st.write(f"- {step}")

    profile_col, weather_col, validation_col = st.columns(3)
    with profile_col:
        _show_profile_summary(result.profile)
        _show_memory_context(result.memory_context)
    with weather_col:
        _show_weather_summary(result.weather)
    with validation_col:
        st.write("Optimiert:", "Ja" if result.optimized else "Nein")
        _show_validation("Finale Validation", result.validation)

    if result.optimized:
        st.markdown("## Optimization Loop")
        before_col, after_col = st.columns(2)
        with before_col:
            _show_validation("Vor Optimierung", result.initial_validation)
        with after_col:
            _show_validation("Nach Optimierung", result.validation)

    st.markdown("## Gefundene Aktivitaeten")
    _show_activity_evaluation(result.activity_evaluation)
    st.dataframe(
        [
            {
                "name": activity.name,
                "category": activity.category,
                "cost": activity.cost,
                "duration_h": activity.duration_hours,
                "indoor": activity.indoor,
                "source": activity.source,
            }
            for activity in result.activities
        ],
        use_container_width=True,
        hide_index=True,
    )

    if result.optimized:
        with st.expander("Erster Plan vor Optimierung", expanded=False):
            _show_itinerary(result.initial_itinerary, "Initialer Reiseplan")

    _show_ai_explanation(result.explanation)

    _show_itinerary(result.itinerary, "Finaler Reiseplan")

    st.markdown("## Final Travel Package")
    package_col, todo_col = st.columns(2)
    with package_col:
        st.markdown("### Budgetuebersicht")
        st.metric("Gesamtkosten", f"{result.itinerary.total_cost:g} {result.itinerary.currency}")
        st.metric("Budget", f"{parsed_request.budget:g} {result.itinerary.currency}")
        st.markdown("### Warum der Plan passt")
        _show_items(result.profile.interests, "Keine Praeferenzen erkannt.")
    with todo_col:
        st.markdown("### Packliste")
        pack_items = ["bequeme Schuhe", "Powerbank", "Reisedokumente"]
        if result.weather.get("rain_expected"):
            pack_items.append("Regenjacke")
        for item in pack_items:
            st.write(f"- {item}")
        st.markdown("### To-do vor der Reise")
        for item in ["API-Daten/Verfuegbarkeit pruefen", "Tickets reservieren", "Route offline speichern"]:
            st.write(f"- {item}")

else:
    st.info("Gib Reiseparameter ein und erstelle den ersten Plan.")
