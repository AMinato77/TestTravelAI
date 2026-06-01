from __future__ import annotations

import sys
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
from app.rag.user_memory import load_user_profile
from app.tools.openai_runtime import MissingLocalAIError, MissingOpenAIKeyError, ai_provider


INTEREST_OPTIONS = [
    "food",
    "culture",
    "history",
    "sport",
    "nightlife",
    "gaming",
    "anime",
    "nature",
    "local spots",
    "shopping",
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
    user_id = st.text_input("User ID", value="demo_user_1")
    travel_style = st.selectbox("Reisestil", ["balanced", "relaxed", "adventure", "luxury", "budget"])
    budget_preference = st.selectbox("Budgetpraeferenz", ["low", "medium", "high"])
    interests = st.multiselect("Interessen", INTEREST_OPTIONS, default=["food", "culture"])
    feedback = st.text_area("Neues Feedback optional", placeholder="z.B. mehr Food, weniger Museen")

    memory = load_user_profile(user_id)
    with st.expander("Gespeichertes Memory", expanded=False):
        st.write(f"Reisestil: {memory.travel_style}")
        st.write(f"Budgetpraeferenz: {memory.budget_preference}")
        st.markdown("**Interessen**")
        _show_items(memory.interests)
        st.markdown("**Bisherige Ziele**")
        _show_items(memory.past_destinations)
        with st.expander("Memory JSON", expanded=False):
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

st.subheader("Reiseparameter")
request_text = st.text_area(
    "Freie Reiseanfrage",
    value="Ich will 4 Tage nach Barcelona, Budget 700 Euro, ich mag Food, Gaming, Anime und lokale Spots und will keinen stressigen Plan.",
)

generate = st.button("Reiseplan erstellen", type="primary")

if generate:
    try:
        preference_sources = _build_preference_sources(uploaded_files, travel_ratings, feedback)
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

    map_points = [
        {"lat": activity.latitude, "lon": activity.longitude}
        for activity in result.activities
        if activity.latitude is not None and activity.longitude is not None
    ]
    if map_points:
        st.markdown("## Karte")
        st.map(map_points)
else:
    st.info("Gib Reiseparameter ein und erstelle den ersten Plan.")
