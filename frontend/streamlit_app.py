from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from app.agents.request_agent import parse_travel_request
from app.models.preference_source import PreferenceSource
from app.models.travel_request import TravelRequest
from app.orchestrator import build_travel_plan
from app.rag.user_memory import load_user_profile
from app.tools.openai_runtime import MissingLocalAIError, MissingOpenAIKeyError, ai_provider


INTEREST_OPTIONS = [
    "food",
    "culture",
    "nightlife",
    "gaming",
    "anime",
    "nature",
    "luxury",
    "budget",
    "relaxed",
    "adventure",
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
col_a, col_b, col_c = st.columns(3)
with col_a:
    destination = st.text_input("Ziel", value="Barcelona")
with col_b:
    days = st.number_input("Tage", min_value=1, max_value=14, value=3, step=1)
with col_c:
    budget = st.number_input("Budget", min_value=50, max_value=10000, value=600, step=50)

generate = st.button("Reiseplan erstellen", type="primary")

if generate:
    try:
        preference_sources = _build_preference_sources(uploaded_files, travel_ratings, feedback)
        parsed_request = parse_travel_request(
            request_text,
            TravelRequest(
                destination=destination,
                duration_days=int(days),
                budget=float(budget),
                interests=interests,
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
            )
    except (MissingOpenAIKeyError, MissingLocalAIError) as exc:
        st.error(str(exc))
        st.stop()

    st.markdown("### Erkannte Reiseanfrage")
    st.json(
        {
            "destination": parsed_request.destination,
            "duration_days": parsed_request.duration_days,
            "budget": parsed_request.budget,
            "interests": parsed_request.interests,
            "travel_style": parsed_request.travel_style,
        }
    )

    st.markdown("### Agent Workflow")
    for step in result.workflow_steps:
        st.write(f"- {step}")

    profile_col, weather_col, validation_col = st.columns(3)
    with profile_col:
        st.markdown("### Profil")
        st.json(
            {
                "user_id": result.profile.user_id,
                "interests": result.profile.interests,
                "travel_style": result.profile.travel_style,
                "budget_preference": result.profile.budget_preference,
                "avoid": result.profile.avoid,
                "preferred_day_structure": result.profile.preferred_day_structure,
                "past_destinations": result.profile.past_destinations,
                "uploaded_sources": result.profile.uploaded_sources,
            }
        )
    with weather_col:
        st.markdown("### Wetter")
        st.json(result.weather)
    with validation_col:
        st.markdown("### Validation")
        st.write("Optimiert:", "Ja" if result.optimized else "Nein")
        if result.validation.ok:
            st.success("Plan ist valide.")
        else:
            for issue in result.validation.issues:
                st.warning(f"Tag {issue.day or '-'}: {issue.message}")

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

    st.markdown("## Reiseplan")
    for day in result.itinerary.days:
        with st.container(border=True):
            st.markdown(f"### Tag {day.day}")
            for activity in day.activities:
                st.markdown(
                    f"**{activity.name}**  \n"
                    f"{activity.category} | {activity.duration_hours:g}h | "
                    f"{activity.cost:g} {result.itinerary.currency} | "
                    f"{'Indoor' if activity.indoor else 'Outdoor'}"
                )
                if activity.description:
                    st.caption(activity.description)
            if day.notes:
                st.info(" ".join(day.notes))
            st.write(f"Tagessumme: {day.total_cost:g} {result.itinerary.currency}")

    st.markdown("## Final Travel Package")
    package_col, todo_col = st.columns(2)
    with package_col:
        st.markdown("### Budgetuebersicht")
        st.metric("Gesamtkosten", f"{result.itinerary.total_cost:g} {result.itinerary.currency}")
        st.metric("Budget", f"{parsed_request.budget:g} {result.itinerary.currency}")
        st.markdown("### Warum der Plan passt")
        st.write(", ".join(result.profile.interests) or "Keine Praeferenzen erkannt.")
    with todo_col:
        st.markdown("### Packliste")
        pack_items = ["bequeme Schuhe", "Powerbank", "Reisedokumente"]
        if result.weather.get("rain_expected"):
            pack_items.append("Regenjacke")
        st.write(pack_items)
        st.markdown("### To-do vor der Reise")
        st.write(["API-Daten/Verfuegbarkeit pruefen", "Tickets reservieren", "Route offline speichern"])

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
