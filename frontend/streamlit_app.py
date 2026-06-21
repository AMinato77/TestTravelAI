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
from app.rag.memory_retrieval import delete_user_memory_sources
from app.rag.user_memory import create_user_profile, list_user_ids, load_user_profile
from app.services.interest_taxonomy import ALLOWED_INTERESTS
from app.tools.gmail_tool import (
    GmailIntegrationError,
    build_gmail_preference_source,
    get_gmail_account_email,
    gmail_credentials_available,
    save_gmail_credentials_file,
)
from app.tools.openai_runtime import MissingLocalAIError, MissingOpenAIKeyError, ai_provider


INTEREST_OPTIONS = ALLOWED_INTERESTS


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
        if not validation.issues:
            return
    for issue in validation.issues:
        prefix = f"Tag {issue.day or '-'}"
        if issue.activity:
            prefix += f" | {issue.activity}"
        if issue.severity == "error":
            st.error(f"{prefix}: {issue.issue_type} - {issue.message}")
        else:
            st.warning(f"{prefix}: {issue.issue_type} - {issue.message}")


def _show_items(items: list[str], empty_text: str = "Keine Daten erkannt.") -> None:
    cleaned = [item for item in items if item and str(item).strip().lower() not in {"none", "unknown", "null", "n/a", "-"}]
    if not cleaned:
        st.caption(empty_text)
        return
    for item in cleaned:
        st.write(f"- {item}")


def _show_sidebar_memory(profile, title: str = "Gespeichertes Memory", expanded: bool = False) -> None:
    with st.expander(title, expanded=expanded):
        st.write(f"Reisestil: {profile.travel_style}")
        st.write(f"Budgetpraeferenz: {profile.budget_preference}")
        st.markdown("**Interessen**")
        _show_items(profile.interests)
        st.markdown("**Bisherige Ziele**")
        _show_items(profile.past_destinations)
        with st.expander("Chroma Memory Snapshot", expanded=False):
            st.json(profile.to_dict())


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
    st.markdown("**Interessen aus aktuellem Freitext**")
    if parsed_request.interests:
        st.write(", ".join(parsed_request.interests))
    else:
        st.write("Keine Interessen im Freitext erkannt. Gmail/Memory-Interessen werden im Profil darunter verarbeitet.")
    if parsed_request.needs_destination_recommendation:
        st.info(
            f"Die Anfrage wurde als Ziel-Empfehlung erkannt "
            f"(Scope: {parsed_request.destination_scope})."
        )
    if parsed_request.avoid:
        st.markdown("**Abneigungen**")
        st.write(", ".join(parsed_request.avoid))
    with st.expander("Technische JSON-Ausgabe", expanded=False):
        st.json(
            {
                "destination": parsed_request.destination,
                "destination_scope": parsed_request.destination_scope,
                "needs_destination_recommendation": parsed_request.needs_destination_recommendation,
                "destination_reasoning": parsed_request.destination_reasoning,
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


def _show_agentic_quality_review(review: dict) -> None:
    if not review:
        return
    st.markdown("## Agents SDK Quality Review")
    if not review.get("enabled"):
        st.caption(review.get("summary", "Agents SDK Review wurde nicht ausgefuehrt."))
        return
    st.write(review.get("summary", "Keine Zusammenfassung vorhanden."))
    col_1, col_2 = st.columns(2)
    with col_1:
        st.markdown("### Budget")
        st.write(review.get("budget_assessment", "Keine Budgetbewertung vorhanden."))
    with col_2:
        st.markdown("### Validierung")
        st.write(review.get("validation_assessment", "Keine Validierungsbewertung vorhanden."))
    improvements = review.get("improvements") or []
    if improvements:
        st.markdown("### Verbesserungsideen")
        _show_items([str(item) for item in improvements])


def _show_agentic_tool_workflow(workflow: dict) -> None:
    st.markdown("## Agentic Tool Workflow")
    if not workflow:
        st.caption("Keine Tool-Workflow-Daten vorhanden.")
        return
    if not workflow.get("enabled"):
        st.caption(workflow.get("summary", "Tool Workflow Agent wurde nicht ausgefuehrt."))
    else:
        st.write(workflow.get("summary", "Tool Workflow Agent wurde ausgefuehrt."))
    tool_calls = workflow.get("tool_calls") or []
    if tool_calls:
        st.markdown("### Interne Tool Calls")
        st.dataframe(tool_calls, use_container_width=True, hide_index=True)
    guidance = workflow.get("planning_guidance") or []
    if guidance:
        st.markdown("### Tool-Empfehlung")
        _show_structured_items(guidance)
    tool_decision = workflow.get("tool_decision")
    if tool_decision:
        st.markdown("### Tool-Entscheidung")
        st.write(tool_decision)
    risks = workflow.get("risks") or []
    if risks:
        st.markdown("### Risiken")
        _show_structured_items(risks)
    coverage = workflow.get("interest_coverage") or {}
    if coverage:
        st.markdown("### Interest Coverage")
        st.json(coverage)
    if not workflow.get("enabled"):
        return


def _show_structured_items(value) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (list, dict)):
                st.markdown(f"**{key}**")
                st.json(item)
            elif str(item).strip():
                st.write(f"- **{key}:** {item}")
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                _show_structured_items(item)
            elif isinstance(item, list):
                _show_structured_items(item)
            elif str(item).strip():
                st.write(f"- {item}")
        return
    if str(value).strip():
        st.write(f"- {value}")


def _show_cost_report(cost_report: dict) -> None:
    st.markdown("## Cost Tracking")
    if not cost_report:
        st.caption("Keine Kostendaten vorhanden.")
        return
    st.metric("Geschaetzte Tool-Kosten", f"{cost_report.get('estimated_total_usd', 0):g} USD")
    col_1, col_2 = st.columns(2)
    with col_1:
        st.markdown("### Nach Provider")
        st.json(cost_report.get("by_provider", {}))
    with col_2:
        st.markdown("### Nach Tool")
        st.json(cost_report.get("by_tool", {}))
    traces = cost_report.get("traces") or []
    if traces:
        with st.expander("Cost Trace Details", expanded=False):
            st.dataframe(traces, use_container_width=True, hide_index=True)
    notes = cost_report.get("notes") or []
    if notes:
        with st.expander("Kostenhinweise", expanded=False):
            _show_items([str(note) for note in notes])


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
                    f"{_category_label(activity.category)} | {activity.duration_hours:g}h | "
                    f"{activity.cost:g} {itinerary.currency} | "
                    f"{'Indoor' if activity.indoor else 'Outdoor'}"
                )
                meta = _activity_meta(activity.description)
                if meta:
                    st.caption(" | ".join(meta))
                links = _activity_links(activity.description)
                if links:
                    with st.expander("Details und Links", expanded=False):
                        for label, url in links:
                            st.markdown(f"- [{label}]({url})")
            if day.notes:
                st.info(_clean_note_text(" ".join(day.notes)))
            st.write(f"Tagessumme: {day.total_cost:g} {itinerary.currency}")


def _category_label(category: str) -> str:
    labels = {
        "food": "Essen",
        "street_food": "Street Food",
        "nature": "Natur",
        "culture": "Kultur",
        "history": "Geschichte",
        "architecture": "Architektur",
        "photography": "Fotografie",
        "shopping": "Shopping",
        "sport": "Sport",
        "gaming": "Gaming",
        "anime": "Anime",
        "technology": "Technik",
        "nightlife": "Nightlife",
        "local spots": "Lokale Orte",
    }
    return labels.get(str(category).strip().lower(), str(category).replace("_", " ").title())


def _activity_meta(description: str) -> list[str]:
    if not description:
        return []
    fields = []
    address = _description_field(description, "Address")
    rating = _description_field(description, "Rating")
    reviews = _description_field(description, "Reviews")
    if address:
        fields.append(f"Adresse: {address}")
    if rating:
        fields.append(f"Bewertung: {rating}")
    if reviews:
        fields.append(f"Reviews: {reviews}")
    return fields


def _activity_links(description: str) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    website = _description_field(description, "Website")
    maps_url = _description_field(description, "Google Maps")
    if website:
        links.append(("Website", website))
    if maps_url:
        links.append(("Google Maps", maps_url))
    return links


def _description_field(description: str, label: str) -> str:
    marker = f"{label}:"
    for part in str(description).split("|"):
        cleaned = part.strip()
        if cleaned.lower().startswith(marker.lower()):
            return cleaned.split(":", 1)[1].strip()
    return ""


def _clean_note_text(text: str) -> str:
    replacements = {
        "Rain-aware planning": "Regenbewusste Planung",
        "Indoor activities were prioritized.": "Indoor-Aktivitaeten wurden priorisiert.",
        "Relaxed pacing: limited number of main activities.": "Entspanntes Tempo: begrenzte Anzahl grosser Aktivitaeten.",
    }
    result = str(text)
    for source, target in replacements.items():
        result = result.replace(source, target)
    return result


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
    memory_panel = st.empty()
    with memory_panel.container():
        _show_sidebar_memory(memory)

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

with st.expander("Optional: Gmail-Newsletter als Preference-Signale verbinden", expanded=False):
    st.caption(
        "Gmail wird lokal per OAuth verbunden. Es werden keine vollstaendigen Mail-Bodies gespeichert, "
        "sondern nur Newsletter-Metadaten, Snippets und schwache Interessenssignale."
    )

    creds_ok = gmail_credentials_available()
    st.write(f"Credentials vorhanden: {'Ja' if creds_ok else 'Nein'}")

    if not creds_ok:
        gmail_credentials_upload = st.file_uploader(
            "Google OAuth Client JSON hochladen",
            type=["json"],
            key="gmail_credentials_upload",
            help="Google Cloud OAuth Client fuer Desktop App mit Scope gmail.readonly.",
        )
        if gmail_credentials_upload and st.button("Gmail OAuth-Datei speichern"):
            try:
                save_gmail_credentials_file(gmail_credentials_upload.getvalue())
                st.success("Gmail OAuth-Credentials lokal gespeichert.")
                st.rerun()
            except GmailIntegrationError as exc:
                st.error(str(exc))

    gmail_col_1, gmail_col_2 = st.columns(2)
    with gmail_col_1:
        gmail_limit = st.number_input("Max. Mails", min_value=1, max_value=50, value=20, step=1)
    with gmail_col_2:
        gmail_lookback_days = st.number_input("Zeitraum in Tagen", min_value=1, max_value=3650, value=365, step=30)

    if st.button("Gmail verbinden / laden", disabled=not creds_ok, use_container_width=True):
        try:
            account_email = get_gmail_account_email(user_id)
            sources, messages = build_gmail_preference_source(
                user_id=user_id,
                max_messages=int(gmail_limit),
                lookback_days=int(gmail_lookback_days),
            )
            st.session_state["gmail_account_email"] = account_email
            st.session_state["gmail_preference_sources"] = sources
            st.session_state["gmail_newsletter_messages"] = messages
            if not messages:
                st.session_state["gmail_preference_sources"] = []
                st.session_state["gmail_newsletter_messages"] = []
                st.info("Keine passenden Gmail-Newsletter im gewaehlten Zeitraum gefunden.")
            else:
                kept_count = sum(1 for message in messages if message.keep_as_preference)
                st.success(f"{len(messages)} Gmail-Newsletter geprüft, {kept_count} als Preference-Signal übernommen.")
        except GmailIntegrationError as exc:
            st.error(str(exc))

    gmail_sources = st.session_state.get("gmail_preference_sources", [])
    gmail_messages = st.session_state.get("gmail_newsletter_messages", [])
    gmail_account_email = st.session_state.get("gmail_account_email", "")
    if gmail_account_email:
        st.write(f"Verbundenes Gmail-Konto: {gmail_account_email}")
    if gmail_messages:
        st.markdown("**Gefundene Newsletter-Mails**")
        kept_gmail_interests = sorted(
            {
                interest
                for message in gmail_messages
                if message.keep_as_preference
                for interest in message.inferred_interests
            }
        )
        kept_budget_signals = sorted(
            {
                message.budget_signal
                for message in gmail_messages
                if message.keep_as_preference and message.budget_signal and message.budget_signal != "unknown"
            }
        )
        kept_style_signals = sorted(
            {
                message.travel_style_signal
                for message in gmail_messages
                if message.keep_as_preference and message.travel_style_signal and message.travel_style_signal != "unknown"
            }
        )
        summary_cols = st.columns(3)
        summary_cols[0].write("**Aus Gmail erkannte Interessen:** " + (", ".join(kept_gmail_interests) or "Keine"))
        summary_cols[1].write("**Budget-Signale:** " + (", ".join(kept_budget_signals) or "Keine"))
        summary_cols[2].write("**Reisestil-Signale:** " + (", ".join(kept_style_signals) or "Keine"))
        st.dataframe(
            [
                {
                    "Sender": message.sender,
                    "Subject": message.subject,
                    "Date": message.date,
                    "Labels": ", ".join(message.labels),
                    "Relevanz": message.travel_relevance_score,
                    "Signal": message.signal_strength,
                    "Keep": "Ja" if message.keep_as_preference else "Nein",
                    "Interessen": ", ".join(message.inferred_interests),
                    "Grund/Zusammenfassung": message.preference_summary or message.ignore_reason,
                    "Snippet": message.snippet,
                }
                for message in gmail_messages
            ],
            use_container_width=True,
            hide_index=True,
        )
    elif gmail_account_email:
        st.info("Fuer dieses Gmail-Konto sind aktuell keine Newsletter-Signale in der Sitzung geladen.")
    if gmail_sources:
        st.markdown("**Gmail PreferenceSource fuer diesen Run**")
        for source in gmail_sources:
            st.write(f"- {source.name} ({source.source_type}, {len(source.text)} Zeichen)")
        with st.expander("Gmail-Signaltext anzeigen", expanded=False):
            st.text(gmail_sources[0].text)
    elif gmail_messages:
        st.info("Gmail-Mails wurden geladen, aber keine Mail war stark genug als Reisepräferenz-Signal.")
    remove_col_1, remove_col_2 = st.columns(2)
    with remove_col_1:
        remove_session = st.button(
            "Gmail-Import aus Sitzung entfernen",
            disabled=not (gmail_sources or gmail_messages or gmail_account_email),
            use_container_width=True,
        )
    with remove_col_2:
        remove_chroma = st.button("Gmail-Memory aus Chroma loeschen", use_container_width=True)
    if gmail_sources or gmail_messages or gmail_account_email:
        if remove_session:
            st.session_state["gmail_preference_sources"] = []
            st.session_state["gmail_newsletter_messages"] = []
            st.session_state["gmail_account_email"] = ""
            st.rerun()
    if remove_chroma:
        deleted = delete_user_memory_sources(user_id, source_type="email_newsletter")
        st.session_state["gmail_preference_sources"] = []
        st.session_state["gmail_newsletter_messages"] = []
        st.session_state["gmail_account_email"] = ""
        st.success(f"{deleted} gespeicherte Gmail-Memory-Chunk(s) aus Chroma geloescht.")
        st.rerun()

st.subheader("Reiseparameter")
request_text = st.text_area(
    "Freie Reiseanfrage",
    value="Ich will 4 Tage nach Barcelona, Budget 700 Euro, ich mag Food, Gaming, Anime und lokale Spots und will keinen stressigen Plan.",
)

generate = st.button("Reiseplan erstellen", type="primary")

if generate:
    try:
        preference_sources = _build_preference_sources(uploaded_files, travel_ratings, feedback)
        gmail_sources = st.session_state.get("gmail_preference_sources", [])
        preference_sources.extend(gmail_sources)
        if gmail_sources:
            st.success(f"{len(gmail_sources)} Gmail-Newsletter-Quelle(n) ins Preference Learning uebernommen.")
        parsed_request = parse_travel_request(
            request_text,
            TravelRequest(
                destination="",
                destination_scope="open",
                duration_days=3,
                budget=600,
                interests=interests,
                must_have=[],
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
                destination_scope=parsed_request.destination_scope,
                needs_destination_recommendation=parsed_request.needs_destination_recommendation,
                must_have=parsed_request.must_have,
            )
    except (MissingOpenAIKeyError, MissingLocalAIError) as exc:
        st.error(str(exc))
        st.stop()

    with memory_panel.container():
        _show_sidebar_memory(result.profile, title="Aktuelles Memory nach diesem Lauf", expanded=True)

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
    _show_agentic_tool_workflow(result.agentic_tool_workflow)
    _show_agentic_quality_review(result.agentic_quality_review)
    _show_cost_report(result.cost_report)

    itinerary_title = "Finaler Reiseplan" if result.validation.ok else "Planentwurf mit offenen Problemen"
    _show_itinerary(result.itinerary, itinerary_title)

    if not result.validation.ok:
        st.warning(
            "Der Plan ist noch nicht final, weil die finale Validierung offene Fehler enthaelt. "
            "Er wird deshalb nicht als fertiges Travel Package ausgegeben."
        )
    else:
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
