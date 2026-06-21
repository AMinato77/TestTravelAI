# Adaptive AI Travel Agent

Streamlit-Projekt fuer einen personalisierten Reiseagenten mit GenAI, User
Memory, RAG ueber ChromaDB, Google Places, WeatherAPI, Gmail-OAuth-Signalen,
einem optionalen FastAPI Tool Server und OpenAI Agents SDK Komponenten.

## Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
streamlit run frontend/streamlit_app.py
```

Die App startet in `frontend/streamlit_app.py`. Der vollstaendige Workflow wird
in `app/orchestrator.py` ausgefuehrt.

## Konfiguration

Provider und API-Keys werden aus `.env` geladen. Die Vorlage liegt in
`.env.example`.

Minimal fuer einen echten Plan:

```powershell
AI_PROVIDER=demo
GOOGLE_PLACES_API_KEY=dein_google_places_key
OPENAI_API_KEY=sk-...
```

`AI_PROVIDER=demo` nutzt regelbasierte Fallbacks fuer LLM-Schritte, aber die
aktive Aktivitaetensuche verwendet weiterhin Google Places. ChromaDB-Memory
benoetigt ebenfalls `OPENAI_API_KEY`, weil Embeddings ueber
`text-embedding-3-small` erzeugt werden.

OpenAI-Modus:

```powershell
AI_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_REQUEST_MODEL=gpt-5-nano
OPENAI_PREFERENCE_MODEL=gpt-5-nano
OPENAI_PLANNING_MODEL=gpt-5-nano
OPENAI_EXPLANATION_MODEL=gpt-5-nano
OPENAI_ACTIVITY_EVALUATION_MODEL=gpt-5-nano
OPENAI_GMAIL_SIGNAL_MODEL=gpt-5-nano
OPENAI_TOOL_WORKFLOW_MODEL=gpt-5-nano
OPENAI_QUALITY_REVIEW_MODEL=gpt-5-nano
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_ALLOW_DEMO_FALLBACK=false
GOOGLE_PLACES_API_KEY=dein_google_places_key
```

Lokales LLM mit Ollama:

```powershell
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

WeatherAPI.com ist optional:

```powershell
WEATHER_API_KEY=dein_weatherapi_key
```

Ohne `WEATHER_API_KEY` nutzt die App Fallback-Wetter, damit der Workflow weiter
testbar bleibt. Mit Key liefert das Weather Tool pro Reisetag `rain_chance`,
`is_rainy`, Temperatur und Wetterbeschreibung fuer Validierung und Optimierung.

## Optionaler Tool Server

Der FastAPI Tool Server stellt interne Werkzeuge per HTTP bereit:

```powershell
uvicorn app.tool_server:app --host 127.0.0.1 --port 8000
```

Wenn `TRAVEL_TOOL_SERVER_URL=http://127.0.0.1:8000` gesetzt ist, koennen
Agents-SDK-Function-Tools die Backend-Tools ueber HTTP aufrufen. Ohne diese
Variable nutzt die App dieselbe Tool-Logik direkt im Prozess.

Endpunkte:

- `GET /health`
- `GET /tools/memory/profile/{user_id}`
- `POST /tools/places/search`
- `POST /tools/weather`
- `POST /tools/memory/retrieve`
- `POST /tools/itinerary/validate`
- `POST /tools/itinerary/optimize`
- `POST /tools/budget/strategy`
- `POST /tools/cost/estimate`

## Agentic Workflow

1. Nutzereingaben und optionale Preference-Quellen erfassen
2. Freitext-Reiseanfrage in strukturierte Felder parsen
3. Ziel normalisieren oder bei offener Anfrage durch den Destination Agent waehlen
4. Gespeichertes User Profile aus ChromaDB laden
5. Uploads, Bewertungen, Feedback und Gmail-Newsletter-Signale als Memory speichern
6. Relevante Memory-Chunks per ChromaDB abrufen
7. Preference Agent aktualisiert das User Profile
8. Google-Places-Queries aus Interessen und Avoid-Praeferenzen erstellen
9. Aktivitaetskandidaten ueber Google Places abrufen, filtern und diversifizieren
10. Internal Tool Workflow Agent prueft Kandidatenpool und Budgetziel
11. Activity Evaluation Agent entfernt schwache oder widerspruechliche Kandidaten
12. Wetter abrufen
13. Planning Agent erstellt den ersten Tagesplan
14. Harte Regeln validieren Budget, Avoids, Regen, Duplikate und Zielort
15. Optimization Agent verbessert den Plan iterativ
16. Agents SDK Quality Review bewertet Budget- und Validierungsqualitaet
17. Explanation Agent erzeugt die UI-Erklaerung und Cost Tracking fasst Toolkosten zusammen

## User Memory und RAG

User Memory wird im aktiven Workflow ueber ChromaDB gespeichert. Profile,
Uploads, Reise-Notizen, Bewertungen, Feedback und optionale
Gmail-Newsletter-Signale werden als eingebettete Memory-Dokumente in
`data/chromadb` abgelegt.

Beim naechsten Plan wird eine semantische Query aus Ziel, Interessen,
Avoid-Praeferenzen und Reisestil gebaut. ChromaDB liefert relevante
Memory-Chunks zurueck, die in Preference Extraction und Planung einbezogen
werden.

Wichtig: Fuer echtes RAG-Memory ist `OPENAI_API_KEY` erforderlich, weil
`app/rag/embeddings.py` OpenAI Embeddings verwendet.

Memory inspizieren:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_chroma_memory.py --user Paris_demo --limit 10
```

## Gmail Newsletter Memory

Gmail wird lokal ueber OAuth mit dem Scope `gmail.readonly` verbunden. Dafuer
wird ein Google OAuth Client JSON fuer eine Desktop-App benoetigt.

Lokale Dateien:

- OAuth Credentials: `data/gmail_credentials.json`
- User Tokens: `data/gmail_tokens/<user_id>.json`

Die App speichert keine vollstaendigen E-Mail-Bodies. Aus Gmail werden nur
newsletterartige Mails anhand von Kategorien, Query-Terms und Headern wie
`List-ID` / `List-Unsubscribe` erkannt. An den Preference Agent geht eine
kompakte `PreferenceSource` mit `source_type="email_newsletter"`, die Sender,
Subject, Date, Labels, Snippet und schwache Interessenssignale enthaelt.

## Aktivitaetensuche

Die aktive Aktivitaetensuche laeuft ueber Google Places Text Search in
`app/tools/places_tool.py`.

Die App normalisiert Reiseziele wie `Wien`/`Vienna` oder `Rom`/`Rome`, bevor
APIs und Memory aktualisiert werden. Im OpenAI-Modus erstellt ein Query Planning
Agent passende Google-Places-Textqueries aus Ziel, Interessen und
Avoid-Praeferenzen. Wenn der LLM-Aufruf fehlschlaegt oder Demo-Modus aktiv ist,
nutzt die App deterministische Template-Queries.

Die Places-Ergebnisse werden in `data/api_cache/google_places` gecacht,
dedupliziert, grob nach Qualitaet sortiert, gegen falsche Zielorte geprueft und
in das interne `Activity`-Modell umgewandelt. Es gibt keinen aktiven lokalen
Aktivitaets-Fallback fuer die Planung.

## Validierung und Optimierung

Deterministische Python-Regeln pruefen harte Constraints:

- Budgetueberschreitung und zu geringe Budgetauslastung
- leere Tage
- zu volle Tage oder zu viele Aktivitaeten fuer `relaxed`
- Avoid-Konflikte wie Food, Museen oder Nightlife
- Duplikate innerhalb eines Tages oder ueber mehrere Tage
- Outdoor-Aktivitaeten bei Regen
- Aktivitaeten, deren Adresse nicht zum Ziel passt

Der Optimizer versucht offene Probleme mit verfuegbaren Alternativen zu
reparieren. Zusaetzlich kann die Budgetstrategie Experience-Upgrades einfuegen,
wenn ein Plan deutlich unter der sinnvollen Budgetzielspanne liegt.

## Cost Tracking

Pro Planung erzeugt die App einen Cost Report mit Tool-Traces, Provider-Summen
und geschaetzten Kosten fuer Google Places und OpenAI-Modellaufrufe. Die Werte
sind Vergleichs- und Demo-Werte, keine abrechnungsgenaue Provider-Rechnung.
Massgeblich bleiben die Provider-Dashboards.

## Lokale Daten und Git

Lokale Runtime-Daten sind in `.gitignore` ausgeschlossen:

- `.env`
- `.venv/`
- `data/chromadb/`
- `data/api_cache/`
- `data/gmail_credentials.json`
- `data/gmail_tokens/`
- `data/user_profiles/*.json`
- `data/user_documents/`
- `data/travel_data/*.json`

Hinweis: Falls Dateien bereits vor dem Eintrag in `.gitignore` getrackt wurden,
muessen sie separat aus dem Git-Index entfernt werden.
