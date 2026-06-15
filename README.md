# Adaptive AI Travel Agent

Streamlit-Projekt fuer einen personalisierten Reiseagenten mit GenAI, User Memory,
RAG, ChromaDB, externen APIs, eigenem Tool-Webserver und OpenAI Agents SDK.

## Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
streamlit run frontend/streamlit_app.py
```

Optionaler Tool-Webserver fuer die agentische Tool-Schicht:

```powershell
uvicorn app.tool_server:app --host 127.0.0.1 --port 8000
```

Wenn `TRAVEL_TOOL_SERVER_URL=http://127.0.0.1:8000` gesetzt ist, koennen
Agents-SDK-Function-Tools die Backend-Tools ueber HTTP aufrufen. Ohne diese
Variable nutzt die App dieselbe Tool-Logik direkt im Prozess.

Provider und API-Keys werden ueber `.env` geladen. Fuer kostenlose Entwicklung
kann `AI_PROVIDER=demo` verwendet werden. Fuer ein lokales kostenloses LLM kann
`AI_PROVIDER=ollama` gesetzt werden, wenn Ollama lokal laeuft. Fuer die finale
OpenAI-Version wird `AI_PROVIDER=openai` plus `OPENAI_API_KEY` verwendet.

Empfohlene OpenAI-Variablen:

```powershell
AI_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_REQUEST_MODEL=gpt-5-nano
OPENAI_PREFERENCE_MODEL=gpt-5-nano
OPENAI_PLANNING_MODEL=gpt-5-nano
OPENAI_TOOL_WORKFLOW_MODEL=gpt-5-nano
OPENAI_ACTIVITY_EVALUATION_MODEL=gpt-5-nano
OPENAI_QUALITY_REVIEW_MODEL=gpt-5-nano
OPENAI_ALLOW_DEMO_FALLBACK=false
TRAVEL_TOOL_SERVER_URL=http://127.0.0.1:8000
```

Kostenlose lokale Entwicklung:

```powershell
AI_PROVIDER=demo
```

Kostenloses lokales LLM mit Ollama:

```powershell
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

WeatherAPI.com Forecast:

```powershell
WEATHER_API_KEY=dein_weatherapi_key
```

Ohne `WEATHER_API_KEY` nutzt die App Fallback-Wetter, damit der Workflow weiter
testbar bleibt. Mit Key liefert das Weather Tool pro Reisetag `rain_chance`,
`is_rainy`, Temperatur und Wetterbeschreibung fuer Validation und Optimization.

Geoapify Places:

```powershell
GEOAPIFY_API_KEY=dein_geoapify_key
```

Quick test:

```powershell
.\.venv\Scripts\python.exe scripts\test_geoapify_places.py Barcelona "food,culture" 5
```

## Agentic Workflow

1. Nutzereingaben erfassen
2. Freitext-Reiseanfrage in strukturierte Felder parsen
3. Gespeichertes User Profile laden
4. Praeferenzen aus Chat-Exports, Notizen, Bewertungen, Feedback und Formular ableiten
5. User Profile aktualisieren und speichern
6. Google-Places-Queries durch den Query Planning Agent erstellen
7. Reiseaktivitaeten ueber Google Places abrufen
8. Internal Tool Workflow Agent prueft Kandidatenpool und Budgetziel
9. Activity Evaluation Agent filtert schwache oder widerspruechliche Kandidaten
10. Wetter abrufen
11. Tagesplanung mit Planning Agent generieren
12. Plan validieren
13. Plan iterativ optimieren und erneut validieren
14. Agents SDK Quality Review Agent bewertet Budget- und Validierungsqualitaet

## User Memory

User Memory wird im aktiven Workflow ueber ChromaDB und OpenAI Embeddings
gespeichert. Profile, Uploads, Reise-Notizen, Bewertungen, Feedback und
optionale Gmail-Newsletter-Signale werden als eingebettete Memory-Dokumente
in `data/chromadb` abgelegt.

Beim naechsten Plan wird eine semantische Query aus Ziel, Interessen,
Avoid-Praeferenzen und Reisestil gebaut. ChromaDB liefert relevante
Memory-Chunks zurueck, die in die Preference Extraction und Planung
einbezogen werden.

Wichtig: Fuer echtes RAG-Memory ist `OPENAI_API_KEY` erforderlich, weil die
Embeddings ueber `text-embedding-3-small` erzeugt werden.

Memory inspizieren:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_chroma_memory.py --user Paris_demo --limit 10
```

### Gmail Newsletter Memory

Die alte IMAP-Mail-Anbindung wurde entfernt. Gmail wird lokal ueber OAuth
mit dem Scope `gmail.readonly` verbunden. Dafuer wird ein Google OAuth
Client JSON fuer eine Desktop-App benoetigt.

Lokale Dateien:

- OAuth Credentials: `data/gmail_credentials.json`
- User Tokens: `data/gmail_tokens/<user_id>.json`

Die App speichert keine vollstaendigen E-Mail-Bodies. Aus Gmail werden nur
newsletterartige Mails anhand von Kategorien, Query-Terms und Headern wie
`List-ID` / `List-Unsubscribe` erkannt. An den Preference Agent geht eine
kompakte `PreferenceSource` mit `source_type="email_newsletter"`, die Sender,
Subject, Date, Labels, Snippet und schwache Interessenssignale enthaelt.

## Aktivitaetensuche

Die App normalisiert Reiseziele wie `Wien`/`Vienna` oder `Rom`/`Rome`, bevor
APIs und Memory aktualisiert werden. Google Places wird nicht mehr nur ueber
starre Keywords abgefragt: Im OpenAI-Modus erstellt ein Query-Planning-Agent
passende Google-Places-Textqueries aus Ziel, Interessen und Avoid-Praeferenzen.
Wenn der LLM-Aufruf fehlschlaegt oder Demo-Modus aktiv ist, nutzt die App den
deterministischen Template-Fallback.

## Tool Server und Agentic Workflow

Die App enthaelt einen FastAPI Tool Server (`app/tool_server.py`) mit
Werkzeug-Endpunkten fuer:

- `POST /tools/places/search`
- `POST /tools/weather`
- `POST /tools/memory/retrieve`
- `POST /tools/itinerary/validate`
- `POST /tools/itinerary/optimize`
- `POST /tools/budget/strategy`
- `POST /tools/cost/estimate`

Der OpenAI Agents SDK Tool Workflow Agent arbeitet ausschliesslich mit internen
Function Tools und bewertet, ob die von Google Places gelieferten
Kandidaten fuer Interessen, Avoid-Praeferenzen und Budgetziel ausreichen. Die
App plant nur mit echten Places-Kandidaten und erfindet keine Aktivitaeten.

## Budget Quality und Agents SDK

Das Budget wird nicht nur gegen Ueberschreitung validiert. Die App berechnet je
nach Reisestil und Budgetpraeferenz eine Zielspanne fuer sinnvolle
Budgetauslastung. Wenn ein Plan zu billig ist, erzeugt die Optimierung explizite
Experience-Budget-Upgrades fuer hochwertigere Restaurants, Tickets,
Reservierungen oder passende lokale Erlebnisse.

Zusaetzlich laeuft im OpenAI-Modus ein OpenAI Agents SDK Quality Review Agent.
Dieser Agent nutzt Function Tools fuer Budget- und Validation-Checks und gibt
eine strukturierte Qualitaetsbewertung fuer die UI zurueck.

## Cost Tracking

Pro Planung erzeugt die App einen Cost Report mit Tool-Traces, Provider-Summen
und geschaetzten Kosten fuer Google Places und OpenAI-Modellaufrufe. Die Werte sind
Vergleichs- und Demo-Werte, keine abrechnungsgenaue Provider-Rechnung. Fuer die
finale Bewertung muessen die tatsaechlichen Provider-Dashboards geprueft werden.
Der OpenAI Usage Dashboard bleibt massgeblich, weil dort auch interne
Agents-SDK-Modellrunden und Tokenmengen auftauchen, die die lokale Schaetzung
nicht exakt abrechnen kann.
