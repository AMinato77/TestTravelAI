# Adaptive AI Travel Agent

Streamlit-Projekt fuer einen personalisierten Reiseagenten mit GenAI, User Memory,
RAG, ChromaDB, externen APIs und spaeter OpenAI Agents SDK.

## Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
streamlit run frontend/streamlit_app.py
```

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
OPENAI_ALLOW_DEMO_FALLBACK=false
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

## MVP Workflow

1. Nutzereingaben erfassen
2. Freitext-Reiseanfrage in strukturierte Felder parsen
3. Gespeichertes User Profile laden
4. Praeferenzen aus Chat-Exports, Notizen, Bewertungen, Feedback und Formular ableiten
5. User Profile aktualisieren und speichern
6. Reiseaktivitaeten abrufen oder aus Fallback-Daten laden
7. Wetter abrufen oder Fallback verwenden
8. Reiseplan mit dem Planning Agent generieren
9. Plan validieren
10. Plan optimieren

## User Memory

User Memory wird im aktiven Workflow ueber ChromaDB und OpenAI Embeddings
gespeichert. Profile, Uploads, Reise-Notizen, Bewertungen, Feedback und
optionale E-Mail-Quellen werden als eingebettete Memory-Dokumente in
`data/chromadb` abgelegt.

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

## Aktivitaetensuche

Die App normalisiert Reiseziele wie `Wien`/`Vienna` oder `Rom`/`Rome`, bevor
APIs und Memory aktualisiert werden. Google Places wird nicht mehr nur ueber
starre Keywords abgefragt: Im OpenAI-Modus erstellt ein Query-Planning-Agent
passende Google-Places-Textqueries aus Ziel, Interessen und Avoid-Praeferenzen.
Wenn der LLM-Aufruf fehlschlaegt oder Demo-Modus aktiv ist, nutzt die App den
deterministischen Template-Fallback.
