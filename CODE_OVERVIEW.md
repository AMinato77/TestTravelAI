# Adaptive AI Travel Agent - Code Overview

This file explains the current codebase in simple terms.

## Entry Points

The app starts in:

`frontend/streamlit_app.py`

Streamlit collects:

- free travel request
- user id
- uploaded notes or ratings
- optional Gmail newsletter signals
- optional feedback

Then it calls:

`app/orchestrator.py`

The orchestrator runs the complete workflow step by step.

## Step 1: Request Extraction

File:

`app/agents/request_agent.py`

Purpose:

The request agent turns free text into structured data.

Example:

```json
{
  "destination": "Hamburg",
  "destination_scope": "city",
  "needs_destination_recommendation": false,
  "duration_days": 4,
  "budget": 700,
  "interests": ["culture", "local spots"],
  "must_have": [],
  "avoid": ["food", "restaurants", "cafes"],
  "travel_style": "relaxed"
}
```

## Step 2: Destination Resolution

File:

`app/agents/destination_agent.py`

Purpose:

If the user asks for a broad destination recommendation instead of a specific
city, the destination agent chooses a concrete destination. After that,
`app/services/destination_normalizer.py` normalizes common names such as
`Wien` to `Vienna` and `Rom` to `Rome`.

## Step 3: User Memory

Files:

`app/rag/user_memory.py`
`app/rag/preference_documents.py`
`app/rag/memory_retrieval.py`
`app/tools/gmail_tool.py`

Purpose:

- save the user profile as an embedded ChromaDB profile snapshot
- save uploaded texts as embedded ChromaDB memory chunks
- fetch optional Gmail newsletter signals via local OAuth
- split uploaded texts into chunks
- create OpenAI embeddings
- store chunks in ChromaDB
- retrieve relevant memories for the current trip

This is the active RAG part of the project. It requires `OPENAI_API_KEY` for
embeddings.

## Step 4: Preference Agent

File:

`app/agents/preference_agent.py`

Purpose:

The preference agent reads manual interests, uploads, feedback, Gmail signals
and retrieved memory chunks. It creates an updated `UserProfile`.

## Step 5: Places API

File:

`app/tools/places_tool.py`

Purpose:

Google Places Text Search returns real place data. The code converts Google
place objects into the internal `Activity` model, filters weak results,
deduplicates candidates, estimates cost and duration, and checks that addresses
match the requested destination.

Important:

There is no active local activity fallback for planning. If Google Places is not
configured or returns no useful candidates, the issue should remain visible.

## Step 6: Tool Workflow Agent

File:

`app/agents/agentic_tool_agent.py`

Purpose:

The OpenAI Agents SDK tool workflow checks whether the candidate pool covers the
requested interests and whether the budget strategy looks reasonable. If the SDK
or OpenAI mode is unavailable, the code returns deterministic fallback guidance.

## Step 7: Activity Evaluation Agent

File:

`app/agents/activity_evaluation_agent.py`

Purpose:

The activity evaluation agent checks if candidate activities really fit the
user. It can remove weak matches while preserving minimum interest coverage.

Example:

- A gaming cafe should not be treated as a food activity unless gaming is requested.
- A nightlife place should not be used if nightlife was not requested.

## Step 8: Weather API

File:

`app/tools/weather_tool.py`

Purpose:

WeatherAPI returns forecast data. Without `WEATHER_API_KEY`, the tool returns a
fallback forecast so the workflow can still run. The planner and validator use
rain chance and temperature.

## Step 9: Planning Agent

File:

`app/agents/planning_agent.py`

Purpose:

The planning agent creates the first itinerary from:

- destination
- days
- budget and target budget range
- activities
- weather
- user profile
- avoid constraints

In demo mode it uses `app/services/itinerary_builder.py` instead of an LLM.

## Step 10: Validation and Optimization

Files:

`app/services/itinerary_validator.py`
`app/services/itinerary_optimizer.py`
`app/tools/validation_tool.py`
`app/tools/optimization_tool.py`

Purpose:

Python checks hard constraints:

- budget exceeded
- budget underused
- empty day
- outdoor activity on rainy day
- too many activities for relaxed travel style
- activity conflicts with avoid preferences
- duplicate activities
- activity appears outside the destination

Then the optimizer tries to fix problems with available alternatives and budget
upgrades.

## Step 11: Quality Review and Explanation

Files:

`app/agents/agentic_quality_agent.py`
`app/agents/explanation_agent.py`

Purpose:

The quality review agent assesses budget and validation quality. The explanation
agent explains why the final plan fits the user and which data sources were
used.

## Step 12: Tool Server

File:

`app/tool_server.py`

Purpose:

The optional FastAPI server exposes internal tools over HTTP:

- `GET /health`
- `GET /tools/memory/profile/{user_id}`
- `POST /tools/places/search`
- `POST /tools/weather`
- `POST /tools/memory/retrieve`
- `POST /tools/itinerary/validate`
- `POST /tools/itinerary/optimize`
- `POST /tools/budget/strategy`
- `POST /tools/cost/estimate`

Without `TRAVEL_TOOL_SERVER_URL`, the app uses the same Python tool logic
directly in process.
