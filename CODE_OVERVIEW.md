# Adaptive AI Travel Agent - Code Overview

This file explains the current prototype in simple terms.

## Main Workflow

The app starts in:

`frontend/streamlit_app.py`

Streamlit collects:

- free travel request
- user id
- manual interests
- uploaded notes or ratings
- optional emails via IMAP
- optional feedback

Then it calls:

`app/orchestrator.py`

The orchestrator runs the complete workflow step by step.

## Step 1: Request Extraction

File:

`app/agents/request_agent.py`

Purpose:

GPT turns free text into structured data.

Example:

```json
{
  "destination": "Hamburg",
  "duration_days": 4,
  "budget": 700,
  "interests": ["culture", "local spots"],
  "avoid": ["food", "restaurants", "cafes"],
  "travel_style": "relaxed"
}
```

## Step 2: User Memory

Files:

`app/rag/user_memory.py`
`app/rag/preference_documents.py`
`app/rag/memory_retrieval.py`
`app/tools/email_tool.py`

Purpose:

- save the user profile as an embedded ChromaDB profile snapshot
- save uploaded texts as embedded ChromaDB memory chunks
- fetch optional emails from an IMAP mailbox
- split uploaded texts into chunks
- create embeddings
- store chunks in ChromaDB
- retrieve relevant memories for the current trip

This is the real RAG part of the project.

Legacy JSON/JSONL memory files may still exist under `data/`, but the active
workflow now reads and writes user memory through ChromaDB.

## Step 3: Preference Agent

File:

`app/agents/preference_agent.py`

Purpose:

GPT reads manual interests, uploads, feedback, and retrieved memory chunks.
It creates an updated user profile.

## Step 4: Places API

File:

`app/tools/places_tool.py`

Purpose:

Google Places returns real place data.
The code converts Google place objects into our simple `Activity` model.

Important:

Google Places does not return perfect travel plans. It returns raw places.
We still need filtering, ranking, and AI evaluation.

## Step 5: Activity Ranking

File:

`app/services/activity_ranker.py`

Purpose:

Simple Python rules:

- normalize Geoapify categories
- normalize Google Places categories
- estimate cost
- estimate duration
- score activities
- reduce too many activities from the same category

## Step 6: No Local Activity Fallback

File:

`app/rag/retrieval.py`

Purpose:

This file only exists as a compatibility stub for older imports.

Important:

The app no longer uses local demo activities for planning.
Real activities come from Google Places.
Real ChromaDB RAG is in `memory_retrieval.py` and is used for user memory.

## Step 7: Activity Evaluation Agent

File:

`app/agents/activity_evaluation_agent.py`

Purpose:

GPT checks if candidate activities really fit the user.

Example:

- A gaming cafe should not be treated as a food activity unless gaming is requested.
- A nightlife place should not be used if nightlife was not requested.

## Step 8: Weather API

File:

`app/tools/weather_tool.py`

Purpose:

WeatherAPI returns forecast data.
The app uses rain chance and temperature for validation and planning.

## Step 9: Planning Agent

File:

`app/agents/planning_agent.py`

Purpose:

GPT creates the first itinerary from:

- destination
- days
- budget
- activities
- weather
- user profile

## Step 10: Validation and Optimization

Files:

`app/services/itinerary_validator.py`
`app/services/itinerary_optimizer.py`

Purpose:

Python checks hard constraints:

- budget exceeded
- outdoor activity on rainy day
- too many activities for relaxed travel style
- activity conflicts with avoid preferences

Then the optimizer tries to fix problems.

Why Python rules?

Hard constraints should be deterministic and easy to explain.
GPT can explain and plan, but Python should enforce clear rules.

## Step 11: Explanation Agent

File:

`app/agents/explanation_agent.py`

Purpose:

GPT explains why the final plan fits the user and which data sources were used.

## What Will Be Replaced Later

Current prototype:

- Python modules named agents
- direct function calls in `orchestrator.py`
- Google Places for real activities
- ChromaDB RAG for user memory

Later final architecture:

- OpenAI Agents SDK
- real function tools
- agents choose tools automatically
- better activity quality filtering
- routing API
- stronger ChromaDB memory retrieval

The core logic is already separated so these parts can be replaced step by step.
