from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.models.preference_source import PreferenceSource
from app.models.user_profile import UserProfile
from app.rag.memory_retrieval import retrieve_user_memory
from app.rag.user_memory import load_user_profile
from app.services.budget_strategy import target_budget_range
from app.services.cost_tracker import estimate_tool_cost_report
from app.services.serialization import activity_from_dict, activity_to_dict, itinerary_from_dict, itinerary_to_dict, validation_to_dict
from app.tools.optimization_tool import optimize_itinerary
from app.tools.places_tool import search_places
from app.tools.validation_tool import validate_itinerary
from app.tools.weather_tool import get_weather


app = FastAPI(title="TravelAI Tool Server", version="1.0.0")


class PlacesSearchRequest(BaseModel):
    destination: str
    interests: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    limit: int = 20


class WeatherRequest(BaseModel):
    destination: str
    days: int = 3


class MemoryRetrieveRequest(BaseModel):
    user_id: str
    query: str
    limit: int = 6


class ValidateRequest(BaseModel):
    itinerary: dict
    budget: float
    weather: dict = Field(default_factory=dict)
    profile: dict = Field(default_factory=dict)
    constraints: dict = Field(default_factory=dict)


class OptimizeRequest(BaseModel):
    itinerary: dict
    alternatives: list[dict] = Field(default_factory=list)
    budget: float
    weather: dict = Field(default_factory=dict)
    profile: dict = Field(default_factory=dict)
    constraints: dict = Field(default_factory=dict)


class BudgetStrategyRequest(BaseModel):
    budget: float
    planned_cost: float = 0
    profile: dict = Field(default_factory=dict)


class CostRequest(BaseModel):
    traces: list[dict] = Field(default_factory=list)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "travelai-tool-server"}


@app.post("/tools/places/search")
def places_search(request: PlacesSearchRequest) -> dict:
    activities = search_places(
        destination=request.destination,
        interests=request.interests,
        avoid=request.avoid,
        limit=request.limit,
    )
    return {"activities": [activity_to_dict(activity) for activity in activities]}


@app.post("/tools/weather")
def weather(request: WeatherRequest) -> dict:
    return get_weather(request.destination, request.days)


@app.post("/tools/memory/retrieve")
def memory_retrieve(request: MemoryRetrieveRequest) -> dict:
    memories = retrieve_user_memory(request.user_id, request.query, limit=request.limit)
    return {
        "memories": [
            {
                "source_type": memory.source.source_type,
                "name": memory.source.name,
                "text": memory.source.text,
                "distance": memory.distance,
            }
            for memory in memories
        ]
    }


@app.get("/tools/memory/profile/{user_id}")
def memory_profile(user_id: str) -> dict:
    return load_user_profile(user_id).to_dict()


@app.post("/tools/itinerary/validate")
def itinerary_validate(request: ValidateRequest) -> dict:
    validation = validate_itinerary(
        itinerary=itinerary_from_dict(request.itinerary),
        budget=request.budget,
        weather=request.weather,
        profile=UserProfile.from_dict(request.profile) if request.profile else None,
        constraints=request.constraints,
    )
    return validation_to_dict(validation)


@app.post("/tools/itinerary/optimize")
def itinerary_optimize(request: OptimizeRequest) -> dict:
    itinerary = optimize_itinerary(
        itinerary=itinerary_from_dict(request.itinerary),
        alternatives=[activity_from_dict(activity) for activity in request.alternatives],
        budget=request.budget,
        weather=request.weather,
        profile=UserProfile.from_dict(request.profile) if request.profile else None,
        constraints=request.constraints,
    )
    return itinerary_to_dict(itinerary)


@app.post("/tools/budget/strategy")
def budget_strategy(request: BudgetStrategyRequest) -> dict:
    profile = UserProfile.from_dict(request.profile) if request.profile else None
    target_min, target_max = target_budget_range(request.budget, profile)
    utilization = request.planned_cost / request.budget if request.budget else 0
    return {
        "budget": request.budget,
        "planned_cost": request.planned_cost,
        "utilization": round(utilization, 3),
        "target_min": round(target_min, 2),
        "target_max": round(target_max, 2),
        "currency": "EUR",
    }


@app.post("/tools/cost/estimate")
def cost_estimate(request: CostRequest) -> dict:
    return estimate_tool_cost_report(request.traces)


def preference_source_from_tool_memory(data: dict) -> PreferenceSource:
    return PreferenceSource(
        source_type=str(data.get("source_type") or "rag_memory"),
        name=str(data.get("name") or "memory_chunk"),
        text=str(data.get("text") or ""),
    )
