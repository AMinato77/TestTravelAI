from __future__ import annotations

from dataclasses import dataclass, field


OPENAI_MODEL_PRICES_USD_PER_1M = {
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}

GOOGLE_PLACES_TEXT_SEARCH_ESTIMATE_USD = 0.035


@dataclass(slots=True)
class ToolTrace:
    name: str
    provider: str
    count: int = 1
    estimated_cost_usd: float = 0.0
    metadata: dict = field(default_factory=dict)


def estimate_tool_cost_report(tool_traces: list[ToolTrace | dict]) -> dict:
    normalized = [_normalize_trace(trace) for trace in tool_traces]
    total = round(sum(trace.estimated_cost_usd for trace in normalized), 5)
    by_provider: dict[str, float] = {}
    by_tool: dict[str, float] = {}
    for trace in normalized:
        by_provider[trace.provider] = by_provider.get(trace.provider, 0.0) + trace.estimated_cost_usd
        by_tool[trace.name] = by_tool.get(trace.name, 0.0) + trace.estimated_cost_usd
    return {
        "currency": "USD",
        "estimated_total_usd": total,
        "by_provider": {key: round(value, 5) for key, value in sorted(by_provider.items())},
        "by_tool": {key: round(value, 5) for key, value in sorted(by_tool.items())},
        "traces": [trace_to_dict(trace) for trace in normalized],
        "notes": [
            "Costs are estimates for comparison, not billing-grade accounting.",
            "Google Places and OpenAI model pricing can change; verify final numbers in provider dashboards.",
            "OpenAI traces include TravelAI generate_json calls when token usage is returned by the SDK.",
            "Agents SDK review/tool-workflow calls may not expose token usage in this local estimate.",
            "Cached Google Places responses are counted as zero-cost cache hits when metadata.cache_hit is true.",
        ],
    }


def google_places_trace(query_count: int, cache_hits: int = 0) -> ToolTrace:
    billable = max(0, query_count - cache_hits)
    return ToolTrace(
        name="google_places_text_search",
        provider="google",
        count=query_count,
        estimated_cost_usd=round(billable * GOOGLE_PLACES_TEXT_SEARCH_ESTIMATE_USD, 5),
        metadata={"billable_calls": billable, "cache_hits": cache_hits},
    )


def openai_llm_trace(name: str, model: str, input_tokens: int = 0, output_tokens: int = 0) -> ToolTrace:
    prices = OPENAI_MODEL_PRICES_USD_PER_1M.get(model, OPENAI_MODEL_PRICES_USD_PER_1M["gpt-5-nano"])
    cost = (input_tokens / 1_000_000 * prices["input"]) + (output_tokens / 1_000_000 * prices["output"])
    return ToolTrace(
        name=name,
        provider="openai",
        estimated_cost_usd=round(cost, 5),
        metadata={"model": model, "input_tokens": input_tokens, "output_tokens": output_tokens},
    )


def trace_to_dict(trace: ToolTrace) -> dict:
    return {
        "name": trace.name,
        "provider": trace.provider,
        "count": trace.count,
        "estimated_cost_usd": round(trace.estimated_cost_usd, 5),
        "metadata": trace.metadata,
    }


def _normalize_trace(trace: ToolTrace | dict) -> ToolTrace:
    if isinstance(trace, ToolTrace):
        return trace
    return ToolTrace(
        name=str(trace.get("name") or "unknown"),
        provider=str(trace.get("provider") or "unknown"),
        count=int(trace.get("count") or 1),
        estimated_cost_usd=float(trace.get("estimated_cost_usd") or 0),
        metadata=trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {},
    )
