from __future__ import annotations

import os

import requests


def tool_server_url() -> str:
    return os.getenv("TRAVEL_TOOL_SERVER_URL", "").rstrip("/")


def tool_server_enabled() -> bool:
    return bool(tool_server_url())


def post_tool(path: str, payload: dict, timeout: int = 60) -> dict:
    base_url = tool_server_url()
    if not base_url:
        raise RuntimeError("TRAVEL_TOOL_SERVER_URL is not configured.")
    response = requests.post(f"{base_url}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Tool server returned non-object JSON for {path}.")
    return data
