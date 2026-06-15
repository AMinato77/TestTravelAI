from __future__ import annotations

import json
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)

_OPENAI_USAGE_RECORDS: list[dict] = []


class MissingOpenAIKeyError(RuntimeError):
    pass


class MissingLocalAIError(RuntimeError):
    pass


def ai_provider() -> str:
    return os.getenv("AI_PROVIDER", "demo").lower()


def demo_fallback_enabled() -> bool:
    return ai_provider() == "demo" or os.getenv("OPENAI_ALLOW_DEMO_FALLBACK", "false").lower() == "true"


def get_openai_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise MissingOpenAIKeyError(
            "OPENAI_API_KEY is required when AI_PROVIDER=openai. "
            "Use AI_PROVIDER=demo for free fallback mode or AI_PROVIDER=ollama for a local free model."
        )
    return OpenAI()


def openai_model(env_name: str, default: str = "gpt-5-nano") -> str:
    return os.getenv(env_name, default)


def reset_openai_usage_records() -> None:
    _OPENAI_USAGE_RECORDS.clear()


def openai_usage_records() -> list[dict]:
    return list(_OPENAI_USAGE_RECORDS)


def generate_json(system_prompt: str, payload: dict, model_env: str) -> dict:
    provider = ai_provider()
    if provider == "demo":
        raise MissingLocalAIError("Demo provider does not generate LLM JSON.")
    if provider == "ollama":
        return _generate_ollama_json(system_prompt, payload)
    if provider == "openai":
        return _generate_openai_json(system_prompt, payload, model_env)

    raise MissingLocalAIError(f"Unknown AI_PROVIDER={provider}. Use demo, ollama, or openai.")


def _generate_openai_json(system_prompt: str, payload: dict, model_env: str) -> dict:
    client = get_openai_client()
    model = openai_model(model_env)
    messages = [
        {
            "role": "system",
            "content": (
                f"{system_prompt}\n\n"
                "Return only one valid JSON object. "
                "Do not include markdown, code fences, explanations, or extra text."
            ),
        },
        {
            "role": "user",
            "content": (
                "Input JSON:\n"
                f"{json.dumps(payload, ensure_ascii=True)}"
            ),
        },
    ]

    if hasattr(client, "responses"):
        response = client.responses.create(model=model, input=messages)
        _record_openai_usage(model_env, model, response)
        return _loads_json_object(response.output_text)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    _record_openai_usage(model_env, model, response)
    return _loads_json_object(response.choices[0].message.content or "{}")


def _generate_ollama_json(system_prompt: str, payload: dict) -> dict:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    prompt = (
        f"{system_prompt}\n\n"
        "Return only valid JSON. Do not include markdown.\n\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=True)}"
    )
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise MissingLocalAIError(
            "AI_PROVIDER=ollama requires a running local Ollama server. "
            "Install Ollama, run `ollama pull llama3.2:3b`, then start the app again. "
            "Use AI_PROVIDER=demo if you want no local model."
        ) from exc

    return _loads_json_object(response.json().get("response", "{}"))


def _loads_json_object(text: str) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        object_text = _extract_first_json_object(text)
        if not object_text:
            raise
        try:
            data = json.loads(object_text)
        except json.JSONDecodeError:
            data = json.loads(object_text, strict=False)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from the AI provider.")
    return data


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _record_openai_usage(name: str, model: str, response) -> None:
    usage = getattr(response, "usage", None)
    input_tokens = _usage_value(usage, "input_tokens", "prompt_tokens")
    output_tokens = _usage_value(usage, "output_tokens", "completion_tokens")
    _OPENAI_USAGE_RECORDS.append(
        {
            "name": name.lower(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    )


def _usage_value(usage, *names: str) -> int:
    if usage is None:
        return 0
    for name in names:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0
