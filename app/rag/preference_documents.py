from __future__ import annotations

import json
from pathlib import Path

from app.models.preference_source import PreferenceSource


DOCUMENT_DIR = Path("data/user_documents")


def save_preference_sources(user_id: str, sources: list[PreferenceSource]) -> None:
    if not sources:
        return

    user_dir = DOCUMENT_DIR / _safe_user_id(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    index_path = user_dir / "index.jsonl"
    with index_path.open("a", encoding="utf-8") as file:
        for source in sources:
            file.write(
                json.dumps(
                    {
                        "source_type": source.source_type,
                        "name": source.name,
                        "text": source.text,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )


def load_preference_sources(user_id: str, limit: int = 50) -> list[PreferenceSource]:
    index_path = DOCUMENT_DIR / _safe_user_id(user_id) / "index.jsonl"
    if not index_path.exists():
        return []

    sources: list[PreferenceSource] = []
    with index_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            data = json.loads(line)
            sources.append(
                PreferenceSource(
                    source_type=data.get("source_type", "unknown"),
                    name=data.get("name", "uploaded_source"),
                    text=data.get("text", ""),
                )
            )
    return sources[-limit:]


def _safe_user_id(user_id: str) -> str:
    return "".join(char for char in user_id if char.isalnum() or char in ("-", "_")).strip() or "demo_user_1"

