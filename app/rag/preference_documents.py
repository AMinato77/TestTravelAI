from __future__ import annotations

from app.models.preference_source import PreferenceSource
from app.rag.memory_retrieval import ingest_preference_sources, load_user_memory_sources


def save_preference_sources(user_id: str, sources: list[PreferenceSource]) -> None:
    """Persist preference sources in ChromaDB as embedded RAG chunks."""
    if not sources:
        return
    ingest_preference_sources(user_id, sources)


def load_preference_sources(user_id: str, limit: int = 50) -> list[PreferenceSource]:
    """Load stored preference-memory chunks from ChromaDB."""
    return load_user_memory_sources(user_id, limit=limit)
