from __future__ import annotations

import hashlib
import time
import re
from dataclasses import dataclass

from app.models.preference_source import PreferenceSource
from app.rag.chroma_db import get_or_create_collection
from app.rag.embeddings import embed_texts


COLLECTION_NAME = "user_preference_memory"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 160


@dataclass(slots=True)
class RetrievedMemory:
    source: PreferenceSource
    distance: float | None = None


def ingest_preference_sources(user_id: str, sources: list[PreferenceSource]) -> int:
    """Store uploaded preference sources as embedded chunks in ChromaDB."""
    chunks: list[str] = []
    ids: list[str] = []
    metadatas: list[dict] = []

    safe_user_id = _safe_user_id(user_id)
    now = time.time()
    for source in sources:
        for chunk_index, chunk in enumerate(_chunk_text(source.text)):
            chunk_id = _chunk_id(safe_user_id, source, chunk_index, chunk)
            ids.append(chunk_id)
            chunks.append(chunk)
            metadatas.append(
                {
                    "user_id": safe_user_id,
                    "memory_kind": "preference_source",
                    "source_type": source.source_type,
                    "source_name": source.name,
                    "chunk_index": chunk_index,
                    "created_at": now,
                }
            )

    if not chunks:
        return 0

    collection = get_or_create_collection(COLLECTION_NAME)
    embeddings = embed_texts(chunks)
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(chunks)


def retrieve_user_memory(
    user_id: str,
    query: str,
    limit: int = 6,
) -> list[RetrievedMemory]:
    """Retrieve semantically relevant user memory chunks for the current trip."""
    if not query.strip():
        return []

    collection = get_or_create_collection(COLLECTION_NAME)
    query_embedding = embed_texts([query])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        where={"user_id": _safe_user_id(user_id)},
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    retrieved: list[RetrievedMemory] = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        retrieved.append(
            RetrievedMemory(
                source=PreferenceSource(
                    source_type=str(metadata.get("source_type") or "rag_memory"),
                    name=str(metadata.get("source_name") or "memory_chunk"),
                    text=str(document or ""),
                ),
                distance=distance,
            )
        )
    return retrieved


def load_user_memory_sources(user_id: str, limit: int = 50) -> list[PreferenceSource]:
    """Load recent stored memory chunks for profile extraction context."""
    collection = get_or_create_collection(COLLECTION_NAME)
    result = collection.get(
        where={"user_id": _safe_user_id(user_id)},
        include=["documents", "metadatas"],
    )
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    rows: list[tuple[float, PreferenceSource]] = []
    seen: set[tuple[str, str]] = set()
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        if metadata.get("memory_kind") != "preference_source":
            continue
        name = str(metadata.get("source_name") or "memory_chunk")
        source_type = str(metadata.get("source_type") or "memory")
        key = (name, str(document or ""))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            (
                float(metadata.get("created_at") or 0),
                PreferenceSource(source_type=source_type, name=name, text=str(document or "")),
            )
        )

    rows.sort(key=lambda row: row[0], reverse=True)
    return [source for _, source in rows[:limit]]


def delete_user_memory_sources(
    user_id: str,
    source_type: str | None = None,
    source_name: str | None = None,
) -> int:
    """Delete stored preference-source chunks for a user and return deleted row count."""
    collection = get_or_create_collection(COLLECTION_NAME)
    result = collection.get(
        where={"user_id": _safe_user_id(user_id)},
        include=["metadatas"],
    )
    ids = result.get("ids") or []
    metadatas = result.get("metadatas") or []
    delete_ids: list[str] = []
    for index, item_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
        if metadata.get("memory_kind") != "preference_source":
            continue
        if source_type and metadata.get("source_type") != source_type:
            continue
        if source_name and metadata.get("source_name") != source_name:
            continue
        delete_ids.append(str(item_id))
    if delete_ids:
        collection.delete(ids=delete_ids)
    return len(delete_ids)


def list_memory_user_ids() -> list[str]:
    """Return user ids known to the Chroma memory collection."""
    collection = get_or_create_collection(COLLECTION_NAME)
    result = collection.get(include=["metadatas"])
    metadatas = result.get("metadatas") or []
    user_ids = {
        str(metadata.get("user_id"))
        for metadata in metadatas
        if metadata.get("user_id")
    }
    return sorted(user_ids)


def build_memory_query(
    destination: str,
    query_terms: list[str],
    avoid: list[str],
    travel_style: str,
) -> str:
    return (
        f"Travel destination: {destination}. "
        f"Concrete user wishes and query terms: {', '.join(query_terms) or 'unknown'}. "
        f"User avoid preferences: {', '.join(avoid) or 'none'}. "
        f"Travel style: {travel_style}. "
        "Find relevant past preferences, travel ratings, notes, dislikes, and planning constraints."
    )


def _chunk_text(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    if len(cleaned) <= CHUNK_SIZE:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + CHUNK_SIZE, len(cleaned))
        chunks.append(cleaned[start:end].strip())
        if end == len(cleaned):
            break
        start = max(0, end - CHUNK_OVERLAP)
    return [chunk for chunk in chunks if chunk]


def _chunk_id(user_id: str, source: PreferenceSource, chunk_index: int, chunk: str) -> str:
    digest = hashlib.sha1(
        f"{user_id}|{source.source_type}|{source.name}|{chunk_index}|{chunk}".encode("utf-8")
    ).hexdigest()
    return f"{user_id}_{digest}"


def _safe_user_id(user_id: str) -> str:
    return "".join(char for char in user_id if char.isalnum() or char in ("-", "_")).strip() or "demo_user_1"
