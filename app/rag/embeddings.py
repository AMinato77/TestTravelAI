from __future__ import annotations

import os

from openai import OpenAI


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings.")

    client = OpenAI()
    response = client.embeddings.create(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        input=texts,
    )
    return [item.embedding for item in response.data]

