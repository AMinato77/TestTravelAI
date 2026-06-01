from __future__ import annotations

from pathlib import Path


def get_chroma_client(path: str = "data/chromadb"):
    import chromadb

    Path(path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=path)


def get_or_create_collection(name: str, path: str = "data/chromadb"):
    client = get_chroma_client(path)
    return client.get_or_create_collection(name=name)

