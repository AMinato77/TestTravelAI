from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.memory_retrieval import COLLECTION_NAME
from app.rag.chroma_db import get_or_create_collection


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ChromaDB user memory documents.")
    parser.add_argument("--user", help="Filter by user_id.")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    collection = get_or_create_collection(COLLECTION_NAME)
    kwargs = {"include": ["documents", "metadatas"], "limit": args.limit}
    if args.user:
        kwargs["where"] = {"user_id": args.user}
    result = collection.get(**kwargs)

    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    for index, item_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
        document = documents[index] if index < len(documents) else ""
        print("=" * 100)
        print(f"id: {item_id}")
        print("metadata:")
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        print("document:")
        print(document)


if __name__ == "__main__":
    main()
