from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def main() -> None:
    destination = sys.argv[1] if len(sys.argv) > 1 else "Barcelona"
    queries = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["local food markets", "cultural attractions", "local neighborhoods"])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    from app.tools.places_tool import search_places

    activities = search_places(destination, queries, limit=limit)
    print(json.dumps([asdict(activity) for activity in activities], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
