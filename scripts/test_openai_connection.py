from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=True)

from app.agents.request_agent import parse_travel_request
from app.models.travel_request import TravelRequest
from app.tools.openai_runtime import ai_provider


def main() -> None:
    request = parse_travel_request(
        "Ich will 4 Tage nach Barcelona, Budget 700 Euro, ich mag Food, Gaming, Anime und lokale Spots und will keinen stressigen Plan.",
        TravelRequest(
            destination="Barcelona",
            duration_days=3,
            budget=600,
            interests=["food", "culture"],
            travel_style="balanced",
        ),
    )
    print(f"AI_PROVIDER={ai_provider()}")
    print(
        json.dumps(
            {
                "destination": request.destination,
                "duration_days": request.duration_days,
                "budget": request.budget,
                "interests": request.interests,
                "travel_style": request.travel_style,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
