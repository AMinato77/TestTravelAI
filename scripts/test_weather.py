from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from app.tools.weather_tool import get_weather


def main() -> None:
    city = sys.argv[1] if len(sys.argv) > 1 else "Barcelona"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    weather = get_weather(city, days=days)
    print(json.dumps(weather, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

