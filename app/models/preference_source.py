from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PreferenceSource:
    source_type: str
    name: str
    text: str

