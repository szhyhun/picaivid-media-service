"""Room-label compatibility helpers used by the MASt3R graph scorer."""
from __future__ import annotations

from typing import Optional

SOFT_COMPATIBLE_ROOM_GROUPS = (
    {
        "living room",
        "family room",
        "dining room",
        "kitchen",
        "entrance",
        "entryway",
        "foyer",
        "hallway",
    },
)


def normalize_room_label(value: Optional[str]) -> str:
    return (value or "").strip().lower().replace("_", " ")


def rooms_soft_compatible(room_a: Optional[str], room_b: Optional[str]) -> bool:
    normalized_a = normalize_room_label(room_a)
    normalized_b = normalize_room_label(room_b)
    if not normalized_a or normalized_a == "unknown":
        return True
    if not normalized_b or normalized_b == "unknown":
        return True
    if normalized_a == normalized_b:
        return True
    return any(normalized_a in group and normalized_b in group for group in SOFT_COMPATIBLE_ROOM_GROUPS)
