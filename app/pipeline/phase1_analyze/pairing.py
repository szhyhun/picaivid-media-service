"""Choose which shots of a room go in the film.

Membership answers "same room?"; this answers "which two views tell it best?".
Those are opposite geometric preferences and must not share thresholds:

* a good two-shot wants a **wide** angle between views -- a narrow one is a
  near-duplicate and adds nothing;
* smooth interpolation wants a **narrow** angle.

Calibrated against the owner's labeled `preferred_cinematic_pairs`. Measured on 48
chosen pairs vs 223 same-room pairs they did not choose:

    median depth_ok   0.44  vs 0.25
    median conf_pair 11.24  vs 2.82
    in 30-120 deg     81%   vs 51%

Per-room caps come from the owner: living room 3 pairs, any other room 2,
exterior+aerial 3-4 combined. Caps are ceilings, never quotas -- a room with one
good pair contributes one, and a room with none contributes a single shot or
nothing. There is deliberately no photo-utilization target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

ANGLE_LOW, ANGLE_HIGH = 30.0, 120.0
DUPLICATE_DEPTH_OK = 0.85
DUPLICATE_BL_OVER_DEPTH = 0.10

DEFAULT_CAPS = {"living": 3, "exterior": 4, "default": 2}


@dataclass
class ShotPair:
    photo_a: int
    photo_b: int
    score: float
    depth_ok: float
    rot_deg: float
    conf_pair: float
    is_duplicate: bool = False


def _angle_fit(rot_deg: float) -> float:
    """1.0 inside the complementary band, tapering outside it.

    Below ANGLE_LOW the two views are near-duplicates; far above ANGLE_HIGH they
    share too little of the room to read as one space.
    """
    if ANGLE_LOW <= rot_deg <= ANGLE_HIGH:
        return 1.0
    if rot_deg < ANGLE_LOW:
        return max(0.0, rot_deg / ANGLE_LOW)
    return max(0.0, math.exp(-(rot_deg - ANGLE_HIGH) / 45.0))


def _confidence_fit(conf_pair: float) -> float:
    """Saturating: chosen pairs sit near 11, unchosen near 2.8."""
    return min(1.0, math.log1p(max(conf_pair - 1.0, 0.0)) / math.log1p(10.0))


def score_pair(evidence: dict) -> float:
    """Editorial quality of a two-shot, 0..1. Higher is a better pair for the film."""
    depth_ok = evidence["depth_ok_min"]
    if depth_ok >= DUPLICATE_DEPTH_OK and evidence["bl_over_depth"] <= DUPLICATE_BL_OVER_DEPTH:
        return 0.0  # same view twice
    return (
        0.45 * min(depth_ok / 0.45, 1.0)          # shared geometry, saturating at the chosen median
        + 0.35 * _angle_fit(evidence["rot_deg"])  # complementary viewpoint
        + 0.20 * _confidence_fit(evidence["conf_pair"])
    )


def is_duplicate(evidence: dict) -> bool:
    return (
        evidence["depth_ok_min"] >= DUPLICATE_DEPTH_OK
        and evidence["bl_over_depth"] <= DUPLICATE_BL_OVER_DEPTH
    )


def cap_for(room_name: str | None, caps: dict[str, int] | None = None) -> int:
    caps = caps or DEFAULT_CAPS
    name = (room_name or "").lower()
    if "living" in name:
        return caps["living"]
    if any(token in name for token in ("exterior", "front", "back", "patio", "yard", "aerial", "drone")):
        return caps["exterior"]
    return caps["default"]


def select_for_room(
    evidence: list[dict], room_name: str | None = None, caps: dict[str, int] | None = None
) -> tuple[list[ShotPair], list[int]]:
    """Pick up to `cap` disjoint pairs for one room, best pair first.

    **Selection is lexicographic (best available pair, repeatedly), not sum-maximizing.**
    Exact maximum-weight matching was implemented and measured worse against the
    owner's labels (73% vs 77% of rooms containing a chosen pair), because
    maximizing a sum trades one excellent pair for two mediocre ones. That is
    backwards under the prime directive: a film of 30 excellent photos beats one of
    60 adequate ones. The plan originally specified max-weight matching; the data
    disagreed, so quality-first stands until evidence says otherwise.

    Returns (chosen pairs, photos left unpaired). Each photo appears at most once,
    so a room never shows the same image twice.
    """
    scored = [
        ShotPair(e["photo_a_id"], e["photo_b_id"], score_pair(e), e["depth_ok_min"],
                 e["rot_deg"], e["conf_pair"], is_duplicate(e))
        for e in evidence
    ]
    scored = [p for p in scored if p.score > 0.0]
    scored.sort(key=lambda p: (-p.score, p.photo_a, p.photo_b))

    cap = cap_for(room_name, caps)
    used: set[int] = set()
    chosen: list[ShotPair] = []
    for pair in scored:
        if len(chosen) >= cap:
            break
        if pair.photo_a in used or pair.photo_b in used:
            continue
        chosen.append(pair)
        used.update((pair.photo_a, pair.photo_b))

    photos = {p for e in evidence for p in (e["photo_a_id"], e["photo_b_id"])}
    return chosen, sorted(photos - used)
