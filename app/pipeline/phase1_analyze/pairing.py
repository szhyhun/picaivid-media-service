"""Choose renderable two-keyframe shots using VGGT evidence.

Membership answers "is this part of the same broad scene?". A pair shot answers
"can the renderer safely use these two images as endpoints?". Only the latter
can create video motion, so a VGGT transition gate is a hard prerequisite.

The older `preferred_cinematic_pairs` labels selected complementary story views;
they are not ground truth for interpolability. They can influence how safe pairs
are ranked, but cannot make a non-overlapping or high-rotation pair renderable.

Per-room caps come from the owner: living room 3 pairs, any other room 2,
exterior+aerial 3-4 combined. Caps are ceilings, never quotas -- a room with one
good pair contributes one, and a room with none contributes a single shot or
nothing. There is deliberately no photo-utilization target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from app.pipeline.phase1_analyze.transitions import is_duplicate, is_transition_safe

MOTION_SWEET_SPOT_DEG = 35.0
MOTION_MAX_DEG = 60.0

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
    """Reward visible but still smooth camera movement inside the safe band."""
    if rot_deg < 0.0 or rot_deg > MOTION_MAX_DEG:
        return 0.0
    if rot_deg <= MOTION_SWEET_SPOT_DEG:
        return rot_deg / MOTION_SWEET_SPOT_DEG
    return max(0.0, 1.0 - (rot_deg - MOTION_SWEET_SPOT_DEG) /
               (MOTION_MAX_DEG - MOTION_SWEET_SPOT_DEG))


def _confidence_fit(conf_pair: float) -> float:
    """Saturating: chosen pairs sit near 11, unchosen near 2.8."""
    return min(1.0, math.log1p(max(conf_pair - 1.0, 0.0)) / math.log1p(10.0))


def score_pair(evidence: dict) -> float:
    """Quality of a safe VGGT pair, 0..1. Unsafe pairs always score zero."""
    if not is_transition_safe(evidence):
        return 0.0
    depth_ok = evidence["depth_ok_min"]
    return (
        0.50 * min(depth_ok / 0.45, 1.0)
        + 0.30 * _angle_fit(evidence["rot_deg"])
        + 0.20 * _confidence_fit(evidence["conf_pair"])
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
