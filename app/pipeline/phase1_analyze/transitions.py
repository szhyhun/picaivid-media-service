"""Provisional transition candidates: pairs that may support a camera move.

Distinct from membership on purpose:

* **Membership** answers "is this the same physical room?" and drives grouping,
  coverage and per-room shot caps.
* **Transition** answers "can the camera move between these two views without the
  cut looking wrong?" It is a *geometric* question, not a semantic one.

A transition may therefore cross a room-instance boundary. The owner confirmed a
living -> kitchen glide across an open-plan seam is desirable, so the gate is not
"same room" -- it is strong mutual visibility plus a modest viewpoint change.

This module is report-only until transition-specific ground truth exists. Passing
these provisional geometry thresholds is not proof that a rendered interpolation
is visually safe.

`must_not_group` is accepted as a conservative blocker for now. It is not the right
long-term label: room-membership rejection and unsafe-motion rejection answer
different questions and need separate fields before editor feedback is wired.
"""
from __future__ import annotations

from dataclasses import dataclass

# Provisional and intentionally uncalibrated. `preferred_cinematic_pairs` cannot
# calibrate this gate because those labels select complementary storytelling views,
# not physically safe interpolation paths.
MIN_DEPTH_OK = 0.35        # shared visible surface, both directions
MAX_BL_OVER_DEPTH = 1.5    # viewpoint change small relative to room size
MAX_ROT_DEG = 60.0         # beyond this the two views share too little framing


@dataclass
class Transition:
    photo_a: int
    photo_b: int
    depth_ok: float
    bl_over_depth: float
    rot_deg: float
    crosses_rooms: bool = False

    @property
    def strength(self) -> float:
        return self.depth_ok


def build(
    evidence: list[dict],
    must_not_group: set[tuple[int, int]] | None = None,
    *,
    min_depth_ok: float = MIN_DEPTH_OK,
    max_bl_over_depth: float = MAX_BL_OVER_DEPTH,
    max_rot_deg: float = MAX_ROT_DEG,
    room_of: dict[int, str] | None = None,
) -> list[Transition]:
    """Return provisional geometry candidates, strongest first.

    `evidence` items are `PairEvidence.to_dict()` payloads with photo ids attached.
    """
    blocked = {(min(a, b), max(a, b)) for a, b in (must_not_group or set())}
    results: list[Transition] = []
    for item in evidence:
        a, b = sorted((item["photo_a_id"], item["photo_b_id"]))
        if (a, b) in blocked:
            continue
        depth_ok = item["depth_ok_min"]
        if depth_ok < min_depth_ok:
            continue
        if item["bl_over_depth"] > max_bl_over_depth:
            continue
        if item["rot_deg"] > max_rot_deg:
            continue
        crosses = bool(room_of and room_of.get(a) and room_of.get(b) and room_of[a] != room_of[b])
        results.append(Transition(a, b, depth_ok, item["bl_over_depth"], item["rot_deg"], crosses))
    return sorted(results, key=lambda t: (-t.strength, t.photo_a, t.photo_b))
