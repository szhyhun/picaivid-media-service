"""VGGT-backed transition candidates for two-keyframe camera moves.

Distinct from membership on purpose:

* **Membership** answers "is this the same physical room?" and drives grouping,
  coverage and per-room shot caps.
* **Transition** answers "can the camera move between these two views without the
  cut looking wrong?" It is a *geometric* question, not a semantic one.

A transition may therefore cross a room-instance boundary. The owner confirmed a
living -> kitchen glide across an open-plan seam is desirable, so the gate is not
"same room" -- it is strong mutual visibility plus a modest viewpoint change.

The thresholds are deliberately conservative. Passing them authorizes a pair as
two real keyframes for the renderer; it is not proof that any particular video
model will interpolate the pair perfectly. Renderer validation remains a
separate, future acceptance gate.

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

# A nearly coincident camera pose is stronger duplicate evidence than a semantic
# label. The lower overlap band covers real-world JPEG/lighting differences where
# the same frame does not retain the very high depth agreement of an exact file
# duplicate. These are still entirely VGGT-derived measurements.
DUPLICATE_DEPTH_OK = 0.85
DUPLICATE_BL_OVER_DEPTH = 0.10
DUPLICATE_NEAR_ZERO_DEPTH_OK = 0.75
DUPLICATE_NEAR_ZERO_BL_OVER_DEPTH = 0.01
DUPLICATE_NEAR_ZERO_ROT_DEG = 1.0


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


def pair_key(item: dict) -> tuple[int, int]:
    """Return a stable undirected key for persisted pair evidence."""
    return tuple(sorted((int(item["photo_a_id"]), int(item["photo_b_id"]))))


def is_duplicate(item: dict) -> bool:
    """Whether VGGT measured the two inputs as the same camera view."""
    depth_ok = float(item.get("depth_ok_min", 0.0))
    baseline = float(item.get("bl_over_depth", float("inf")))
    rotation = float(item.get("rot_deg", float("inf")))
    return (
        depth_ok >= DUPLICATE_DEPTH_OK
        and baseline <= DUPLICATE_BL_OVER_DEPTH
    ) or (
        depth_ok >= DUPLICATE_NEAR_ZERO_DEPTH_OK
        and baseline <= DUPLICATE_NEAR_ZERO_BL_OVER_DEPTH
        and rotation <= DUPLICATE_NEAR_ZERO_ROT_DEG
    )


def is_transition_safe(
    item: dict,
    must_not_group: set[tuple[int, int]] | None = None,
    *,
    min_depth_ok: float = MIN_DEPTH_OK,
    max_bl_over_depth: float = MAX_BL_OVER_DEPTH,
    max_rot_deg: float = MAX_ROT_DEG,
) -> bool:
    """Return whether *measured VGGT geometry* supports a keyframe pair.

    Membership is intentionally wider than this gate. Do not substitute a
    component id, CLIP room label, upload adjacency, or a planner score here.
    """
    blocked = {tuple(sorted((int(a), int(b)))) for a, b in (must_not_group or set())}
    if pair_key(item) in blocked or is_duplicate(item):
        return False
    return (
        float(item.get("depth_ok_min", 0.0)) >= min_depth_ok
        and float(item.get("bl_over_depth", float("inf"))) <= max_bl_over_depth
        and float(item.get("rot_deg", float("inf"))) <= max_rot_deg
    )


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
    results: list[Transition] = []
    for item in evidence:
        if not is_transition_safe(
            item,
            must_not_group,
            min_depth_ok=min_depth_ok,
            max_bl_over_depth=max_bl_over_depth,
            max_rot_deg=max_rot_deg,
        ):
            continue
        a, b = pair_key(item)
        depth_ok = float(item["depth_ok_min"])
        crosses = bool(room_of and room_of.get(a) and room_of.get(b) and room_of[a] != room_of[b])
        results.append(Transition(a, b, depth_ok, item["bl_over_depth"], item["rot_deg"], crosses))
    return sorted(results, key=lambda t: (-t.strength, t.photo_a, t.photo_b))
