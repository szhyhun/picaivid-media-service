"""Build room components from verified pair evidence.

Replaces naive connected components. Two rules matter:

1. **Absence of evidence is not a negative.** Measured: at conf_pair <= 1.5 the
   low-confidence band still contains 10 genuine same-room pairs per listing, so a
   confidence-based blocker would break real rooms. Only explicit human
   `must_not_group` evidence may block a merge.

2. **Merge in evidence order.** Strongest edges first, so a weak edge can never
   pre-empt a strong one, and the result is independent of input ordering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations


@dataclass
class MergeDecision:
    """One accepted or blocked merge, with the evidence behind it."""

    photo_a: int
    photo_b: int
    accepted: bool
    reason: str
    strength: float = 0.0
    blocking_pair: tuple[int, int] | None = None


@dataclass
class MembershipResult:
    components: list[list[int]] = field(default_factory=list)
    merge_log: list[MergeDecision] = field(default_factory=list)

    def component_of(self) -> dict[int, int]:
        return {photo: index for index, comp in enumerate(self.components) for photo in comp}


def constrained_merge(
    photo_ids: list[int],
    edges: list[tuple[int, int, float]],
    must_not_group: set[tuple[int, int]] | None = None,
) -> MembershipResult:
    """Merge photos into components using verified edges and human negatives.

    `edges` are (photo_a, photo_b, strength); only edges the caller has already
    classified as same-room should be passed. `must_not_group` holds human-labeled
    pairs that may never share a component -- the only automatic hard blocker,
    because no confidence band separates different-room pairs cleanly.
    """
    blocked = {(min(a, b), max(a, b)) for a, b in (must_not_group or set())}
    parent = {photo: photo for photo in photo_ids}
    members: dict[int, set[int]] = {photo: {photo} for photo in photo_ids}

    def find(photo: int) -> int:
        while parent[photo] != photo:
            parent[photo] = parent[parent[photo]]
            photo = parent[photo]
        return photo

    def blocking_pair(left: set[int], right: set[int]) -> tuple[int, int] | None:
        # Sets do not promise a stable iteration order. Returning the first
        # blocker is part of the persisted audit log, so make it deterministic.
        for a in sorted(left):
            for b in sorted(right):
                key = (min(a, b), max(a, b))
                if key in blocked:
                    return key
        return None

    log: list[MergeDecision] = []
    canonical_edges = [(min(a, b), max(a, b), strength) for a, b, strength in edges]
    # Strongest first; canonical endpoint ids break ties independently of the
    # orientation in which a caller supplied the undirected edge.
    for a, b, strength in sorted(canonical_edges, key=lambda e: (-e[2], e[0], e[1])):
        if a not in parent or b not in parent:
            continue
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            continue
        conflict = blocking_pair(members[root_a], members[root_b])
        if conflict is not None:
            log.append(MergeDecision(a, b, False, "must_not_group", strength, conflict))
            continue
        # union by size keeps the tree shallow
        if len(members[root_a]) < len(members[root_b]):
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        members[root_a] |= members[root_b]
        del members[root_b]
        log.append(MergeDecision(a, b, True, "verified_same_room", strength))

    groups: dict[int, list[int]] = {}
    for photo in photo_ids:
        groups.setdefault(find(photo), []).append(photo)
    return MembershipResult(
        components=[sorted(group) for group in sorted(groups.values(), key=min)],
        merge_log=log,
    )


def blocked_pairs_from_truth(room_instances: list[dict], must_not_group: list[list]) -> set:
    """Convenience: normalize labeled negatives into an id-keyed set."""
    return {(min(a, b), max(a, b)) for a, b in must_not_group if a is not None and b is not None}
