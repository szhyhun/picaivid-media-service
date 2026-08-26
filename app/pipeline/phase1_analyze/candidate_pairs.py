"""Nominate photo pairs worth verifying.

Semantics nominate, geometry decides. Nothing here may merge photos or imply
membership: a nomination is only a proposal to spend ~0.8 s of GPU time.

Measured on the labeled 56-photo listings: the primary union is 258 pairs and
connects every truth room. Global candidates are retained as a fallback tier;
including them immediately raises the union to 413-448 pairs.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

LABEL_CAP = 4          # nearest same-label peers per photo, by upload position
CLIP_K = 5             # nearest CLIP neighbours per photo
ADJACENCY = 2          # +/- upload positions
GLOBAL_K = 4           # nearest by global same_scene_score
GLOBAL_DIST = 0.5      # x scene scale
GLOBAL_VIEW_DOT = 0.3


PRIMARY_SOURCES = {"label", "clip", "adjacency"}


@dataclass
class CandidatePair:
    photo_a: int
    photo_b: int
    sources: set[str] = field(default_factory=set)

    @property
    def key(self) -> tuple[int, int]:
        return (min(self.photo_a, self.photo_b), max(self.photo_a, self.photo_b))

    @property
    def tier(self) -> str:
        """`primary` pairs are verified first; `fallback` is for unresolved photos/components.

        Measured: the primary union alone connects 23/23, 19/19 and 19/19 labeled rooms
        on the calibration listings, while global nomination adds 155-195 pairs per
        listing. Those pairs are not free -- each costs an Omega run -- and they come
        from the global reconstruction that is unreliable at room scale, so they are a
        fallback tier rather than part of the default union.
        """
        return "primary" if self.sources & PRIMARY_SOURCES else "fallback"


def _normalize(label: str | None) -> str:
    return (label or "").strip().lower()


def nominate(
    photos: list[dict],
    *,
    embeddings: dict[int, list[float]] | None = None,
    global_scores: dict[tuple[int, int], float] | None = None,
    centers: dict[int, np.ndarray] | None = None,
    views: dict[int, np.ndarray] | None = None,
    label_cap: int = LABEL_CAP,
    clip_k: int = CLIP_K,
    adjacency: int = ADJACENCY,
) -> list[CandidatePair]:
    """Return deterministic candidate pairs for `photos` (dicts: id, position, room_label)."""
    by_id = {int(photo["id"]): photo for photo in photos}
    ids = sorted(by_id)
    position = {pid: int(by_id[pid].get("position") or 0) for pid in ids}
    label = {pid: _normalize(by_id[pid].get("room_label")) for pid in ids}
    pairs: dict[tuple[int, int], CandidatePair] = {}

    def add(a: int, b: int, source: str) -> None:
        if a == b:
            return
        key = (min(a, b), max(a, b))
        pairs.setdefault(key, CandidatePair(*key)).sources.add(source)

    # 1. same room label, capped by upload proximity. Uncapped, a 7-bedroom listing
    #    contributes every bedroom x bedroom pair and the graph grows quadratically
    #    in the largest label class.
    for pid in ids:
        if not label[pid]:
            continue
        peers = sorted(
            (other for other in ids if other != pid and label[other] == label[pid]),
            key=lambda other: (abs(position[other] - position[pid]), other),
        )[:label_cap]
        for other in peers:
            add(pid, other, "label")

    # 2. CLIP nearest neighbours
    if embeddings:
        usable = [pid for pid in ids if embeddings.get(pid)]
        if len(usable) > 1:
            matrix = np.array([embeddings[pid] for pid in usable], dtype=np.float64)
            matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
            similarity = matrix @ matrix.T
            np.fill_diagonal(similarity, -np.inf)
            for row, pid in enumerate(usable):
                for column in np.argsort(-similarity[row])[:clip_k]:
                    add(pid, usable[int(column)], "clip")

    # 3. upload adjacency
    ordered = sorted(ids, key=lambda pid: (position[pid], pid))
    for index, pid in enumerate(ordered):
        for other in ordered[index + 1 : index + 1 + adjacency]:
            add(pid, other, "adjacency")

    # 4. Global features are already available, but verifying the pairs they
    #    nominate is not free. Keep global-only proposals in the fallback tier.
    if global_scores:
        ranked: dict[int, list[tuple[float, int]]] = defaultdict(list)
        for (a, b), score in global_scores.items():
            ranked[a].append((score, b))
            ranked[b].append((score, a))
        for pid, entries in ranked.items():
            for _, other in sorted(entries, reverse=True)[:GLOBAL_K]:
                add(pid, other, "global_rank")
    if centers and views:
        distances = [
            float(np.linalg.norm(centers[a] - centers[b]))
            for a, b in itertools.combinations(sorted(centers), 2)
        ]
        scale = float(np.median(distances)) if distances else 1.0
        for a, b in itertools.combinations(sorted(centers), 2):
            close = float(np.linalg.norm(centers[a] - centers[b])) < GLOBAL_DIST * scale
            aligned = float(views[a] @ views[b]) > GLOBAL_VIEW_DOT
            if close and aligned:
                add(a, b, "global_frustum")

    return [pairs[key] for key in sorted(pairs)]


def split_tiers(candidates: list[CandidatePair]) -> tuple[list[CandidatePair], list[CandidatePair]]:
    """Return (primary, fallback). Verify primary first; escalate only where needed."""
    primary = [c for c in candidates if c.tier == "primary"]
    fallback = [c for c in candidates if c.tier == "fallback"]
    return primary, fallback


def unconnected_photos(
    candidates: list[CandidatePair], verified_pairs: set[tuple[int, int]], photo_ids: list[int]
) -> set[int]:
    """Photos with no verified edge — the only ones worth escalating to the fallback tier."""
    connected: set[int] = set()
    for a, b in verified_pairs:
        connected.add(a)
        connected.add(b)
    return {pid for pid in photo_ids if pid not in connected}
