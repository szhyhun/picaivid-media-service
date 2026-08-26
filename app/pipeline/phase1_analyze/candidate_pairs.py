"""Nominate photo pairs worth verifying.

Semantics nominate, geometry decides. Nothing here may merge photos or imply
membership: a nomination is only a proposal to spend ~0.8 s of GPU time.

Measured on the labeled 56-photo listings: the label + CLIP + upload-adjacency
union is about 258 pairs and has oracle connectivity for every truth room.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LABEL_CAP = 4          # nearest same-label peers per photo, by upload position
CLIP_K = 5             # nearest CLIP neighbours per photo
ADJACENCY = 2          # +/- upload positions


@dataclass
class CandidatePair:
    photo_a: int
    photo_b: int
    sources: set[str] = field(default_factory=set)

    @property
    def key(self) -> tuple[int, int]:
        return (min(self.photo_a, self.photo_b), max(self.photo_a, self.photo_b))


def _normalize(label: str | None) -> str:
    return (label or "").strip().lower()


def nominate(
    photos: list[dict],
    *,
    embeddings: dict[int, list[float]] | None = None,
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

    return [pairs[key] for key in sorted(pairs)]
