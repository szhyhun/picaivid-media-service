"""Candidate retrieval for the precision-first transition pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from sklearn.preprocessing import normalize


TOP_K = 8
MIN_DINO_SIMILARITY = 0.50
STRONG_ONE_WAY_DINO_SIMILARITY = 0.65
LOCAL_ADJACENCY_MAX_GAP = 1

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


@dataclass(frozen=True)
class CandidatePair:
    left_idx: int
    right_idx: int
    dinov2_similarity: float
    pair_source: str


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


def rooms_explicitly_incompatible(room_a: Optional[str], room_b: Optional[str]) -> bool:
    normalized_a = normalize_room_label(room_a)
    normalized_b = normalize_room_label(room_b)
    if not normalized_a or normalized_a == "unknown":
        return False
    if not normalized_b or normalized_b == "unknown":
        return False
    if rooms_soft_compatible(normalized_a, normalized_b):
        return False
    return normalized_a != normalized_b


def _mutual_topk_pairs(similarity: np.ndarray, k: int) -> tuple[set[tuple[int, int]], list[list[int]]]:
    n = similarity.shape[0]
    topk_lists: list[list[int]] = []
    mutual: set[tuple[int, int]] = set()
    for idx in range(n):
        sorted_indices = np.argsort(-similarity[idx])
        neighbors = [j for j in sorted_indices if j != idx][:k]
        topk_lists.append(neighbors)
    for i in range(n):
        for j in topk_lists[i]:
            if i in topk_lists[j]:
                mutual.add((min(i, j), max(i, j)))
    return mutual, topk_lists


def build_candidate_pairs(
    images: Iterable[object],
    positions: list[int],
    room_labels: Optional[list[str]] = None,
    k: int = TOP_K,
    embeddings: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, list[CandidatePair]]:
    image_list = list(images)
    if embeddings is None:
        from app.pipeline.phase1_analyze.learned_matching import compute_dinov2_embeddings

        embeddings = compute_dinov2_embeddings(image_list)
    embeddings = normalize(embeddings)
    similarity = embeddings @ embeddings.T

    mutual_pairs, topk_lists = _mutual_topk_pairs(similarity, k)
    candidates: dict[tuple[int, int], CandidatePair] = {}
    n = len(image_list)

    for i in range(n):
        for j in topk_lists[i]:
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            score = float(similarity[i, j])
            if score < MIN_DINO_SIMILARITY:
                continue
            if room_labels and rooms_explicitly_incompatible(room_labels[i], room_labels[j]):
                continue
            is_mutual = key in mutual_pairs
            is_strong_one_way = score >= STRONG_ONE_WAY_DINO_SIMILARITY
            is_adjacent = abs(int(positions[i]) - int(positions[j])) <= LOCAL_ADJACENCY_MAX_GAP
            if not (is_mutual or is_strong_one_way or is_adjacent):
                continue
            if is_mutual:
                pair_source = "dinov2_mutual_topk"
            elif is_strong_one_way:
                pair_source = "dinov2_strong_one_way"
            else:
                pair_source = "dinov2_adjacent_recovery"
            candidates[key] = CandidatePair(
                left_idx=key[0],
                right_idx=key[1],
                dinov2_similarity=score,
                pair_source=pair_source,
            )

    return embeddings, similarity, sorted(candidates.values(), key=lambda item: (item.left_idx, item.right_idx))
