"""Certified sequence construction for precision-first transition planning."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


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


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (min(a, b), max(a, b))


def _room_compatible(room_a: str | None, room_b: str | None) -> bool:
    left = (room_a or "").strip().lower()
    right = (room_b or "").strip().lower()
    if not left or left == "unknown":
        return True
    if not right or right == "unknown":
        return True
    if any(left in group and right in group for group in SOFT_COMPATIBLE_ROOM_GROUPS):
        return True
    return left == right


def _sequence_score(path_edges: list[dict[str, Any]], positions: dict[int, int], blob_centers: list[float]) -> float:
    pair_ranks = [float(edge.get("pair_rank", 0.0) or 0.0) for edge in path_edges]
    overlaps = [float(edge.get("overlap_ratio", 0.0) or 0.0) for edge in path_edges]
    coverages = [float(edge.get("coverage_4x4", 0.0) or 0.0) for edge in path_edges]
    crossings = [1.0 - float(edge.get("crossing_penalty", 0.0) or 0.0) for edge in path_edges]
    order_steps = []
    for edge in path_edges:
        order_steps.append(float(edge.get("order_proximity", 0.0) or 0.0))
    overlap_stability = 1.0 - float(np.std(overlaps)) if len(overlaps) > 1 else (overlaps[0] if overlaps else 0.0)
    coverage_stability = 1.0 - float(np.std(coverages)) if len(coverages) > 1 else (coverages[0] if coverages else 0.0)
    order_smoothness = float(np.mean(order_steps)) if order_steps else 0.0
    crossing_safety = float(np.mean(crossings)) if crossings else 0.0
    anchor_persistence = 1.0 - float(np.var(blob_centers)) if len(blob_centers) > 1 else 1.0
    return float(
        np.clip(
            0.35 * float(np.mean(pair_ranks))
            + 0.20 * overlap_stability
            + 0.15 * coverage_stability
            + 0.15 * order_smoothness
            + 0.10 * crossing_safety
            + 0.05 * anchor_persistence,
            0.0,
            1.0,
        )
    )


def build_transition_sequences(
    photo_ids: list[int],
    pair_records: list[dict[str, Any]],
    room_labels: dict[int, str | None],
    cluster_by_photo: dict[int, int | None],
    max_sequences: int = 5,
) -> list[dict[str, Any]]:
    by_status = defaultdict(list)
    for record in pair_records:
        by_status[str(record.get("certification_status") or "reject")].append(record)

    def _search(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        adjacency: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            left = int(edge["photo_a_id"])
            right = int(edge["photo_b_id"])
            if not _room_compatible(room_labels.get(left), room_labels.get(right)):
                continue
            adjacency[left].append(edge)
            adjacency[right].append(edge)

        sequences: list[dict[str, Any]] = []
        for start in photo_ids:
            beams: list[tuple[list[int], list[dict[str, Any]], list[float]]] = [([start], [], [])]
            for _ in range(4):
                next_beams: list[tuple[list[int], list[dict[str, Any]], list[float]]] = []
                for path, path_edges, centers in beams:
                    current = path[-1]
                    for edge in sorted(adjacency.get(current, []), key=lambda row: float(row.get("pair_rank", 0.0) or 0.0), reverse=True)[:6]:
                        neighbor = int(edge["photo_b_id"] if int(edge["photo_a_id"]) == current else edge["photo_a_id"])
                        if neighbor in path:
                            continue
                        if float(edge.get("crossing_penalty", 0.0) or 0.0) >= 0.8:
                            continue
                        next_centers = centers + [float(edge.get("dominant_foreground_side_a", 1) or 1)]
                        next_beams.append((path + [neighbor], path_edges + [edge], next_centers))
                if not next_beams:
                    break
                next_beams.sort(
                    key=lambda item: _sequence_score(
                        item[1],
                        {},
                        item[2],
                    ),
                    reverse=True,
                )
                beams = next_beams[:8]
            for path, path_edges, centers in beams:
                if 3 <= len(path) <= 5 and path_edges:
                    source_clusters = sorted({cluster_by_photo.get(pid) for pid in path if cluster_by_photo.get(pid) is not None})
                    sequences.append(
                        {
                            "photo_ids": path,
                            "edges": path_edges,
                            "source_cluster_ids": source_clusters,
                            "sequence_score": _sequence_score(path_edges, {}, centers),
                            "certification_status": "strong" if all(str(edge.get("certification_status")) == "strong" for edge in path_edges) else "usable",
                        }
                    )
        sequences.sort(key=lambda item: float(item["sequence_score"]), reverse=True)
        deduped: list[dict[str, Any]] = []
        used_sets: list[set[int]] = []
        for sequence in sequences:
            current = set(int(pid) for pid in sequence["photo_ids"])
            if any(len(current & existing) >= 2 for existing in used_sets):
                continue
            used_sets.append(current)
            deduped.append(sequence)
            if len(deduped) >= max_sequences:
                break
        return deduped

    strong_sequences = _search(by_status.get("strong", []))
    if strong_sequences:
        return strong_sequences
    return _search(by_status.get("usable", []))
