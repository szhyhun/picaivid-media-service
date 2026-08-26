"""Component-local camera poses from canonical VGGT pair evidence.

Each two-photo Omega run has an arbitrary scale. The verifier stores translation
divided by that run's median depth, which makes edge lengths comparable enough for
debug visualization and component ordering. Disconnected components deliberately
remain in separate coordinate frames; this module never fabricates a whole-property
layout.
"""
from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraPose:
    """World-to-camera pose in a component-local coordinate frame."""

    rotation: np.ndarray
    translation: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return -self.rotation.T @ self.translation

    @property
    def view_direction(self) -> np.ndarray:
        direction = self.rotation.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        return direction / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])

    def extrinsic(self) -> np.ndarray:
        return np.column_stack((self.rotation, self.translation))


def _relative_transform(relation, source: int) -> tuple[int, np.ndarray, np.ndarray] | None:
    payload = (relation.relative_transform or {}).get("relative_pose") or {}
    try:
        rotation = np.asarray(payload["rotation"], dtype=np.float64)
        translation = np.asarray(payload["translation"], dtype=np.float64)
        scale = float(payload["scale"])
    except (KeyError, TypeError, ValueError):
        return None
    if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(rotation).all():
        return None
    if not np.isfinite(translation).all() or not math.isfinite(scale) or abs(scale) <= 1e-9:
        return None

    translation = translation / scale
    if source == relation.photo_a_id:
        return int(relation.photo_b_id), rotation, translation
    if source == relation.photo_b_id:
        inverse_rotation = rotation.T
        return int(relation.photo_a_id), inverse_rotation, -inverse_rotation @ translation
    return None


def solve_component_poses(
    photo_ids: list[int],
    relations: list,
) -> tuple[dict[int, CameraPose], dict[str, float | int]]:
    """Propagate the strongest verified relative poses through one component.

    This is a deterministic maximum-confidence spanning estimate, not bundle
    adjustment. Cycle residuals are reported so a later optimizer can replace the
    propagation without changing consumers.
    """
    if not photo_ids:
        return {}, {
            "pose_edges": 0,
            "constraint_edges": 0,
            "cycle_edges": 0,
            "missing_pose_count": 0,
            "median_cycle_translation_error": 0.0,
            "median_cycle_rotation_error_degrees": 0.0,
        }

    ids = {int(photo_id) for photo_id in photo_ids}
    usable = [
        relation
        for relation in relations
        if relation.is_connected
        and int(relation.photo_a_id) in ids
        and int(relation.photo_b_id) in ids
        and (relation.relative_transform or {}).get("relative_pose")
    ]
    adjacency: dict[int, list] = {photo_id: [] for photo_id in ids}
    for relation in usable:
        adjacency[int(relation.photo_a_id)].append(relation)
        adjacency[int(relation.photo_b_id)].append(relation)
    anchor = min(ids)
    poses = {
        anchor: CameraPose(np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    }
    selected_edges: set[tuple[int, int]] = set()
    frontier: list[tuple[float, int, int, int, int, object]] = []
    sequence = itertools.count()

    def add_frontier(source: int) -> None:
        for relation in adjacency[source]:
            target = (
                int(relation.photo_b_id)
                if source == int(relation.photo_a_id)
                else int(relation.photo_a_id)
            )
            if target in poses:
                continue
            heapq.heappush(
                frontier,
                (
                    -float(relation.relation_confidence or 0.0),
                    min(source, target),
                    max(source, target),
                    next(sequence),
                    source,
                    relation,
                ),
            )

    add_frontier(anchor)
    while frontier:
        _, _, _, _, source, relation = heapq.heappop(frontier)
        if source not in poses:
            continue
        transform = _relative_transform(relation, source)
        if transform is None:
            continue
        target, relative_rotation, relative_translation = transform
        if target in poses:
            continue
        source_pose = poses[source]
        poses[target] = CameraPose(
            rotation=relative_rotation @ source_pose.rotation,
            translation=relative_rotation @ source_pose.translation + relative_translation,
        )
        selected_edges.add((min(source, target), max(source, target)))
        add_frontier(target)

    # A constrained component should be connected by construction. Keep any
    # malformed-pose photo visible at the origin and report it instead of inventing
    # a transform or failing the whole listing.
    missing = sorted(ids - poses.keys())
    for photo_id in missing:
        poses[photo_id] = CameraPose(np.eye(3), np.zeros(3))

    cycle_translation_errors: list[float] = []
    cycle_rotation_errors: list[float] = []
    for relation in usable:
        left = int(relation.photo_a_id)
        right = int(relation.photo_b_id)
        if (min(left, right), max(left, right)) in selected_edges:
            continue
        transform = _relative_transform(relation, left)
        if transform is None:
            continue
        _, relative_rotation, relative_translation = transform
        predicted_rotation = poses[right].rotation @ poses[left].rotation.T
        predicted_translation = poses[right].translation - predicted_rotation @ poses[left].translation
        cycle_translation_errors.append(
            float(np.linalg.norm(predicted_translation - relative_translation))
        )
        rotation_delta = relative_rotation @ predicted_rotation.T
        cycle_rotation_errors.append(
            math.degrees(
                math.acos(float(np.clip((np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0)))
            )
        )

    return poses, {
        "pose_edges": len(selected_edges),
        "constraint_edges": len(usable),
        "cycle_edges": len(cycle_translation_errors),
        "missing_pose_count": len(missing),
        "median_cycle_translation_error": (
            float(np.median(cycle_translation_errors)) if cycle_translation_errors else 0.0
        ),
        "median_cycle_rotation_error_degrees": (
            float(np.median(cycle_rotation_errors)) if cycle_rotation_errors else 0.0
        ),
    }
