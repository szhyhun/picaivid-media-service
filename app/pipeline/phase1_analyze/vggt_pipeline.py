"""VGGT V2 pair verification, component membership, and cinematic ordering.

There is intentionally no whole-listing reconstruction path here. Omega runs on
candidate photo pairs, cached raw evidence decides membership, and verified
relative poses create one honest coordinate frame per connected component.
"""
from __future__ import annotations

import gc
import logging
import math
import os
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from app.models.vggt import vggt_model
from app.pipeline.phase1_analyze.pose_graph import CameraPose, solve_component_poses
logger = logging.getLogger(__name__)

# Calibrated on 813 verified pairs across three human-labeled listings. These are
# provisional until a true holdout is labeled; the acceptance target is not met.
MEMBERSHIP_DEPTH_OK = 0.20
MEMBERSHIP_BL_OVER_DEPTH = 2.0
PAIR_CACHE_RELEASE_EVERY = 25

SOCIAL_LABEL_TOKENS = ("living", "kitchen", "dining", "great room", "family room")


@dataclass
class PhotoGeometryResult:
    photo_id: int
    pose_confidence: float
    depth_confidence: float
    visibility_score: float
    reprojection_error: float
    camera_extrinsic: list[list[float]]
    camera_center: list[float]
    view_direction: list[float]
    local_metrics: dict[str, Any]


@dataclass
class PhotoRelationResult:
    photo_a_id: int
    photo_b_id: int
    overlap_score: float
    reprojection_score: float
    relation_confidence: float
    baseline_distance: float
    relative_transform: dict[str, Any]
    direction_dx: float
    direction_dy: float
    continuity_type: str
    is_bridge_edge: bool
    is_connected: bool
    debug_metrics: dict[str, Any]


@dataclass
class SceneComponentResult:
    component_key: str
    photo_ids: list[int]
    ordered_photo_ids: list[int]
    scene_type: str
    geometry_confidence: float
    connectivity_confidence: float
    avg_reprojection_error: float
    hero_photo_id: int
    motion_affordance: str
    debug_metrics: dict[str, Any]


def _save_input_images(images: list[object], photo_ids: list[int]) -> tuple[str, list[str]]:
    directory = tempfile.mkdtemp(prefix="vggt_scene_")
    filelist: list[str] = []
    for index, (image, photo_id) in enumerate(zip(images, photo_ids)):
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            raise RuntimeError(f"Unsupported VGGT image type: {type(image)!r}")
        path = os.path.join(directory, f"{index:03d}_{int(photo_id)}.png")
        pil_image.convert("RGB").save(path)
        filelist.append(path)
    return directory, filelist


def _cleanup_files(directory: str, filelist: list[str]) -> None:
    for path in filelist:
        try:
            os.remove(path)
        except OSError:
            pass
    try:
        os.rmdir(directory)
    except OSError:
        pass


def _release_accelerator_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


def _scene_type_for_label(label: str | None) -> str:
    text = str(label or "").lower()
    if "drone" in text or "aerial" in text:
        return "drone"
    if any(token in text for token in ("exterior", "front", "backyard", "yard", "patio", "deck", "pool")):
        return "exterior"
    return "interior" if text else "mixed"


def _scene_type(component: list[int], room_labels: dict[int, str]) -> str:
    counts = Counter(_scene_type_for_label(room_labels.get(photo_id)) for photo_id in component)
    return min(counts, key=lambda value: (-counts[value], value)) if counts else "mixed"


def _confidence_fit(raw_confidence: float) -> float:
    return float(np.clip(math.log1p(max(raw_confidence - 1.0, 0.0)) / math.log1p(10.0), 0.0, 1.0))


def _relative_depth_errors(evidence: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for direction in (evidence.get("forward") or {}, evidence.get("backward") or {}):
        value = direction.get("median_relative_depth_error")
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _reprojection_score(evidence: dict[str, Any]) -> float:
    from app.pipeline.phase1_analyze.pairwise_verify import DEPTH_AGREEMENT

    errors = _relative_depth_errors(evidence)
    if not errors:
        return 0.0
    return float(np.clip(1.0 - np.median(errors) / DEPTH_AGREEMENT, 0.0, 1.0))


def _normalized_pair_translation(evidence: dict[str, Any]) -> list[float]:
    pose = evidence.get("relative_pose") or {}
    try:
        translation = np.asarray(pose["translation"], dtype=np.float64)
        scale = float(pose["scale"])
    except (KeyError, TypeError, ValueError):
        return [0.0, 0.0, 0.0]
    if translation.shape != (3,) or not np.isfinite(translation).all() or abs(scale) <= 1e-9:
        return [0.0, 0.0, 0.0]
    return [float(value) for value in translation / scale]


def _v2_scene_graph(
    file_by_photo: dict[int, str],
    labels: dict[int, str],
    positions: dict[int, int],
    embeddings: dict[int, list] | None = None,
    must_not_group: set[tuple[int, int]] | None = None,
    *,
    runtime: dict[str, Any] | None = None,
) -> tuple[list[PhotoRelationResult], list[list[int]], list[dict[str, Any]], dict[str, Any]]:
    """Verify nominated pairs and build constrained physical-room components."""
    from app.pipeline.phase1_analyze import pairing, transitions
    from app.pipeline.phase1_analyze.candidate_pairs import nominate
    from app.pipeline.phase1_analyze.membership import constrained_merge
    from app.pipeline.phase1_analyze.pairwise_verify import (
        canonical_order_with_ids,
        verify_with_cache,
    )

    photo_ids = sorted(file_by_photo)
    photos = [
        {"id": photo_id, "position": positions.get(photo_id, 0), "room_label": labels.get(photo_id)}
        for photo_id in photo_ids
    ]
    candidates = nominate(photos, embeddings=embeddings or None)
    logger.info(
        "VGGT_V2_CANDIDATES photos=%s pairs=%s",
        len(photo_ids),
        len(candidates),
    )

    runtime = runtime or vggt_model.runtime_metadata()
    evidence: list[dict[str, Any]] = []
    cache_hits = computed = failed = released_at = 0
    for index, candidate in enumerate(candidates, 1):
        left, right = candidate.key
        try:
            (path_a, evidence_a), (path_b, evidence_b) = canonical_order_with_ids(
                file_by_photo[left], left, file_by_photo[right], right
            )
            record, cache_hit, _ = verify_with_cache(path_a, path_b, runtime=runtime)
            cache_hits += int(cache_hit)
            computed += int(not cache_hit)
            record.update(
                {
                    "photo_a_id": evidence_a,
                    "photo_b_id": evidence_b,
                    "sources": sorted(candidate.sources),
                }
            )
            evidence.append(record)
            if not cache_hit and computed - released_at >= PAIR_CACHE_RELEASE_EVERY:
                _release_accelerator_cache()
                released_at = computed
        except Exception as error:  # one bad pair must not fail the listing
            failed += 1
            logger.warning("Pair verification failed for %s/%s: %s", left, right, error)
        if index % PAIR_CACHE_RELEASE_EVERY == 0 or index == len(candidates):
            logger.info(
                "VGGT_V2_PAIR_PROGRESS attempted=%s/%s computed=%s cache_hits=%s failed=%s",
                index,
                len(candidates),
                computed,
                cache_hits,
                failed,
            )
    if computed > released_at:
        _release_accelerator_cache()

    merge_edges = [
        (record["photo_a_id"], record["photo_b_id"], float(record["depth_ok_min"]))
        for record in evidence
        if record["depth_ok_min"] > MEMBERSHIP_DEPTH_OK
        and record["bl_over_depth"] < MEMBERSHIP_BL_OVER_DEPTH
    ]
    merge_edge_keys = {(min(left, right), max(left, right)) for left, right, _ in merge_edges}
    membership = constrained_merge(photo_ids, merge_edges, must_not_group)
    component_of = membership.component_of()

    transition_keys = {
        (min(item.photo_a, item.photo_b), max(item.photo_a, item.photo_b))
        for item in transitions.build(evidence, must_not_group)
    }
    relations: list[PhotoRelationResult] = []
    for record in evidence:
        left = int(record["photo_a_id"])
        right = int(record["photo_b_id"])
        key = (min(left, right), max(left, right))
        same_component = component_of.get(left) is not None and component_of.get(left) == component_of.get(right)
        direct_membership_edge = same_component and key in merge_edge_keys
        is_transition = key in transition_keys
        if direct_membership_edge:
            continuity = "same_scene"
        elif is_transition:
            continuity = "doorway_bridge"
        elif record["depth_ok_min"] > 0.0:
            continuity = "cut_only"
        else:
            continuity = "unrelated"
        relations.append(
            PhotoRelationResult(
                photo_a_id=left,
                photo_b_id=right,
                overlap_score=float(record["depth_ok_min"]),
                reprojection_score=_reprojection_score(record),
                relation_confidence=_confidence_fit(float(record["conf_pair"])),
                baseline_distance=float(record["baseline"]),
                relative_transform={
                    "translation": _normalized_pair_translation(record),
                    "rotation_degrees": float(record["rot_deg"]),
                    "bl_over_depth": float(record["bl_over_depth"]),
                    "relative_pose": record.get("relative_pose"),
                    "coordinate_frame": "pair_local",
                },
                direction_dx=float(record["forward"]["median_dx"]),
                direction_dy=float(record["forward"]["median_dy"]),
                continuity_type=continuity,
                is_bridge_edge=continuity == "doorway_bridge",
                is_connected=direct_membership_edge,
                debug_metrics={
                    "verification_source": "vggt_pairwise_v2",
                    "depth_ok_forward": record["forward"]["depth_ok"],
                    "depth_ok_backward": record["backward"]["depth_ok"],
                    "conf_pair": record["conf_pair"],
                    "bl_over_depth": record["bl_over_depth"],
                    "rotation_degrees": record["rot_deg"],
                    "pair_score": pairing.score_pair(record),
                    "is_duplicate": pairing.is_duplicate(record),
                    "is_transition": is_transition,
                    "same_component": same_component,
                    "direct_membership_edge": direct_membership_edge,
                    "sources": record["sources"],
                },
            )
        )

    blocked = sum(1 for decision in membership.merge_log if not decision.accepted)
    stats = {
        "candidate_pairs": len(candidates),
        "verified_pairs": len(evidence),
        "computed_pairs": computed,
        "cache_hits": cache_hits,
        "failed_pairs": failed,
        "blocked_merges": blocked,
        "coordinate_scope": "component_local",
    }
    logger.info(
        "VGGT_V2_COMPONENTS count=%s blocked_merges=%s",
        len(membership.components),
        blocked,
    )
    return relations, membership.components, evidence, stats


def _relation_lookup(relations: list[PhotoRelationResult]) -> dict[tuple[int, int], PhotoRelationResult]:
    return {
        (min(relation.photo_a_id, relation.photo_b_id), max(relation.photo_a_id, relation.photo_b_id)): relation
        for relation in relations
    }


def _edge_score(
    left_id: int,
    right_id: int,
    relations: dict[tuple[int, int], PhotoRelationResult],
    positions: dict[int, int],
) -> float:
    relation = relations.get((min(left_id, right_id), max(left_id, right_id)))
    if relation is None or not relation.is_connected:
        return -1e9
    upload_proximity = math.exp(-abs(positions.get(left_id, 0) - positions.get(right_id, 0)) / 8.0)
    camera_motion = math.hypot(relation.direction_dx, relation.direction_dy)
    smoothness = math.exp(-max(camera_motion - 160.0, 0.0) / 160.0)
    return (
        0.35 * relation.relation_confidence
        + 0.25 * relation.overlap_score
        + 0.20 * relation.reprojection_score
        + 0.10 * smoothness
        + 0.10 * upload_proximity
    )


def _order_component(
    component: list[int],
    relations: list[PhotoRelationResult],
    positions: dict[int, int],
) -> list[int]:
    if len(component) <= 1:
        return list(component)
    relation_by_pair = _relation_lookup(relations)
    states: list[tuple[float, list[int]]] = [(0.0, [photo_id]) for photo_id in component]
    for _ in range(1, len(component)):
        next_states: list[tuple[float, list[int]]] = []
        for score, path in states:
            for candidate in component:
                if candidate in path:
                    continue
                edge = _edge_score(path[-1], candidate, relation_by_pair, positions)
                if edge <= -1e8:
                    continue
                next_states.append((score + edge, path + [candidate]))
        if not next_states:
            break
        states = sorted(
            next_states,
            key=lambda state: (state[0], [-positions.get(photo_id, 0) for photo_id in state[1]]),
            reverse=True,
        )[:32]
    complete = [state for state in states if len(state[1]) == len(component)]
    if complete:
        return max(complete, key=lambda state: state[0])[1]
    return sorted(component, key=lambda photo_id: (positions.get(photo_id, 0), photo_id))


def _incident_records(evidence: list[dict[str, Any]], photo_id: int) -> list[dict[str, Any]]:
    return [
        record
        for record in evidence
        if int(record["photo_a_id"]) == photo_id or int(record["photo_b_id"]) == photo_id
    ]


def _photo_stats(
    photo_id: int,
    evidence: list[dict[str, Any]],
    relations: list[PhotoRelationResult],
) -> dict[str, float | int]:
    records = _incident_records(evidence, photo_id)
    raw_confidence: list[float] = []
    valid_fraction: list[float] = []
    directional_overlap: list[float] = []
    errors: list[float] = []
    for record in records:
        is_a = int(record["photo_a_id"]) == photo_id
        raw_confidence.append(float(record["conf_frame_a"] if is_a else record["conf_frame_b"]))
        valid_fraction.append(float(record["valid_fraction_a"] if is_a else record["valid_fraction_b"]))
        direction = record["forward"] if is_a else record["backward"]
        directional_overlap.append(float(direction["depth_ok"]))
        error = direction.get("median_relative_depth_error")
        if error is not None and math.isfinite(float(error)):
            errors.append(float(error))
    direct = [
        relation
        for relation in relations
        if relation.is_connected and photo_id in (relation.photo_a_id, relation.photo_b_id)
    ]
    depth_confidence = float(np.median([_confidence_fit(value) for value in raw_confidence])) if raw_confidence else 0.0
    direct_confidence = float(np.mean([relation.relation_confidence for relation in direct])) if direct else 0.0
    return {
        "pose_confidence": 0.60 * depth_confidence + 0.40 * direct_confidence,
        "depth_confidence": depth_confidence,
        "visibility_score": float(np.mean(directional_overlap)) if directional_overlap else 0.0,
        "reprojection_error": float(np.median(errors)) if errors else 0.0,
        "valid_fraction": float(np.mean(valid_fraction)) if valid_fraction else 1.0,
        "incident_pair_count": len(records),
        "direct_edge_count": len(direct),
    }


def _hero_photo(
    component: list[int],
    evidence: list[dict[str, Any]],
    relations: list[PhotoRelationResult],
    quality_scores: dict[int, float],
    room_labels: dict[int, str],
    editorial_roles: dict[int, str],
) -> int:
    forced = sorted(photo_id for photo_id in component if editorial_roles.get(photo_id) == "hero")
    if forced:
        return forced[0]

    def score(photo_id: int) -> tuple[float, int]:
        stats = _photo_stats(photo_id, evidence, relations)
        incident = _incident_records(evidence, photo_id)
        novelty = max(
            (min(abs(float(record["rot_deg"])) / 120.0, 1.0) for record in incident),
            default=0.0,
        )
        importance = 1.0 if any(
            token in str(room_labels.get(photo_id, "")).lower() for token in SOCIAL_LABEL_TOKENS
        ) else 0.45
        value = (
            0.30 * float(np.clip(quality_scores.get(photo_id, 0.0), 0.0, 1.0))
            + 0.25 * importance
            + 0.20 * float(stats["visibility_score"])
            + 0.15 * float(stats["pose_confidence"])
            + 0.10 * novelty
        )
        return value, -photo_id

    return max(component, key=score)


def _motion_affordance(component: list[int], relations: list[PhotoRelationResult]) -> str:
    if len(component) == 1:
        return "micro_push_in"
    safe = [
        relation
        for relation in relations
        if relation.is_connected
        and relation.continuity_type == "interpolation_safe"
        and relation.photo_a_id in component
        and relation.photo_b_id in component
    ]
    return "multi_view" if safe else "micro_push_in"


def _geometry_result(
    photo_id: int,
    component_key: str,
    pose: CameraPose,
    evidence: list[dict[str, Any]],
    relations: list[PhotoRelationResult],
    runtime: dict[str, Any],
) -> PhotoGeometryResult:
    stats = _photo_stats(photo_id, evidence, relations)
    return PhotoGeometryResult(
        photo_id=photo_id,
        pose_confidence=float(stats["pose_confidence"]),
        depth_confidence=float(stats["depth_confidence"]),
        visibility_score=float(stats["visibility_score"]),
        reprojection_error=float(stats["reprojection_error"]),
        camera_extrinsic=pose.extrinsic().tolist(),
        camera_center=pose.center.tolist(),
        view_direction=pose.view_direction.tolist(),
        local_metrics={
            **stats,
            "geometry_source": "vggt_pairwise_pose_graph",
            "coordinate_frame": component_key,
            "coordinate_scope": "component_local",
            "runtime": runtime,
        },
    )


def run_vggt_scene_pipeline(
    images: list[object],
    photo_ids: list[int],
    room_labels: list[str],
    positions: list[int],
    *,
    job_id: int,
    quality_scores: dict[int, float] | None = None,
    editorial_roles: dict[int, str] | None = None,
    embeddings: dict[int, list] | None = None,
) -> tuple[list[PhotoGeometryResult], list[PhotoRelationResult], list[SceneComponentResult]]:
    """Run the V2-only scene graph without whole-listing Omega inference."""
    if not images:
        return [], [], []
    if len(images) != len(photo_ids):
        raise ValueError("images and photo_ids must have the same length")

    quality_scores = quality_scores or {}
    editorial_roles = editorial_roles or {}
    labels = {
        int(photo_id): str(room_labels[index] if index < len(room_labels) else "")
        for index, photo_id in enumerate(photo_ids)
    }
    position_map = {
        int(photo_id): int(positions[index] if index < len(positions) else index)
        for index, photo_id in enumerate(photo_ids)
    }
    logger.info("VGGT_V2_PIPELINE_START job_id=%s photos=%s", job_id, len(images))
    started = time.monotonic()
    directory, filelist = _save_input_images(images, photo_ids)
    try:
        runtime = vggt_model.runtime_metadata()
        file_by_photo = {int(photo_id): filelist[index] for index, photo_id in enumerate(photo_ids)}
        relations, components, evidence, graph_stats = _v2_scene_graph(
            file_by_photo,
            labels,
            position_map,
            embeddings or {},
            runtime=runtime,
        )
        runtime = {
            **runtime,
            **graph_stats,
            "image_count": len(images),
            "runtime_seconds": time.monotonic() - started,
            "planner": "scene_graph_v2_pairwise_only",
        }

        geometry_by_photo: dict[int, PhotoGeometryResult] = {}
        results: list[SceneComponentResult] = []
        for index, component in enumerate(components):
            component_key = f"component-{index + 1}"
            ordered = _order_component(component, relations, position_map)
            internal = [
                relation
                for relation in relations
                if relation.is_connected
                and relation.photo_a_id in component
                and relation.photo_b_id in component
            ]
            poses, pose_stats = solve_component_poses(component, relations)
            for photo_id in component:
                geometry_by_photo[photo_id] = _geometry_result(
                    photo_id,
                    component_key,
                    poses[photo_id],
                    evidence,
                    relations,
                    runtime,
                )
            component_geometries = [geometry_by_photo[photo_id] for photo_id in component]
            results.append(
                SceneComponentResult(
                    component_key=component_key,
                    photo_ids=list(component),
                    ordered_photo_ids=ordered,
                    scene_type=_scene_type(component, labels),
                    geometry_confidence=float(
                        np.mean([geometry.pose_confidence for geometry in component_geometries])
                    ),
                    connectivity_confidence=(
                        float(np.mean([relation.relation_confidence for relation in internal]))
                        if internal
                        else 0.0
                    ),
                    avg_reprojection_error=(
                        float(np.mean([1.0 - relation.reprojection_score for relation in internal]))
                        if internal
                        else 0.0
                    ),
                    hero_photo_id=_hero_photo(
                        component,
                        evidence,
                        relations,
                        quality_scores,
                        labels,
                        editorial_roles,
                    ),
                    motion_affordance=_motion_affordance(component, relations),
                    debug_metrics={
                        "runtime": runtime,
                        "photo_ids": list(component),
                        "ordered_photo_ids": ordered,
                        "verified_edge_count": len(internal),
                        "coordinate_frame": component_key,
                        "pose_graph": pose_stats,
                    },
                )
            )

        geometries = [geometry_by_photo[int(photo_id)] for photo_id in photo_ids]
        logger.info(
            "VGGT_V2_PIPELINE_COMPLETE job_id=%s geometries=%s relations=%s components=%s runtime_seconds=%.3f",
            job_id,
            len(geometries),
            len(relations),
            len(results),
            time.monotonic() - started,
        )
        return geometries, relations, results
    finally:
        _cleanup_files(directory, filelist)
        _release_accelerator_cache()
