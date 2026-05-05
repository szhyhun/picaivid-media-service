"""VGGT scene reconstruction and relation graph helpers."""
from __future__ import annotations

import io
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from PIL import Image

from app.core.config import settings
from app.models.vggt import vggt_model
from app.services.storage.s3_client import S3Client

logger = logging.getLogger(__name__)

BRIDGE_LABEL_TOKENS = (
    "hall",
    "hallway",
    "entry",
    "entryway",
    "foyer",
    "landing",
    "stair",
    "door",
    "patio",
    "deck",
    "balcony",
    "porch",
    "terrace",
    "walkway",
)


@dataclass
class PhotoGeometryResult:
    photo_id: int
    pose_confidence: float
    depth_confidence: float
    point_confidence: float
    visibility_score: float
    reprojection_error: float
    camera_extrinsic: list[list[float]]
    camera_intrinsic: list[list[float]]
    camera_center: list[float]
    view_direction: list[float]
    depth_artifact_uri: str | None
    point_map_artifact_uri: str | None
    local_metrics: dict


@dataclass
class PhotoRelationResult:
    photo_a_id: int
    photo_b_id: int
    overlap_score: float
    track_support: float
    reprojection_score: float
    relation_confidence: float
    baseline_distance: float
    relative_transform: dict
    direction_dx: float
    direction_dy: float
    continuity_type: str
    is_bridge_edge: bool
    is_connected: bool
    debug_metrics: dict


@dataclass
class SceneComponentResult:
    component_key: str
    photo_ids: list[int]
    ordered_photo_ids: list[int]
    scene_type: str
    geometry_confidence: float
    connectivity_confidence: float
    track_coverage: float
    avg_reprojection_error: float
    hero_photo_id: int
    depth_range: float
    motion_affordance: str
    debug_metrics: dict


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return np.zeros_like(vec)
    return vec / norm


def _camera_center_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    world_from_camera = np.linalg.inv(extrinsic)
    return world_from_camera[:3, 3]


def _view_direction_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    world_from_camera = np.linalg.inv(extrinsic)
    return _normalize(world_from_camera[:3, 2])


def _artifact_uri(s3_client: S3Client | None, key: str, array: np.ndarray) -> str | None:
    if s3_client is None:
        return None
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=array)
    return s3_client.upload_bytes(key, buffer.getvalue(), content_type="application/octet-stream")


def _save_input_images(images: list[object], photo_ids: list[int]) -> tuple[str, list[str]]:
    tmpdir = tempfile.mkdtemp(prefix="vggt_scene_")
    filelist: list[str] = []
    for idx, (image, photo_id) in enumerate(zip(images, photo_ids)):
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            raise RuntimeError(f"Unsupported image type for VGGT: {type(image)!r}")
        path = os.path.join(tmpdir, f"{idx:03d}_{int(photo_id)}.png")
        pil_image.convert("RGB").save(path)
        filelist.append(path)
    return tmpdir, filelist


def _scene_type_for_label(label: str | None) -> str:
    text = str(label or "").strip().lower()
    if "drone" in text or "aerial" in text:
        return "drone"
    if any(token in text for token in ("exterior", "front", "backyard", "yard", "patio", "deck", "pool")):
        return "exterior"
    if not text:
        return "mixed"
    return "interior"


def _is_bridge_label(label: str | None) -> bool:
    text = str(label or "").strip().lower().replace("_", " ")
    return any(token in text for token in BRIDGE_LABEL_TOKENS)


def _domain_penalty(left_domain: str, right_domain: str) -> float:
    domain_pair = {left_domain, right_domain}
    if len(domain_pair) <= 1:
        return 0.0
    if domain_pair == {"interior", "exterior"}:
        return float(settings.VGGT_INTERIOR_EXTERIOR_PENALTY)
    if domain_pair == {"interior", "drone"}:
        return float(settings.VGGT_INTERIOR_DRONE_PENALTY)
    if domain_pair == {"exterior", "drone"}:
        return float(settings.VGGT_EXTERIOR_DRONE_PENALTY)
    return 0.0


def _synthetic_vggt_outputs(
    photo_ids: list[int],
    room_labels: list[str],
    positions: list[int],
) -> tuple[list[PhotoGeometryResult], list[dict[str, object]]]:
    geometries: list[PhotoGeometryResult] = []
    labels = [_scene_type_for_label(label) for label in room_labels]
    for idx, photo_id in enumerate(photo_ids):
        position = float(positions[idx] if idx < len(positions) else idx)
        scene_type = labels[idx] if idx < len(labels) else "interior"
        z_bias = 1.5 if scene_type == "interior" else 4.0
        center = np.array([position * 0.65, float(idx % 3) * 0.25, z_bias], dtype=float)
        target = np.array([center[0] + 0.8, center[1], center[2] + (0.2 if scene_type == "exterior" else -0.1)])
        forward = _normalize(target - center)
        extrinsic = np.eye(4, dtype=float)
        extrinsic[:3, 3] = -center
        intrinsic = np.array([[900.0, 0.0, 512.0], [0.0, 900.0, 512.0], [0.0, 0.0, 1.0]], dtype=float)
        depth_conf = 0.68 if scene_type == "interior" else 0.62
        point_conf = 0.64 if scene_type == "interior" else 0.60
        geometries.append(
            PhotoGeometryResult(
                photo_id=int(photo_id),
                pose_confidence=0.58 + 0.02 * (idx % 5),
                depth_confidence=depth_conf,
                point_confidence=point_conf,
                visibility_score=0.55 + 0.03 * (idx % 4),
                reprojection_error=0.16 + 0.01 * (idx % 5),
                camera_extrinsic=extrinsic.tolist(),
                camera_intrinsic=intrinsic.tolist(),
                camera_center=center.tolist(),
                view_direction=forward.tolist(),
                depth_artifact_uri=None,
                point_map_artifact_uri=None,
                local_metrics={"fallback": True, "scene_type": scene_type},
            )
        )
    return geometries, []


def _compute_pair_relations(
    geometries: list[PhotoGeometryResult],
    room_labels: dict[int, str],
    positions: dict[int, int],
) -> list[PhotoRelationResult]:
    if not geometries:
        return []
    centers = {g.photo_id: np.array(g.camera_center, dtype=float) for g in geometries}
    directions = {g.photo_id: _normalize(np.array(g.view_direction, dtype=float)) for g in geometries}
    all_distances = []
    for i in range(len(geometries)):
        for j in range(i + 1, len(geometries)):
            all_distances.append(float(np.linalg.norm(centers[geometries[i].photo_id] - centers[geometries[j].photo_id])))
    scene_scale = float(np.median(all_distances)) if all_distances else 1.0
    scene_scale = max(scene_scale, 0.5)

    relations: list[PhotoRelationResult] = []
    for i in range(len(geometries)):
        for j in range(i + 1, len(geometries)):
            left = geometries[i]
            right = geometries[j]
            left_center = centers[left.photo_id]
            right_center = centers[right.photo_id]
            delta = right_center - left_center
            baseline = float(np.linalg.norm(delta))
            normalized_baseline = baseline / scene_scale
            baseline_score = float(math.exp(-abs(normalized_baseline - 1.0)))
            view_alignment = float(np.clip(np.dot(directions[left.photo_id], directions[right.photo_id]), 0.0, 1.0))
            room_match = 1.0 if room_labels.get(left.photo_id) == room_labels.get(right.photo_id) else 0.0
            position_gap = abs(int(positions.get(left.photo_id, 0)) - int(positions.get(right.photo_id, 0)))
            position_bonus = float(math.exp(-position_gap / 4.0))
            left_domain = _scene_type_for_label(room_labels.get(left.photo_id))
            right_domain = _scene_type_for_label(room_labels.get(right.photo_id))
            domain_penalty = _domain_penalty(left_domain, right_domain)
            bridge_candidate = (
                (_is_bridge_label(room_labels.get(left.photo_id)) or _is_bridge_label(room_labels.get(right.photo_id)))
                and position_gap <= int(settings.VGGT_BRIDGE_POSITION_GAP_MAX)
            )
            overlap_score = float(np.clip(
                0.48 * view_alignment + 0.28 * baseline_score + 0.14 * position_bonus + 0.10 * room_match - domain_penalty,
                0.0,
                1.0,
            ))
            track_support = float(np.clip(
                0.55 * overlap_score + 0.45 * min(left.point_confidence, right.point_confidence) - 0.5 * domain_penalty,
                0.0,
                1.0,
            ))
            reprojection_score = float(np.clip(1.0 - 0.5 * (left.reprojection_error + right.reprojection_error), 0.0, 1.0))
            relation_confidence = float(np.clip(
                0.40 * overlap_score
                + 0.30 * track_support
                + 0.20 * reprojection_score
                + 0.10 * room_match
                + (float(settings.VGGT_CROSS_DOMAIN_CONFIDENCE_BONUS) if bridge_candidate else 0.0),
                0.0,
                1.0,
            ))
            continuity_type = "weak"
            if room_labels.get(left.photo_id) == room_labels.get(right.photo_id):
                continuity_type = "same_room"
            elif left_domain == right_domain == "exterior":
                continuity_type = "exterior"
            elif left_domain == right_domain == "drone":
                continuity_type = "drone"
            elif bridge_candidate and room_labels.get(left.photo_id) and room_labels.get(right.photo_id):
                continuity_type = "doorway"
            elif left_domain != right_domain:
                continuity_type = "cross_domain"
            is_bridge_edge = continuity_type == "doorway" and relation_confidence >= float(settings.VGGT_BRIDGE_SCORE_THRESHOLD)
            requires_bridge = "interior" in {left_domain, right_domain} and left_domain != right_domain
            threshold = float(settings.VGGT_RELATION_SCORE_THRESHOLD)
            if requires_bridge:
                threshold = max(threshold, float(settings.VGGT_MIXED_COMPONENT_SPLIT_THRESHOLD))
            elif left_domain != right_domain:
                threshold = max(threshold, float(settings.VGGT_RELATION_SCORE_THRESHOLD) + 0.04)
            is_connected = relation_confidence >= threshold and (bridge_candidate if requires_bridge else True)
            relations.append(
                PhotoRelationResult(
                    photo_a_id=int(left.photo_id),
                    photo_b_id=int(right.photo_id),
                    overlap_score=overlap_score,
                    track_support=track_support,
                    reprojection_score=reprojection_score,
                    relation_confidence=relation_confidence,
                    baseline_distance=baseline,
                    relative_transform={
                        "translation": [float(v) for v in delta.tolist()],
                        "distance": baseline,
                    },
                    direction_dx=float(delta[0]),
                    direction_dy=float(delta[1]),
                    continuity_type=continuity_type,
                    is_bridge_edge=is_bridge_edge,
                    is_connected=is_connected,
                    debug_metrics={
                        "view_alignment": view_alignment,
                        "baseline_score": baseline_score,
                        "position_gap": position_gap,
                        "room_match": room_match,
                        "left_domain": left_domain,
                        "right_domain": right_domain,
                        "domain_penalty": domain_penalty,
                        "bridge_candidate": bridge_candidate,
                    },
                )
            )
    return relations


def _connected_components(photo_ids: Iterable[int], relations: list[PhotoRelationResult]) -> list[list[int]]:
    adjacency: dict[int, set[int]] = {int(photo_id): set() for photo_id in photo_ids}
    for relation in relations:
        if not relation.is_connected:
            continue
        adjacency.setdefault(int(relation.photo_a_id), set()).add(int(relation.photo_b_id))
        adjacency.setdefault(int(relation.photo_b_id), set()).add(int(relation.photo_a_id))
    components: list[list[int]] = []
    seen: set[int] = set()
    for photo_id in adjacency:
        if photo_id in seen:
            continue
        stack = [photo_id]
        component: list[int] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            stack.extend(sorted(adjacency.get(current, set()) - seen))
        components.append(sorted(component))
    return components


def _relation_lookup(relations: list[PhotoRelationResult]) -> dict[tuple[int, int], PhotoRelationResult]:
    return {
        (min(int(relation.photo_a_id), int(relation.photo_b_id)), max(int(relation.photo_a_id), int(relation.photo_b_id))): relation
        for relation in relations
    }


def _component_internal_scores(
    component_photo_ids: list[int],
    relation_by_pair: dict[tuple[int, int], PhotoRelationResult],
) -> dict[int, tuple[int, float]]:
    scores: dict[int, list[float]] = {int(photo_id): [] for photo_id in component_photo_ids}
    for idx, left_photo_id in enumerate(component_photo_ids):
        for right_photo_id in component_photo_ids[idx + 1:]:
            relation = relation_by_pair.get((min(int(left_photo_id), int(right_photo_id)), max(int(left_photo_id), int(right_photo_id))))
            if relation is None or not relation.is_connected:
                continue
            scores[int(left_photo_id)].append(float(relation.relation_confidence))
            scores[int(right_photo_id)].append(float(relation.relation_confidence))
    return {
        photo_id: (len(values), float(np.mean(values)) if values else 0.0)
        for photo_id, values in scores.items()
    }


def _split_mixed_component(
    component_photo_ids: list[int],
    relation_by_pair: dict[tuple[int, int], PhotoRelationResult],
    room_labels: dict[int, str],
) -> list[list[int]]:
    domains = {_scene_type_for_label(room_labels.get(photo_id)) for photo_id in component_photo_ids}
    if len(component_photo_ids) <= 2 or len(domains) <= 1:
        return [sorted(component_photo_ids)]

    cross_domain_relations = []
    for idx, left_photo_id in enumerate(component_photo_ids):
        for right_photo_id in component_photo_ids[idx + 1:]:
            relation = relation_by_pair.get((min(int(left_photo_id), int(right_photo_id)), max(int(left_photo_id), int(right_photo_id))))
            if relation is None or not relation.is_connected:
                continue
            left_domain = _scene_type_for_label(room_labels.get(left_photo_id))
            right_domain = _scene_type_for_label(room_labels.get(right_photo_id))
            if left_domain != right_domain:
                cross_domain_relations.append(relation)

    if not cross_domain_relations:
        return [sorted(component_photo_ids)]

    avg_cross_domain_confidence = float(np.mean([relation.relation_confidence for relation in cross_domain_relations]))
    strong_bridge_count = sum(1 for relation in cross_domain_relations if relation.is_bridge_edge)
    bridge_photos = {
        photo_id
        for relation in cross_domain_relations
        if relation.is_bridge_edge
        for photo_id in (int(relation.photo_a_id), int(relation.photo_b_id))
        if _is_bridge_label(room_labels.get(photo_id))
    }
    keep_cross_domain_edges = (
        strong_bridge_count >= 2
        and len(bridge_photos) >= 2
        and avg_cross_domain_confidence >= float(settings.VGGT_MIXED_COMPONENT_SPLIT_THRESHOLD) + 0.04
    )
    if keep_cross_domain_edges:
        return [sorted(component_photo_ids)]

    adjacency: dict[int, set[int]] = {int(photo_id): set() for photo_id in component_photo_ids}
    for idx, left_photo_id in enumerate(component_photo_ids):
        for right_photo_id in component_photo_ids[idx + 1:]:
            relation = relation_by_pair.get((min(int(left_photo_id), int(right_photo_id)), max(int(left_photo_id), int(right_photo_id))))
            if relation is None or not relation.is_connected:
                continue
            left_domain = _scene_type_for_label(room_labels.get(left_photo_id))
            right_domain = _scene_type_for_label(room_labels.get(right_photo_id))
            if left_domain != right_domain and not keep_cross_domain_edges:
                continue
            if left_domain != right_domain and not relation.is_bridge_edge:
                continue
            adjacency[int(left_photo_id)].add(int(right_photo_id))
            adjacency[int(right_photo_id)].add(int(left_photo_id))

    subcomponents: list[list[int]] = []
    seen: set[int] = set()
    for photo_id in sorted(adjacency):
        if photo_id in seen:
            continue
        stack = [photo_id]
        current_component: list[int] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            current_component.append(current)
            stack.extend(sorted(adjacency.get(current, set()) - seen))
        subcomponents.append(sorted(current_component))
    return subcomponents


def _prune_outliers(
    component_photo_ids: list[int],
    relation_by_pair: dict[tuple[int, int], PhotoRelationResult],
    room_labels: dict[int, str],
) -> list[list[int]]:
    if len(component_photo_ids) <= 2:
        return [sorted(component_photo_ids)]

    kept = sorted(component_photo_ids)
    peeled: list[int] = []
    while len(kept) > 2:
        internal_scores = _component_internal_scores(kept, relation_by_pair)
        weak_photo_id = None
        weak_score = None
        dominant_domain = max(
            {_scene_type_for_label(room_labels.get(photo_id)) for photo_id in kept},
            key=lambda domain: sum(1 for photo_id in kept if _scene_type_for_label(room_labels.get(photo_id)) == domain),
        )
        for photo_id in kept:
            degree, avg_confidence = internal_scores.get(int(photo_id), (0, 0.0))
            domain = _scene_type_for_label(room_labels.get(photo_id))
            bridge = _is_bridge_label(room_labels.get(photo_id))
            should_remove = degree == 0 or avg_confidence < float(settings.VGGT_OUTLIER_CONFIDENCE_THRESHOLD)
            if domain != dominant_domain and not bridge and avg_confidence < float(settings.VGGT_MIXED_COMPONENT_SPLIT_THRESHOLD):
                should_remove = True
            if should_remove and (weak_score is None or avg_confidence < weak_score):
                weak_photo_id = int(photo_id)
                weak_score = avg_confidence
        if weak_photo_id is None:
            break
        kept = [photo_id for photo_id in kept if int(photo_id) != weak_photo_id]
        peeled.append(weak_photo_id)

    components = [sorted(kept)] if kept else []
    components.extend([[int(photo_id)] for photo_id in peeled])
    return components


def _refine_components(
    photo_ids: list[int],
    relations: list[PhotoRelationResult],
    room_labels: dict[int, str],
) -> list[list[int]]:
    relation_by_pair = _relation_lookup(relations)
    initial_components = _connected_components(photo_ids, relations)
    refined: list[list[int]] = []
    for component_photo_ids in initial_components:
        for split_component in _split_mixed_component(component_photo_ids, relation_by_pair, room_labels):
            refined.extend(_prune_outliers(split_component, relation_by_pair, room_labels))
    return [sorted(component) for component in refined if component]


def _order_component(component_photo_ids: list[int], geometries_by_photo: dict[int, PhotoGeometryResult], positions: dict[int, int]) -> list[int]:
    if len(component_photo_ids) <= 1:
        return component_photo_ids
    centers = np.array([geometries_by_photo[photo_id].camera_center for photo_id in component_photo_ids], dtype=float)
    centered = centers - centers.mean(axis=0, keepdims=True)
    if np.linalg.norm(centered) <= 1e-8:
        return sorted(component_photo_ids, key=lambda photo_id: positions.get(photo_id, 0))
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    projections = centered @ axis
    order_pairs = sorted(
        zip(component_photo_ids, projections.tolist()),
        key=lambda item: (item[1], positions.get(int(item[0]), 0)),
    )
    return [int(photo_id) for photo_id, _ in order_pairs]


def _motion_affordance(photo_ids: list[int], geometries_by_photo: dict[int, PhotoGeometryResult], relations: list[PhotoRelationResult]) -> tuple[str, float, float, float]:
    if len(photo_ids) <= 1:
        return "static", 0.0, 0.0, 0.0
    component_relations = [
        relation
        for relation in relations
        if relation.photo_a_id in photo_ids and relation.photo_b_id in photo_ids
    ]
    avg_relation = float(np.mean([relation.relation_confidence for relation in component_relations])) if component_relations else 0.0
    avg_track = float(np.mean([relation.track_support for relation in component_relations])) if component_relations else 0.0
    baselines = [relation.baseline_distance for relation in component_relations]
    baseline_spread = float(np.mean(baselines)) if baselines else 0.0
    if avg_relation >= 0.74 and avg_track >= 0.62 and baseline_spread >= 1.4:
        return "multi_view", avg_relation, avg_track, baseline_spread
    if avg_relation >= 0.64 and avg_track >= 0.52:
        return "parallax", avg_relation, avg_track, baseline_spread
    if avg_relation >= 0.52:
        return "reveal", avg_relation, avg_track, baseline_spread
    return "micro_push_in", avg_relation, avg_track, baseline_spread


def _scene_type(component_photo_ids: list[int], room_labels: dict[int, str]) -> str:
    component_labels = [_scene_type_for_label(room_labels.get(photo_id)) for photo_id in component_photo_ids]
    label_counts: dict[str, int] = {}
    for label in component_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    top_label, top_count = max(label_counts.items(), key=lambda item: item[1])
    if top_count != len(component_labels) and len(label_counts) > 1:
        return "mixed"
    return top_label


def run_vggt_scene_pipeline(
    images: list[object],
    photo_ids: list[int],
    room_labels: list[str],
    positions: list[int],
    *,
    job_id: int,
    s3_client: S3Client | None = None,
) -> tuple[list[PhotoGeometryResult], list[PhotoRelationResult], list[SceneComponentResult]]:
    try:
        tmpdir, filelist = _save_input_images(images, photo_ids)
        try:
            predictions = vggt_model.predict(filelist)
        finally:
            for path in filelist:
                try:
                    os.remove(path)
                except OSError:
                    pass
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass

        extrinsics = predictions["extrinsic"].numpy()
        intrinsics = predictions["intrinsic"].numpy()
        depth_maps = predictions["depth_map"].numpy()
        depth_conf = predictions["depth_conf"].numpy()
        point_maps = predictions["point_map_unprojected"].numpy()
        point_conf = predictions["point_conf"].numpy()

        geometries: list[PhotoGeometryResult] = []
        for idx, photo_id in enumerate(photo_ids):
            extrinsic = np.asarray(extrinsics[idx], dtype=float)
            intrinsic = np.asarray(intrinsics[idx], dtype=float)
            center = _camera_center_from_extrinsic(extrinsic)
            view_direction = _view_direction_from_extrinsic(extrinsic)
            depth_uri = _artifact_uri(
                s3_client,
                f"jobs/{job_id}/geometry/photo_{photo_id}_depth.npz",
                np.asarray(depth_maps[idx]),
            )
            point_uri = _artifact_uri(
                s3_client,
                f"jobs/{job_id}/geometry/photo_{photo_id}_pointmap.npz",
                np.asarray(point_maps[idx]),
            )
            geometries.append(
                PhotoGeometryResult(
                    photo_id=int(photo_id),
                    pose_confidence=float(np.clip(np.mean(depth_conf[idx]) * np.mean(point_conf[idx]), 0.0, 1.0)),
                    depth_confidence=float(np.clip(np.mean(depth_conf[idx]), 0.0, 1.0)),
                    point_confidence=float(np.clip(np.mean(point_conf[idx]), 0.0, 1.0)),
                    visibility_score=float(np.clip(np.percentile(point_conf[idx], 75), 0.0, 1.0)),
                    reprojection_error=float(max(0.0, 1.0 - np.mean(point_conf[idx]))),
                    camera_extrinsic=extrinsic.tolist(),
                    camera_intrinsic=intrinsic.tolist(),
                    camera_center=center.tolist(),
                    view_direction=view_direction.tolist(),
                    depth_artifact_uri=depth_uri,
                    point_map_artifact_uri=point_uri,
                    local_metrics={
                        "depth_mean": float(np.mean(depth_maps[idx])),
                        "depth_std": float(np.std(depth_maps[idx])),
                        "point_conf_p90": float(np.percentile(point_conf[idx], 90)),
                    },
                )
            )
    except Exception as err:
        logger.warning("VGGT runtime unavailable, using synthetic scene fallback: %s", err)
        geometries, _ = _synthetic_vggt_outputs(photo_ids, room_labels, positions)

    room_label_map = {int(photo_id): str(room_labels[idx] if idx < len(room_labels) else "") for idx, photo_id in enumerate(photo_ids)}
    position_map = {int(photo_id): int(positions[idx] if idx < len(positions) else idx) for idx, photo_id in enumerate(photo_ids)}
    geometries_by_photo = {geometry.photo_id: geometry for geometry in geometries}
    relations = _compute_pair_relations(geometries, room_label_map, position_map)
    components = _refine_components(photo_ids, relations, room_label_map)

    component_results: list[SceneComponentResult] = []
    for index, component_photo_ids in enumerate(components):
        ordered_photo_ids = _order_component(component_photo_ids, geometries_by_photo, position_map)
        scene_type = _scene_type(component_photo_ids, room_label_map)
        component_geometries = [geometries_by_photo[photo_id] for photo_id in component_photo_ids]
        component_relations = [
            relation
            for relation in relations
            if relation.photo_a_id in component_photo_ids and relation.photo_b_id in component_photo_ids
        ]
        geometry_confidence = float(np.mean([geometry.pose_confidence for geometry in component_geometries])) if component_geometries else 0.0
        connectivity_confidence = float(np.mean([relation.relation_confidence for relation in component_relations])) if component_relations else geometry_confidence
        track_coverage = float(np.mean([relation.track_support for relation in component_relations])) if component_relations else 0.0
        avg_reprojection_error = float(np.mean([geometry.reprojection_error for geometry in component_geometries])) if component_geometries else 0.0
        depth_means = [float(geometry.local_metrics.get("depth_mean", 0.0)) for geometry in component_geometries]
        depth_range = float(max(depth_means) - min(depth_means)) if depth_means else 0.0
        motion_affordance, avg_relation, avg_track, baseline_spread = _motion_affordance(component_photo_ids, geometries_by_photo, relations)
        hero_photo = max(component_geometries, key=lambda geometry: (geometry.pose_confidence, geometry.depth_confidence)).photo_id
        bridge_count = sum(1 for relation in component_relations if relation.is_bridge_edge)
        domain_counts: dict[str, int] = {}
        for photo_id in component_photo_ids:
            domain = _scene_type_for_label(room_label_map.get(photo_id))
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        component_results.append(
            SceneComponentResult(
                component_key=f"component-{index + 1}",
                photo_ids=[int(photo_id) for photo_id in component_photo_ids],
                ordered_photo_ids=[int(photo_id) for photo_id in ordered_photo_ids],
                scene_type=scene_type,
                geometry_confidence=geometry_confidence,
                connectivity_confidence=connectivity_confidence,
                track_coverage=track_coverage,
                avg_reprojection_error=avg_reprojection_error,
                hero_photo_id=int(hero_photo),
                depth_range=depth_range,
                motion_affordance=motion_affordance,
                debug_metrics={
                    "avg_relation_confidence": avg_relation,
                    "avg_track_support": avg_track,
                    "baseline_spread": baseline_spread,
                    "bridge_edge_count": bridge_count,
                    "domain_counts": domain_counts,
                    "photo_ids": [int(photo_id) for photo_id in component_photo_ids],
                },
            )
        )

    return geometries, relations, component_results
