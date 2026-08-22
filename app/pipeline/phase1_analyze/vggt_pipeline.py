"""Geometry-first VGGT reconstruction, relation verification, and component ordering."""
from __future__ import annotations

import gc
import io
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from app.core.config import settings
from app.models.vggt import vggt_model
from app.services.storage.s3_client import S3Client

logger = logging.getLogger(__name__)

SOCIAL_LABEL_TOKENS = ("living", "kitchen", "dining", "great room", "family room")


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


@dataclass
class _PhotoArrays:
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    depth: np.ndarray
    depth_conf: np.ndarray
    point_map: np.ndarray
    point_conf: np.ndarray
    world_points: np.ndarray


def _normalize(vector: np.ndarray) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    return vector / magnitude if magnitude > 1e-8 else np.zeros_like(vector)


def _confidence_unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if float(np.nanmax(values, initial=0.0)) <= 1.0:
        return np.clip(values, 0.0, 1.0)
    return np.clip(1.0 - np.exp(-(values - 1.0)), 0.0, 1.0)


def _camera_center(extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:3, :3]
    translation = extrinsic[:3, 3]
    return -rotation.T @ translation


def _view_direction(extrinsic: np.ndarray) -> np.ndarray:
    return _normalize(np.linalg.inv(np.vstack([extrinsic, [0.0, 0.0, 0.0, 1.0]]))[:3, 2])


def _artifact_uri(s3_client: S3Client | None, key: str, array: np.ndarray) -> str | None:
    if s3_client is None:
        return None
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=array)
    return s3_client.upload_bytes(key, buffer.getvalue(), content_type="application/octet-stream")


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


def _scene_type_for_label(label: str | None) -> str:
    text = str(label or "").lower()
    if "drone" in text or "aerial" in text:
        return "drone"
    if any(token in text for token in ("exterior", "front", "backyard", "yard", "patio", "deck", "pool")):
        return "exterior"
    return "interior" if text else "mixed"


def _is_oom(error: Exception) -> bool:
    return "out of memory" in str(error).lower() or "mps backend out of memory" in str(error).lower()


def _release_accelerator_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


def _window_ranges(count: int, size: int, overlap: int) -> list[tuple[int, int]]:
    if size <= overlap:
        raise ValueError("VGGT window size must exceed overlap")
    if count <= size:
        return [(0, count)]
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < count:
        end = min(start + size, count)
        ranges.append((start, end))
        if end == count:
            break
        start = end - overlap
    return ranges


def _as_numpy_predictions(predictions: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"runtime": dict(predictions["runtime"])}
    for key in ("extrinsic", "intrinsic", "depth_map", "depth_conf", "point_map", "point_conf", "point_map_unprojected"):
        value = predictions[key]
        result[key] = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    return result


def _estimate_similarity(local_points: np.ndarray, global_points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    local = np.asarray(local_points, dtype=np.float64)
    target = np.asarray(global_points, dtype=np.float64)
    if len(local) < 3:
        raise RuntimeError("Window stitching requires at least three shared points")
    local_center = local.mean(axis=0)
    target_center = target.mean(axis=0)
    local_zero = local - local_center
    target_zero = target - target_center
    covariance = (target_zero.T @ local_zero) / len(local)
    left, singular, right = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right) < 0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right
    variance = float(np.mean(np.sum(local_zero**2, axis=1)))
    if variance <= 1e-10:
        raise RuntimeError("Window stitching received degenerate shared points")
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = target_center - scale * rotation @ local_center
    return scale, rotation, translation


def _apply_similarity_to_window(predictions: dict[str, Any], scale: float, rotation: np.ndarray, translation: np.ndarray) -> dict[str, Any]:
    transformed = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in predictions.items()}
    for key in ("point_map", "point_map_unprojected"):
        points = transformed[key]
        transformed[key] = (scale * (points @ rotation.T) + translation).astype(np.float32)
    transformed["depth_map"] = (transformed["depth_map"] * scale).astype(np.float32)
    extrinsics = transformed["extrinsic"].copy()
    for index, extrinsic in enumerate(extrinsics):
        center = _camera_center(extrinsic)
        camera_rotation = extrinsic[:3, :3] @ rotation.T
        transformed_center = scale * rotation @ center + translation
        extrinsics[index, :3, :3] = camera_rotation
        extrinsics[index, :3, 3] = -camera_rotation @ transformed_center
    transformed["extrinsic"] = extrinsics
    return transformed


def _shared_correspondences(
    local_predictions: dict[str, Any],
    global_predictions: dict[str, Any],
    local_indices: list[int],
    global_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    source_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    for local_index, global_index in zip(local_indices, global_indices):
        source = local_predictions["point_map_unprojected"][local_index].reshape(-1, 3)
        target = global_predictions["point_map_unprojected"][global_index].reshape(-1, 3)
        source_conf = _confidence_unit(local_predictions["point_conf"][local_index]).reshape(-1)
        target_conf = _confidence_unit(global_predictions["point_conf"][global_index]).reshape(-1)
        confidence = np.minimum(source_conf, target_conf)
        valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1) & (confidence >= np.quantile(confidence, 0.75))
        chosen = np.flatnonzero(valid)
        if len(chosen) > 512:
            chosen = chosen[np.linspace(0, len(chosen) - 1, 512, dtype=int)]
        source_chunks.append(source[chosen])
        target_chunks.append(target[chosen])
    if not source_chunks:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    return np.concatenate(source_chunks, axis=0), np.concatenate(target_chunks, axis=0)


def _merge_window_predictions(filelist: list[str], windows: list[tuple[int, int]]) -> tuple[dict[str, Any], dict[str, Any]]:
    merged: dict[str, Any] | None = None
    filled: set[int] = set()
    strategy = {"kind": "windowed", "windows": windows, "stitching": []}
    for start, end in windows:
        current = _as_numpy_predictions(vggt_model.predict(filelist[start:end]))
        if merged is None:
            merged = {
                key: np.zeros((len(filelist), *value.shape[1:]), dtype=value.dtype)
                for key, value in current.items()
                if isinstance(value, np.ndarray)
            }
            for key, value in current.items():
                if isinstance(value, np.ndarray):
                    merged[key][start:end] = value
            merged["runtime"] = current["runtime"]
            filled.update(range(start, end))
            continue
        shared_global = [index for index in range(start, end) if index in filled]
        shared_local = [index - start for index in shared_global]
        source, target = _shared_correspondences(current, merged, shared_local, shared_global)
        point_count = min(len(source), len(target))
        scale, rotation, translation = _estimate_similarity(source[:point_count], target[:point_count])
        current = _apply_similarity_to_window(current, scale, rotation, translation)
        residual = float(np.median(np.linalg.norm(scale * (source[:point_count] @ rotation.T) + translation - target[:point_count], axis=1)))
        strategy["stitching"].append({"window": [start, end], "shared_frames": len(shared_global), "scale": scale, "residual": residual})
        for key, value in current.items():
            if not isinstance(value, np.ndarray):
                continue
            for local_index, global_index in enumerate(range(start, end)):
                if global_index not in filled:
                    merged[key][global_index] = value[local_index]
        filled.update(range(start, end))
    if merged is None:
        raise RuntimeError("VGGT received no windows")
    return merged, strategy


def _predict_with_retries(filelist: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        prediction = _as_numpy_predictions(vggt_model.predict(filelist))
        return prediction, {"kind": "global", "windows": [[0, len(filelist)]]}
    except Exception as error:
        if not _is_oom(error):
            raise
        _release_accelerator_cache()
        logger.warning("VGGT global inference ran out of memory; retrying in windows: %s", error)
    retry_configs = (
        (int(settings.VGGT_RETRY_WINDOW_SIZE), int(settings.VGGT_RETRY_WINDOW_OVERLAP)),
        (int(settings.VGGT_FINAL_WINDOW_SIZE), int(settings.VGGT_FINAL_WINDOW_OVERLAP)),
    )
    last_error: Exception | None = None
    for size, overlap in retry_configs:
        try:
            return _merge_window_predictions(filelist, _window_ranges(len(filelist), size, overlap))
        except Exception as error:
            last_error = error
            if not _is_oom(error):
                raise
            _release_accelerator_cache()
            logger.warning("VGGT window size=%s ran out of memory", size)
    raise RuntimeError("VGGT inference exhausted all configured window strategies") from last_error


def _project_metrics(source: _PhotoArrays, target: _PhotoArrays) -> dict[str, float]:
    source_points = source.world_points.reshape(-1, 3)
    source_conf = _confidence_unit(source.point_conf).reshape(-1)
    valid = np.isfinite(source_points).all(axis=1) & (source_conf >= np.quantile(source_conf, 0.65))
    indices = np.flatnonzero(valid)
    maximum = int(settings.VGGT_MAX_GEOMETRY_POINTS_PER_PAIR)
    if len(indices) > maximum:
        indices = indices[np.linspace(0, len(indices) - 1, maximum, dtype=int)]
    if len(indices) == 0:
        return {"visible_fraction": 0.0, "depth_consistency": 0.0, "reprojection_error": 1.0}
    points = source_points[indices]
    camera = points @ target.extrinsic[:3, :3].T + target.extrinsic[:3, 3]
    depth = camera[:, 2]
    projected = camera @ target.intrinsic.T
    pixels = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-8)
    height, width = target.depth.shape[:2]
    x = np.rint(pixels[:, 0]).astype(int)
    y = np.rint(pixels[:, 1]).astype(int)
    visible = (depth > 1e-6) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.any(visible):
        return {"visible_fraction": 0.0, "depth_consistency": 0.0, "reprojection_error": 1.0}
    x_visible, y_visible, z_visible = x[visible], y[visible], depth[visible]
    target_depth = np.asarray(target.depth[y_visible, x_visible]).reshape(-1)
    relative_depth_error = np.abs(z_visible - target_depth) / np.maximum(np.abs(target_depth), 1e-6)
    consistent = relative_depth_error <= 0.05
    target_points = target.world_points[y_visible, x_visible]
    target_camera = target_points @ target.extrinsic[:3, :3].T + target.extrinsic[:3, 3]
    target_pixels = target_camera @ target.intrinsic.T
    target_pixels = target_pixels[:, :2] / np.maximum(target_pixels[:, 2:3], 1e-8)
    diagonal = math.hypot(width, height)
    reprojection = np.linalg.norm(pixels[visible] - target_pixels, axis=1) / max(diagonal, 1.0)
    return {
        "visible_fraction": float(np.mean(visible)),
        "depth_consistency": float(np.mean(consistent)),
        "reprojection_error": float(np.median(reprojection)) if len(reprojection) else 1.0,
    }


def _relative_rotation_degrees(left: np.ndarray, right: np.ndarray) -> float:
    rotation = left[:3, :3] @ right[:3, :3].T
    value = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(value))


def _compute_pair_relations(
    photo_arrays: dict[int, _PhotoArrays],
    room_labels: dict[int, str],
) -> list[PhotoRelationResult]:
    photo_ids = sorted(photo_arrays)
    centers = {photo_id: _camera_center(photo_arrays[photo_id].extrinsic) for photo_id in photo_ids}
    all_distances = [float(np.linalg.norm(centers[left] - centers[right])) for index, left in enumerate(photo_ids) for right in photo_ids[index + 1:]]
    scene_scale = max(float(np.median(all_distances)) if all_distances else 1.0, 1e-4)
    base_metrics: dict[tuple[int, int], dict[str, Any]] = {}
    for index, left_id in enumerate(photo_ids):
        for right_id in photo_ids[index + 1:]:
            forward = _project_metrics(photo_arrays[left_id], photo_arrays[right_id])
            backward = _project_metrics(photo_arrays[right_id], photo_arrays[left_id])
            baseline = float(np.linalg.norm(centers[right_id] - centers[left_id]))
            base_metrics[(left_id, right_id)] = {
                "overlap": 0.5 * (forward["visible_fraction"] + backward["visible_fraction"]),
                "depth_consistency": 0.5 * (forward["depth_consistency"] + backward["depth_consistency"]),
                "reprojection_error": 0.5 * (forward["reprojection_error"] + backward["reprojection_error"]),
                "baseline": baseline,
                "normalized_baseline": baseline / scene_scale,
                "rotation_degrees": _relative_rotation_degrees(photo_arrays[left_id].extrinsic, photo_arrays[right_id].extrinsic),
                "forward": forward,
                "backward": backward,
            }
    logger.info(
        "VGGT_RELATION_GEOMETRY_COMPLETE photos=%s all_pairs=%s",
        len(photo_ids),
        len(base_metrics),
    )
    relations: list[PhotoRelationResult] = []
    for pair, metrics in base_metrics.items():
        left_id, right_id = pair
        left_domain = _scene_type_for_label(room_labels.get(left_id))
        right_domain = _scene_type_for_label(room_labels.get(right_id))
        duplicate = metrics["overlap"] >= 0.85 and metrics["normalized_baseline"] <= 0.03
        interpolation_safe = (
            metrics["overlap"] >= 0.35
            and metrics["depth_consistency"] >= 0.45
            and metrics["reprojection_error"] <= 0.015
            and metrics["rotation_degrees"] <= 50.0
        )
        doorway_bridge = (
            metrics["overlap"] >= 0.15
            and metrics["depth_consistency"] >= 0.25
            and metrics["reprojection_error"] <= 0.03
        )
        if duplicate:
            continuity_type = "duplicate"
        elif interpolation_safe:
            continuity_type = "interpolation_safe"
        elif doorway_bridge:
            continuity_type = "doorway_bridge"
        elif metrics["overlap"] >= 0.08 or metrics["depth_consistency"] >= 0.15:
            continuity_type = "cut_only"
        else:
            continuity_type = "unrelated"
        rotation_score = max(0.0, 1.0 - metrics["rotation_degrees"] / 90.0)
        relation_confidence = float(np.clip(
            0.40 * metrics["overlap"]
            + 0.30 * metrics["depth_consistency"]
            + 0.20 * (1.0 - min(metrics["reprojection_error"] / 0.05, 1.0))
            + 0.10 * rotation_score,
            0.0,
            1.0,
        ))
        delta = centers[right_id] - centers[left_id]
        relations.append(PhotoRelationResult(
            photo_a_id=left_id,
            photo_b_id=right_id,
            overlap_score=float(metrics["overlap"]),
            track_support=0.0,
            reprojection_score=float(np.clip(1.0 - metrics["reprojection_error"] / 0.05, 0.0, 1.0)),
            relation_confidence=relation_confidence,
            baseline_distance=float(metrics["baseline"]),
            relative_transform={"translation": [float(value) for value in delta], "normalized_baseline": float(metrics["normalized_baseline"]), "rotation_degrees": float(metrics["rotation_degrees"])},
            direction_dx=0.0,
            direction_dy=0.0,
            continuity_type=continuity_type,
            is_bridge_edge=continuity_type == "doorway_bridge",
            is_connected=continuity_type == "interpolation_safe",
            debug_metrics={
                **metrics,
                "verification_source": "vggt_joint_geometry",
                "left_domain": left_domain,
                "right_domain": right_domain,
            },
        ))
    return relations


def _relation_lookup(relations: list[PhotoRelationResult]) -> dict[tuple[int, int], PhotoRelationResult]:
    return {(min(relation.photo_a_id, relation.photo_b_id), max(relation.photo_a_id, relation.photo_b_id)): relation for relation in relations}


def _connected_components(photo_ids: Iterable[int], relations: list[PhotoRelationResult]) -> list[list[int]]:
    adjacency = {int(photo_id): set() for photo_id in photo_ids}
    for relation in relations:
        if relation.is_connected:
            adjacency[relation.photo_a_id].add(relation.photo_b_id)
            adjacency[relation.photo_b_id].add(relation.photo_a_id)
    result: list[list[int]] = []
    seen: set[int] = set()
    for photo_id in sorted(adjacency):
        if photo_id in seen:
            continue
        stack, component = [photo_id], []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            stack.extend(sorted(adjacency[current] - seen))
        result.append(sorted(component))
    return result


def _query_points_from_confidence(confidence: np.ndarray, count: int) -> torch.Tensor:
    confidence = _confidence_unit(confidence)
    height, width = confidence.shape[:2]
    side = max(1, int(math.ceil(math.sqrt(count))))
    points: list[list[float]] = []
    for y_start in np.linspace(0, height, side + 1, dtype=int)[:-1]:
        y_end = min(height, y_start + max(1, int(math.ceil(height / side))))
        for x_start in np.linspace(0, width, side + 1, dtype=int)[:-1]:
            x_end = min(width, x_start + max(1, int(math.ceil(width / side))))
            cell = confidence[y_start:y_end, x_start:x_end]
            if cell.size == 0:
                continue
            row, column = np.unravel_index(int(np.argmax(cell)), cell.shape)
            points.append([float(x_start + column), float(y_start + row)])
    return torch.tensor(points[:count], dtype=torch.float32)


def _enrich_tracks(components: list[list[int]], relations: list[PhotoRelationResult], file_by_photo: dict[int, str], arrays: dict[int, _PhotoArrays]) -> None:
    relation_by_pair = _relation_lookup(relations)
    for component in components:
        # A pathological global component must never turn into an all-listing track request.
        group_size = max(2, int(settings.VGGT_TRACK_GROUP_MAX))
        for start in range(0, len(component), group_size):
            group = component[start:start + group_size]
            if len(group) < 2:
                continue
            anchor = group[0]
            query_points = _query_points_from_confidence(arrays[anchor].depth_conf, int(settings.VGGT_TRACK_POINTS_PER_IMAGE))
            try:
                tracks = vggt_model.predict_tracks([file_by_photo[photo_id] for photo_id in group], query_points)
            except Exception as error:  # Do not create a fake track score.
                logger.warning("VGGT track verification unavailable for component=%s: %s", group, error)
                for left_index, left_id in enumerate(group):
                    for right_id in group[left_index + 1:]:
                        relation = relation_by_pair.get((min(left_id, right_id), max(left_id, right_id)))
                        if relation is not None:
                            relation.debug_metrics["track_status"] = "unavailable"
                continue
            visibility = _confidence_unit(tracks["visibility"].numpy())
            confidence = _confidence_unit(tracks["confidence"].numpy())
            track_positions = tracks["track"].numpy()
            for left_index, left_id in enumerate(group):
                for right_index in range(left_index + 1, len(group)):
                    right_id = group[right_index]
                    relation = relation_by_pair.get((min(left_id, right_id), max(left_id, right_id)))
                    if relation is None:
                        continue
                    valid = (visibility[left_index] >= 0.5) & (visibility[right_index] >= 0.5) & (confidence[left_index] >= 0.5) & (confidence[right_index] >= 0.5)
                    support = float(np.mean(valid)) if len(valid) else 0.0
                    motion = np.median(track_positions[right_index, valid] - track_positions[left_index, valid], axis=0) if np.any(valid) else np.zeros(2)
                    valid_indices = np.flatnonzero(valid)
                    if len(valid_indices) > 24:
                        valid_indices = valid_indices[np.linspace(0, len(valid_indices) - 1, 24, dtype=int)]
                    track_matches = [
                        {
                            "from": [round(float(value), 2) for value in track_positions[left_index, point_index]],
                            "to": [round(float(value), 2) for value in track_positions[right_index, point_index]],
                        }
                        for point_index in valid_indices
                    ]
                    relation.track_support = support
                    relation.direction_dx = float(motion[0])
                    relation.direction_dy = float(motion[1])
                    relation.relation_confidence = float(np.clip(0.85 * relation.relation_confidence + 0.15 * support, 0.0, 1.0))
                    relation.debug_metrics.update({
                        "track_status": "verified",
                        "track_points": int(len(valid)),
                        "track_visible_fraction": support,
                        "track_group_size": len(group),
                        "track_matches": track_matches,
                    })


def _edge_score(left_id: int, right_id: int, relations: dict[tuple[int, int], PhotoRelationResult], positions: dict[int, int]) -> float:
    relation = relations.get((min(left_id, right_id), max(left_id, right_id)))
    if relation is None or not relation.is_connected:
        return -1e9
    reprojection = relation.reprojection_score
    upload_proximity = math.exp(-abs(positions.get(left_id, 0) - positions.get(right_id, 0)) / 8.0)
    camera_motion = math.hypot(relation.direction_dx, relation.direction_dy)
    smoothness = math.exp(-max(camera_motion - 160.0, 0.0) / 160.0)
    return 0.30 * relation.relation_confidence + 0.20 * relation.overlap_score + 0.15 * relation.track_support + 0.15 * reprojection + 0.10 * smoothness + 0.10 * upload_proximity


def _order_component(component: list[int], relations: list[PhotoRelationResult], positions: dict[int, int]) -> list[int]:
    if len(component) <= 1:
        return component
    relation_by_pair = _relation_lookup(relations)
    width = 32
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
        states = sorted(next_states, key=lambda state: (state[0], [-positions.get(photo_id, 0) for photo_id in state[1]]), reverse=True)[:width]
    complete = [state for state in states if len(state[1]) == len(component)]
    if complete:
        return max(complete, key=lambda state: state[0])[1]
    return sorted(component, key=lambda photo_id: positions.get(photo_id, 0))


def _scene_type(component: list[int], room_labels: dict[int, str]) -> str:
    types = [_scene_type_for_label(room_labels.get(photo_id)) for photo_id in component]
    return max(set(types), key=types.count) if types else "interior"


def _motion_affordance(component: list[int], relations: list[PhotoRelationResult]) -> str:
    internal = [relation for relation in relations if relation.photo_a_id in component and relation.photo_b_id in component]
    if len(component) == 1:
        return "micro_push_in"
    average_confidence = float(np.mean([relation.relation_confidence for relation in internal])) if internal else 0.0
    average_tracks = float(np.mean([relation.track_support for relation in internal])) if internal else 0.0
    if average_confidence >= 0.74 and average_tracks >= 0.62:
        return "multi_view"
    if average_confidence >= 0.62:
        return "parallax"
    return "reveal" if average_confidence >= 0.50 else "micro_push_in"


def _hero_photo(component: list[int], arrays: dict[int, _PhotoArrays], quality_scores: dict[int, float], room_labels: dict[int, str], editorial_roles: dict[int, str]) -> int:
    forced = [photo_id for photo_id in component if editorial_roles.get(photo_id) == "hero"]
    if forced:
        return forced[0]
    def score(photo_id: int) -> float:
        data = arrays[photo_id]
        quality = float(np.clip(quality_scores.get(photo_id, 0.0), 0.0, 1.0))
        importance = 1.0 if any(token in str(room_labels.get(photo_id, "")).lower() for token in SOCIAL_LABEL_TOKENS) else 0.45
        coverage = float(np.mean(_confidence_unit(data.point_conf) >= 0.5))
        geometry = 0.5 * float(np.mean(_confidence_unit(data.depth_conf))) + 0.5 * float(np.mean(_confidence_unit(data.point_conf)))
        novelty = min(1.0, float(np.std(data.depth)) / max(float(np.mean(data.depth)), 1e-6))
        return 0.30 * quality + 0.25 * importance + 0.20 * coverage + 0.15 * geometry + 0.10 * novelty
    return max(component, key=score)


def run_vggt_scene_pipeline(
    images: list[object],
    photo_ids: list[int],
    room_labels: list[str],
    positions: list[int],
    *,
    job_id: int,
    s3_client: S3Client | None = None,
    quality_scores: dict[int, float] | None = None,
    editorial_roles: dict[int, str] | None = None,
) -> tuple[list[PhotoGeometryResult], list[PhotoRelationResult], list[SceneComponentResult]]:
    if not images:
        return [], [], []
    quality_scores = quality_scores or {}
    editorial_roles = editorial_roles or {}
    logger.info("VGGT_SCENE_PIPELINE_START job_id=%s photos=%s", job_id, len(images))
    directory, filelist = _save_input_images(images, photo_ids)
    try:
        predictions, window_strategy = _predict_with_retries(filelist)
        logger.info(
            "VGGT_RECONSTRUCTION_COMPLETE job_id=%s photos=%s strategy=%s runtime_seconds=%s",
            job_id,
            len(images),
            window_strategy.get("kind"),
            predictions.get("runtime", {}).get("runtime_seconds"),
        )
        predictions["runtime"]["window_strategy"] = window_strategy
        arrays: dict[int, _PhotoArrays] = {}
        geometries: list[PhotoGeometryResult] = []
        for index, photo_id in enumerate(photo_ids):
            data = _PhotoArrays(
                extrinsic=np.asarray(predictions["extrinsic"][index], dtype=np.float64),
                intrinsic=np.asarray(predictions["intrinsic"][index], dtype=np.float64),
                depth=np.asarray(predictions["depth_map"][index], dtype=np.float64),
                depth_conf=np.asarray(predictions["depth_conf"][index], dtype=np.float64),
                point_map=np.asarray(predictions["point_map"][index], dtype=np.float64),
                point_conf=np.asarray(predictions["point_conf"][index], dtype=np.float64),
                world_points=np.asarray(predictions["point_map_unprojected"][index], dtype=np.float64),
            )
            arrays[int(photo_id)] = data
            depth_confidence = float(np.mean(_confidence_unit(data.depth_conf)))
            point_confidence = float(np.mean(_confidence_unit(data.point_conf)))
            depth_uri = _artifact_uri(s3_client, f"jobs/{job_id}/geometry/photo_{photo_id}_depth.npz", data.depth)
            point_uri = _artifact_uri(s3_client, f"jobs/{job_id}/geometry/photo_{photo_id}_pointmap.npz", data.world_points)
            geometries.append(PhotoGeometryResult(
                photo_id=int(photo_id),
                pose_confidence=float(0.5 * depth_confidence + 0.5 * point_confidence),
                depth_confidence=depth_confidence,
                point_confidence=point_confidence,
                visibility_score=float(np.mean(_confidence_unit(data.point_conf) >= 0.5)),
                reprojection_error=0.0,
                camera_extrinsic=data.extrinsic.tolist(),
                camera_intrinsic=data.intrinsic.tolist(),
                camera_center=_camera_center(data.extrinsic).tolist(),
                view_direction=_view_direction(data.extrinsic).tolist(),
                depth_artifact_uri=depth_uri,
                point_map_artifact_uri=point_uri,
                local_metrics={"depth_mean": float(np.mean(data.depth)), "depth_std": float(np.std(data.depth)), "runtime": predictions["runtime"]},
            ))
        labels = {int(photo_id): str(room_labels[index] if index < len(room_labels) else "") for index, photo_id in enumerate(photo_ids)}
        position_map = {int(photo_id): int(positions[index] if index < len(positions) else index) for index, photo_id in enumerate(photo_ids)}
        file_by_photo = {int(photo_id): filelist[index] for index, photo_id in enumerate(photo_ids)}
        relations = _compute_pair_relations(arrays, labels)
        components = _connected_components(photo_ids, relations)
        logger.info("SCENE_COMPONENTS_BUILT job_id=%s components=%s", job_id, len(components))
        _enrich_tracks(components, relations, file_by_photo, arrays)
        results: list[SceneComponentResult] = []
        for index, component in enumerate(components):
            ordered = _order_component(component, relations, position_map)
            internal = [relation for relation in relations if relation.photo_a_id in component and relation.photo_b_id in component]
            component_geometries = [geometry for geometry in geometries if geometry.photo_id in component]
            results.append(SceneComponentResult(
                component_key=f"component-{index + 1}",
                photo_ids=component,
                ordered_photo_ids=ordered,
                scene_type=_scene_type(component, labels),
                geometry_confidence=float(np.mean([geometry.pose_confidence for geometry in component_geometries])),
                connectivity_confidence=float(np.mean([relation.relation_confidence for relation in internal])) if internal else 0.0,
                track_coverage=float(np.mean([relation.track_support for relation in internal])) if internal else 0.0,
                avg_reprojection_error=float(np.mean([1.0 - relation.reprojection_score for relation in internal])) if internal else 0.0,
                hero_photo_id=_hero_photo(component, arrays, quality_scores, labels, editorial_roles),
                depth_range=float(np.ptp([float(np.mean(arrays[photo_id].depth)) for photo_id in component])),
                motion_affordance=_motion_affordance(component, relations),
                debug_metrics={"runtime": predictions["runtime"], "photo_ids": component, "ordered_photo_ids": ordered, "verified_edge_count": sum(1 for relation in internal if relation.is_connected)},
            ))
        logger.info(
            "VGGT_SCENE_PIPELINE_COMPLETE job_id=%s geometries=%s relations=%s components=%s",
            job_id,
            len(geometries),
            len(relations),
            len(results),
        )
        return geometries, relations, results
    finally:
        _cleanup_files(directory, filelist)
        _release_accelerator_cache()
