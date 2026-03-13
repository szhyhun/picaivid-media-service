"""Crossing-safety analysis for precision-first pair certification."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.models.midas import midas_model


LEFT_BUCKET = 0
CENTER_BUCKET = 1
RIGHT_BUCKET = 2
DEPTH_CACHE_LONG_SIDE = 512


def _bucket_for_x(x_norm: float) -> int:
    if x_norm < 0.33:
        return LEFT_BUCKET
    if x_norm > 0.66:
        return RIGHT_BUCKET
    return CENTER_BUCKET


def _sample_depth_values(depth_map: np.ndarray, points_norm: np.ndarray) -> np.ndarray:
    if depth_map.size == 0 or points_norm.size == 0:
        return np.empty((0,), dtype=np.float32)
    height, width = depth_map.shape[:2]
    xs = np.clip(np.round(points_norm[:, 0] * max(width - 1, 1)).astype(np.int32), 0, max(width - 1, 0))
    ys = np.clip(np.round(points_norm[:, 1] * max(height - 1, 1)).astype(np.int32), 0, max(height - 1, 0))
    return depth_map[ys, xs].astype(np.float32)


def _compress_depth_map_for_cache(depth_map: np.ndarray) -> np.ndarray:
    if depth_map.size == 0:
        return np.empty((0, 0), dtype=np.float16)
    height, width = depth_map.shape[:2]
    long_side = max(height, width)
    if long_side <= DEPTH_CACHE_LONG_SIDE:
        return depth_map.astype(np.float16, copy=False)
    scale = float(DEPTH_CACHE_LONG_SIDE) / float(long_side)
    resized = cv2.resize(
        depth_map,
        dsize=(
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )
    return resized.astype(np.float16, copy=False)


def _cluster_near_points(points_norm: np.ndarray, max_clusters: int = 2) -> list[np.ndarray]:
    if len(points_norm) <= 2:
        return [points_norm] if len(points_norm) else []
    cluster_count = int(min(max_clusters, len(points_norm)))
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        0.01,
    )
    compactness, labels, centers = cv2.kmeans(
        np.float32(points_norm),
        cluster_count,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    del compactness, centers
    labels = labels.reshape(-1)
    clusters: list[np.ndarray] = []
    for cluster_id in range(cluster_count):
        cluster_points = points_norm[labels == cluster_id]
        if len(cluster_points) > 0:
            clusters.append(cluster_points)
    clusters.sort(key=len, reverse=True)
    return clusters[:max_clusters]


def _blob_metrics(points_norm_a: np.ndarray, points_norm_b: np.ndarray, total_count: int) -> dict[str, float | int]:
    support_fraction = float(len(points_norm_a) / max(1, total_count))
    center_a = float(np.mean(points_norm_a[:, 0]))
    center_b = float(np.mean(points_norm_b[:, 0]))
    return {
        "support_fraction": support_fraction,
        "center_a": center_a,
        "center_b": center_b,
        "side_a": _bucket_for_x(center_a),
        "side_b": _bucket_for_x(center_b),
    }


def _flow_magnitude(points_a: np.ndarray, points_b: np.ndarray, width: int, height: int) -> np.ndarray:
    if len(points_a) == 0 or len(points_b) == 0:
        return np.empty((0,), dtype=np.float32)
    dx = (points_b[:, 0] - points_a[:, 0]) * float(max(width, 1))
    dy = (points_b[:, 1] - points_a[:, 1]) * float(max(height, 1))
    return np.sqrt((dx * dx) + (dy * dy)).astype(np.float32)


def estimate_depth_cached(cache: dict[int, np.ndarray], photo_id: int, image: Image.Image) -> np.ndarray:
    depth = cache.get(photo_id)
    if depth is not None:
        return depth
    depth = np.asarray(midas_model.estimate_depth(image), dtype=np.float32)
    depth = _compress_depth_map_for_cache(depth)
    cache[photo_id] = depth
    return depth


def analyze_crossing_safety(
    photo_a_id: int,
    photo_b_id: int,
    image_a: Image.Image,
    image_b: Image.Image,
    inlier_matches: list[dict[str, float]],
    depth_cache: dict[int, np.ndarray],
) -> dict[str, Any]:
    if not inlier_matches:
        return {
            "near_positive_ratio": 0.0,
            "near_negative_ratio": 0.0,
            "split_score": 0.0,
            "depth_monotonicity_score": 0.0,
            "dominant_foreground_side_a": CENTER_BUCKET,
            "dominant_foreground_side_b": CENTER_BUCKET,
            "dominant_side_shift_penalty": 0.0,
            "foreground_support_persistence_penalty": 0.0,
            "crossing_penalty": 0.0,
            "strong_side_flip": False,
            "hard_reject": False,
        }

    points_a = np.asarray([[m["x0"], m["y0"]] for m in inlier_matches], dtype=np.float32)
    points_b = np.asarray([[m["x1"], m["y1"]] for m in inlier_matches], dtype=np.float32)
    depth_a = estimate_depth_cached(depth_cache, photo_a_id, image_a)
    sampled_depth = _sample_depth_values(depth_a, points_a)
    if sampled_depth.size == 0:
        return {
            "near_positive_ratio": 0.0,
            "near_negative_ratio": 0.0,
            "split_score": 0.0,
            "depth_monotonicity_score": 0.0,
            "dominant_foreground_side_a": CENTER_BUCKET,
            "dominant_foreground_side_b": CENTER_BUCKET,
            "dominant_side_shift_penalty": 0.0,
            "foreground_support_persistence_penalty": 0.0,
            "crossing_penalty": 0.0,
            "strong_side_flip": False,
            "hard_reject": False,
        }

    q_near = float(np.quantile(sampled_depth, 0.30))
    q_far = float(np.quantile(sampled_depth, 0.70))
    near_mask = sampled_depth <= q_near
    far_mask = sampled_depth >= q_far
    mid_mask = (~near_mask) & (~far_mask)

    near_a = points_a[near_mask]
    near_b = points_b[near_mask]
    if len(near_a) == 0:
        near_positive_ratio = 0.0
        near_negative_ratio = 0.0
        split_score = 0.0
    else:
        near_dx = near_b[:, 0] - near_a[:, 0]
        near_positive_ratio = float(np.mean(near_dx > 0))
        near_negative_ratio = float(np.mean(near_dx < 0))
        split_score = float(min(near_positive_ratio, near_negative_ratio))

    width_a, height_a = image_a.size
    near_motion = float(np.median(_flow_magnitude(points_a[near_mask], points_b[near_mask], width_a, height_a))) if np.any(near_mask) else 0.0
    mid_motion = float(np.median(_flow_magnitude(points_a[mid_mask], points_b[mid_mask], width_a, height_a))) if np.any(mid_mask) else 0.0
    far_motion = float(np.median(_flow_magnitude(points_a[far_mask], points_b[far_mask], width_a, height_a))) if np.any(far_mask) else 0.0

    monotonic_hits = 0
    monotonic_hits += 1 if near_motion > mid_motion else 0
    monotonic_hits += 1 if mid_motion > far_motion else 0
    depth_monotonicity_score = float(monotonic_hits / 2.0)
    depth_monotonicity_penalty = float(1.0 - depth_monotonicity_score)

    near_clusters_a = _cluster_near_points(near_a, max_clusters=2)
    near_clusters_b = _cluster_near_points(near_b, max_clusters=2)
    dominant_side_shift_penalty = 0.0
    persistence_penalty = 0.0
    dominant_side_a = CENTER_BUCKET
    dominant_side_b = CENTER_BUCKET
    strong_side_flip = False

    for idx, cluster_a in enumerate(near_clusters_a):
        if idx >= len(near_clusters_b):
            break
        cluster_b = near_clusters_b[idx]
        metrics = _blob_metrics(cluster_a, cluster_b, max(1, len(near_a)))
        support_fraction = float(metrics["support_fraction"])
        center_shift = abs(float(metrics["center_a"]) - float(metrics["center_b"]))
        support_ratio = min(len(cluster_a), len(cluster_b)) / max(len(cluster_a), len(cluster_b), 1)
        current_persistence = float(np.clip(center_shift / 0.5, 0.0, 1.0) * (1.0 - support_ratio))
        persistence_penalty = max(persistence_penalty, current_persistence)
        if idx == 0:
            dominant_side_a = int(metrics["side_a"])
            dominant_side_b = int(metrics["side_b"])
        if support_fraction >= 0.20 and {int(metrics["side_a"]), int(metrics["side_b"])} == {LEFT_BUCKET, RIGHT_BUCKET}:
            strong_side_flip = True
            dominant_side_shift_penalty = max(dominant_side_shift_penalty, 1.0)
        else:
            dominant_side_shift_penalty = max(dominant_side_shift_penalty, float(np.clip(center_shift / 0.66, 0.0, 1.0)))

    # Normal orbit / lateral movement around a central anchor should not be treated
    # as suspicious when near-flow is coherent and the dominant support stays centered.
    if (
        split_score <= 0.1
        and dominant_side_a == CENTER_BUCKET
        and dominant_side_b == CENTER_BUCKET
        and not strong_side_flip
    ):
        dominant_side_shift_penalty *= 0.15

    crossing_penalty = float(
        max(
            split_score,
            dominant_side_shift_penalty,
            depth_monotonicity_penalty * 0.2,
            persistence_penalty * 0.5,
        )
    )
    strong_split = split_score > 0.35
    hard_reject = bool(strong_split and strong_side_flip)
    return {
        "near_positive_ratio": float(near_positive_ratio),
        "near_negative_ratio": float(near_negative_ratio),
        "split_score": float(split_score),
        "near_motion": float(near_motion),
        "mid_motion": float(mid_motion),
        "far_motion": float(far_motion),
        "depth_monotonicity_score": float(depth_monotonicity_score),
        "dominant_foreground_side_a": int(dominant_side_a),
        "dominant_foreground_side_b": int(dominant_side_b),
        "dominant_side_shift_penalty": float(dominant_side_shift_penalty),
        "foreground_support_persistence_penalty": float(persistence_penalty),
        "crossing_penalty": float(crossing_penalty),
        "strong_split": bool(strong_split),
        "strong_side_flip": bool(strong_side_flip),
        "hard_reject": bool(hard_reject),
    }
