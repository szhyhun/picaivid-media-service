"""Precision-first pair verification and ranking."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.pipeline.phase1_analyze.crossing_safety import analyze_crossing_safety


MIN_RAW_MATCHES = 120
MIN_INLIERS = 70
MIN_INLIER_RATIO = 0.22
MIN_COVERAGE = 0.28


def _points_array(matches: list[dict[str, float]], key_x: str, key_y: str) -> np.ndarray:
    if not matches:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray([[float(m[key_x]), float(m[key_y])] for m in matches], dtype=np.float32)


def _coverage_and_entropy(points_norm: np.ndarray) -> tuple[float, float]:
    if len(points_norm) == 0:
        return 0.0, 0.0
    bins = np.zeros((4, 4), dtype=np.float32)
    xs = np.clip((points_norm[:, 0] * 4).astype(np.int32), 0, 3)
    ys = np.clip((points_norm[:, 1] * 4).astype(np.int32), 0, 3)
    for x_idx, y_idx in zip(xs, ys):
        bins[y_idx, x_idx] += 1.0
    coverage = float(np.count_nonzero(bins) / 16.0)
    flat = bins.reshape(-1)
    total = float(np.sum(flat))
    if total <= 0.0:
        return coverage, 0.0
    probs = flat[flat > 0] / total
    entropy = float(-np.sum(probs * np.log(probs)))
    return coverage, entropy


def _convex_hull_area_ratio(points_norm: np.ndarray) -> float:
    if len(points_norm) < 3:
        return 0.0
    try:
        hull = cv2.convexHull(np.float32(points_norm))
        area = float(cv2.contourArea(hull))
    except cv2.error:
        area = 0.0
    return float(np.clip(area, 0.0, 1.0))


def _compute_homography_ratio(raw_matches: list[dict[str, float]], inliers: int) -> float:
    points0 = _points_array(raw_matches, "x0", "y0")
    points1 = _points_array(raw_matches, "x1", "y1")
    if len(points0) < 4 or len(points1) < 4:
        return 0.0
    try:
        _, mask = cv2.findHomography(points0, points1, cv2.RANSAC, 0.01)
    except cv2.error:
        mask = None
    h_inliers = int(mask.sum()) if mask is not None else 0
    return float(h_inliers / max(int(inliers), 1))


def _median_flow_magnitude(inlier_matches: list[dict[str, float]], image_a: Image.Image, image_b: Image.Image) -> float:
    if not inlier_matches:
        return 0.0
    width_a, height_a = image_a.size
    width_b, height_b = image_b.size
    avg_w = max(1.0, float(width_a + width_b) / 2.0)
    avg_h = max(1.0, float(height_a + height_b) / 2.0)
    magnitudes = []
    for match in inlier_matches:
        dx = float(match["dx"]) * avg_w
        dy = float(match["dy"]) * avg_h
        magnitudes.append((dx * dx + dy * dy) ** 0.5)
    return float(np.median(np.asarray(magnitudes, dtype=np.float32))) if magnitudes else 0.0


def _combined_geometry_score(f_inliers: int, f_inlier_ratio: float, median_epipolar_error: float) -> float:
    inlier_count_term = float(np.clip((float(f_inliers) - 70.0) / (220.0 - 70.0), 0.0, 1.0))
    inlier_ratio_term = float(np.clip((float(f_inlier_ratio) - 0.22) / (0.50 - 0.22), 0.0, 1.0))
    residual_term = float(np.clip(1.0 - (float(median_epipolar_error) / 3.0), 0.0, 1.0))
    return float(0.45 * inlier_ratio_term + 0.35 * inlier_count_term + 0.20 * residual_term)


def _order_proximity(position_a: int, position_b: int) -> float:
    return float(np.clip(1.0 - (abs(int(position_a) - int(position_b)) / 12.0), 0.0, 1.0))


def _certification_status(pair_rank: float) -> str:
    if pair_rank >= 0.68:
        return "strong"
    if pair_rank >= 0.45:
        return "usable"
    return "reject"


def verify_pair_precision_first(
    photo_a_id: int,
    photo_b_id: int,
    image_a: Image.Image,
    image_b: Image.Image,
    dinov2_similarity: float,
    position_a: int,
    position_b: int,
    depth_cache: dict[int, np.ndarray],
    preprocessed_cache: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from app.pipeline.phase1_analyze.learned_matching import (
        DEFAULT_LOFTR_INPUT_SIZE,
        _get_native_preprocessed_entry,
        _match_loftr_kornia_indoor_native,
    )

    target_long_side = max(64, int(max(DEFAULT_LOFTR_INPUT_SIZE)))
    prep0 = None
    prep1 = None
    if preprocessed_cache is not None:
        prep0 = preprocessed_cache.get(photo_a_id)
        if prep0 is None:
            prep0 = _get_native_preprocessed_entry(image_a, target_long_side=target_long_side)
            preprocessed_cache[photo_a_id] = prep0
        prep1 = preprocessed_cache.get(photo_b_id)
        if prep1 is None:
            prep1 = _get_native_preprocessed_entry(image_b, target_long_side=target_long_side)
            preprocessed_cache[photo_b_id] = prep1

    # Use direct forward-only native LoFTR here.
    # The precision-first production path must not inherit legacy reverse retry
    # or legacy strict-gate decisions from match_image_pair().
    num_matches, num_inliers, _, direction, diagnostics = _match_loftr_kornia_indoor_native(
        img1=image_a,
        img2=image_b,
        full_diagnostics=True,
        preprocessed_entry0=prep0,
        preprocessed_entry1=prep1,
    )
    return summarize_pair_precision_first(
        photo_a_id=photo_a_id,
        photo_b_id=photo_b_id,
        image_a=image_a,
        image_b=image_b,
        dinov2_similarity=dinov2_similarity,
        position_a=position_a,
        position_b=position_b,
        depth_cache=depth_cache,
        num_matches=num_matches,
        num_inliers=num_inliers,
        direction=direction,
        diagnostics=diagnostics,
    )


def summarize_pair_precision_first(
    photo_a_id: int,
    photo_b_id: int,
    image_a: Image.Image,
    image_b: Image.Image,
    dinov2_similarity: float,
    position_a: int,
    position_b: int,
    depth_cache: dict[int, np.ndarray],
    num_matches: int,
    num_inliers: int,
    direction: tuple[float, float] | None,
    diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    raw_matches = diagnostics.get("raw_matches") if isinstance(diagnostics.get("raw_matches"), list) else []
    inlier_matches = diagnostics.get("inlier_matches") if isinstance(diagnostics.get("inlier_matches"), list) else []
    raw_matches = [m for m in raw_matches if isinstance(m, dict)]
    inlier_matches = [m for m in inlier_matches if isinstance(m, dict)]

    raw_match_count = int(diagnostics.get("raw_correspondence_count") or num_matches or len(raw_matches))
    f_inliers = int(num_inliers)
    f_inlier_ratio = float(f_inliers / max(raw_match_count, 1))
    inlier_points_a = _points_array(inlier_matches, "x0", "y0")
    inlier_points_b = _points_array(inlier_matches, "x1", "y1")
    coverage_4x4, grid_entropy = _coverage_and_entropy(inlier_points_a)
    overlap_ratio = float(min(_convex_hull_area_ratio(inlier_points_a), _convex_hull_area_ratio(inlier_points_b)))
    effective_overlap = float(max(overlap_ratio, 0.55 * coverage_4x4))
    homography_ratio = _compute_homography_ratio(raw_matches, f_inliers)
    score_components = diagnostics.get("score_components") if isinstance(diagnostics.get("score_components"), dict) else {}
    median_epipolar_error = float(score_components.get("median_epipolar_error", 5.0) or 5.0)
    median_flow_magnitude = _median_flow_magnitude(inlier_matches, image_a, image_b)
    combined_geometry_score = _combined_geometry_score(f_inliers, f_inlier_ratio, median_epipolar_error)

    gate_pass = (
        raw_match_count >= MIN_RAW_MATCHES
        and f_inliers >= MIN_INLIERS
        and f_inlier_ratio >= MIN_INLIER_RATIO
        and coverage_4x4 >= MIN_COVERAGE
        and not (homography_ratio > 1.25 and coverage_4x4 < 0.35)
    )
    if not gate_pass:
        rejection_reason = "weak_geometry"
    else:
        rejection_reason = None

    crossing = analyze_crossing_safety(
        photo_a_id=photo_a_id,
        photo_b_id=photo_b_id,
        image_a=image_a,
        image_b=image_b,
        inlier_matches=inlier_matches,
        depth_cache=depth_cache,
    )
    if crossing["hard_reject"]:
        rejection_reason = "crossing_hard_reject"

    order_proximity = _order_proximity(position_a, position_b)
    homography_penalty = 0.0
    if coverage_4x4 < 0.45:
        homography_penalty = float(np.clip((homography_ratio - 1.2) / 0.3, 0.0, 1.0))
    low_flow_penalty = 0.0
    if median_flow_magnitude < 5.0:
        low_flow_penalty = float(np.clip((5.0 - median_flow_magnitude) / 5.0, 0.0, 1.0))

    pair_rank = float(
        np.clip(
            0.24 * effective_overlap
            + 0.20 * coverage_4x4
            + 0.18 * combined_geometry_score
            + 0.15 * float(dinov2_similarity)
            + 0.10 * order_proximity
            - 0.08 * float(crossing["crossing_penalty"])
            - 0.03 * homography_penalty
            - 0.02 * low_flow_penalty,
            0.0,
            1.0,
        )
    )
    certification_status = _certification_status(pair_rank)
    if rejection_reason is None and certification_status == "reject":
        rejection_reason = "low_pair_rank"

    return {
        "pair_source": None,
        "dinov2_similarity": float(dinov2_similarity),
        "raw_matches": raw_match_count,
        "f_inliers": f_inliers,
        "f_inlier_ratio": float(f_inlier_ratio),
        "coverage_4x4": float(coverage_4x4),
        "grid_entropy": float(grid_entropy),
        "overlap_ratio": float(overlap_ratio),
        "homography_ratio": float(homography_ratio),
        "median_epipolar_error": float(median_epipolar_error),
        "median_flow_magnitude": float(median_flow_magnitude),
        "combined_geometry_score": float(combined_geometry_score),
        "near_positive_ratio": float(crossing["near_positive_ratio"]),
        "near_negative_ratio": float(crossing["near_negative_ratio"]),
        "split_score": float(crossing["split_score"]),
        "depth_monotonicity_score": float(crossing["depth_monotonicity_score"]),
        "dominant_foreground_side_a": int(crossing["dominant_foreground_side_a"]),
        "dominant_foreground_side_b": int(crossing["dominant_foreground_side_b"]),
        "foreground_support_persistence_penalty": float(crossing["foreground_support_persistence_penalty"]),
        "crossing_penalty": float(crossing["crossing_penalty"]),
        "order_proximity": float(order_proximity),
        "pair_rank": float(pair_rank),
        "certification_status": certification_status,
        "rejection_reason": rejection_reason,
        "direction_dx": float(direction[0]) if direction else None,
        "direction_dy": float(direction[1]) if direction else None,
        "is_connected": 1 if certification_status in {"strong", "usable"} and rejection_reason is None else 0,
        "homography_penalty": float(homography_penalty),
        "low_flow_penalty": float(low_flow_penalty),
        "hard_reject": bool(crossing["hard_reject"] or not gate_pass),
        "raw_matches_payload": raw_matches,
        "inlier_matches_payload": inlier_matches,
    }
