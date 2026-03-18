"""Precision-first pair verification and ranking."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.pipeline.phase1_analyze.crossing_safety import analyze_crossing_safety
from app.pipeline.phase1_analyze.repeated_room_cues import (
    compute_repeated_room_instance_penalty,
    extract_repeated_room_cues_with_semantics,
)
from app.pipeline.phase1_analyze.semantic_regions import (
    get_semantic_regions_cached,
    point_semantic_label,
)


MIN_RAW_MATCHES = 120
MIN_INLIERS = 70
MIN_INLIER_RATIO = 0.22
MIN_COVERAGE = 0.28
SOFT_RAW_MATCH_BAND = 24.0
SOFT_INLIER_BAND = 10.0
SOFT_INLIER_RATIO_BAND = 0.06
SOFT_COVERAGE_BAND = 0.10
WINDOW_PRONE_REGIONS = (
    (0.00, 0.08, 0.42, 0.68),
    (0.58, 0.08, 1.00, 0.68),
)
SEMANTIC_MATCH_COUNT_KEYS = (
    "bed_to_bed",
    "table_to_table",
    "seating_cluster_to_seating_cluster",
    "window_to_window",
    "background_to_background",
)


def _points_array(matches: list[dict[str, float]], key_x: str, key_y: str) -> np.ndarray:
    if not matches:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray([[float(m[key_x]), float(m[key_y])] for m in matches], dtype=np.float32)


def _match_density_in_region(
    matches: list[dict[str, float]],
    key_x: str,
    key_y: str,
    region: tuple[float, float, float, float] | None,
) -> float:
    if not matches or region is None:
        return 0.0
    x0, y0, x1, y1 = [float(v) for v in region]
    if x1 <= x0 or y1 <= y0:
        return 0.0
    hits = 0
    for match in matches:
        px = float(match.get(key_x, -1.0))
        py = float(match.get(key_y, -1.0))
        if x0 <= px <= x1 and y0 <= py <= y1:
            hits += 1
    return float(hits / max(len(matches), 1))


def _match_density_in_regions(
    matches: list[dict[str, float]],
    key_x: str,
    key_y: str,
    regions: tuple[tuple[float, float, float, float], ...],
) -> float:
    if not matches:
        return 0.0
    hits = 0
    for match in matches:
        px = float(match.get(key_x, -1.0))
        py = float(match.get(key_y, -1.0))
        for x0, y0, x1, y1 in regions:
            if x0 <= px <= x1 and y0 <= py <= y1:
                hits += 1
                break
    return float(hits / max(len(matches), 1))


def _annotate_match_semantics(
    matches: list[dict[str, float]],
    semantic_a: dict[str, Any] | None,
    semantic_b: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for match in matches:
        label_a, region_type_a = point_semantic_label(semantic_a, float(match.get("x0", 0.0)), float(match.get("y0", 0.0)))
        label_b, region_type_b = point_semantic_label(semantic_b, float(match.get("x1", 0.0)), float(match.get("y1", 0.0)))
        is_anchor_match = bool(region_type_a == "anchor" and region_type_b == "anchor")
        is_window_match = bool(region_type_a == "window" or region_type_b == "window")
        is_background_match = bool(region_type_a == "background" or region_type_b == "background")
        is_object_match = bool(
            region_type_a in {"anchor", "object"} and region_type_b in {"anchor", "object"}
        )
        semantic_accept = bool(is_anchor_match or (is_object_match and not is_window_match and not is_background_match))
        annotated.append(
            {
                **match,
                "label_a": label_a,
                "label_b": label_b,
                "region_type_a": region_type_a,
                "region_type_b": region_type_b,
                "is_anchor_match": is_anchor_match,
                "is_window_match": is_window_match,
                "is_background_match": is_background_match,
                "is_object_match": is_object_match,
                "semantic_accept": semantic_accept,
            }
        )
    return annotated


def _semantic_match_metrics(matches: list[dict[str, Any]]) -> dict[str, Any]:
    if not matches:
        return {
            "anchor_inlier_ratio": 0.0,
            "object_match_ratio": 0.0,
            "background_match_ratio": 0.0,
            "window_match_ratio": 0.0,
            "same_anchor_label_ratio": 0.0,
            "cross_label_object_ratio": 0.0,
            "semantic_match_counts": {key: 0 for key in SEMANTIC_MATCH_COUNT_KEYS},
        }
    total = max(len(matches), 1)
    anchor_hits = 0
    object_hits = 0
    background_hits = 0
    window_hits = 0
    same_anchor_hits = 0
    cross_label_object_hits = 0
    semantic_match_counts = {key: 0 for key in SEMANTIC_MATCH_COUNT_KEYS}
    for match in matches:
        is_anchor_match = bool(match.get("is_anchor_match"))
        is_object_match = bool(match.get("is_object_match"))
        is_background_match = bool(match.get("is_background_match"))
        is_window_match = bool(match.get("is_window_match"))
        label_a = str(match.get("label_a") or "")
        label_b = str(match.get("label_b") or "")
        if is_anchor_match:
            anchor_hits += 1
            if label_a and label_a == label_b:
                same_anchor_hits += 1
        if is_object_match:
            object_hits += 1
            if label_a and label_b and label_a != label_b:
                cross_label_object_hits += 1
        if is_background_match:
            background_hits += 1
        if is_window_match:
            window_hits += 1
        key = f"{label_a}_to_{label_b}"
        if key in semantic_match_counts:
            semantic_match_counts[key] += 1
    return {
        "anchor_inlier_ratio": float(anchor_hits / total),
        "object_match_ratio": float(object_hits / total),
        "background_match_ratio": float(background_hits / total),
        "window_match_ratio": float(window_hits / total),
        "same_anchor_label_ratio": float(same_anchor_hits / max(anchor_hits, 1)),
        "cross_label_object_ratio": float(cross_label_object_hits / max(object_hits, 1)),
        "semantic_match_counts": semantic_match_counts,
    }


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


def _soft_shortfall(value: float, minimum: float, band: float) -> float:
    if band <= 0.0:
        return 0.0
    return float(np.clip((minimum - value) / band, 0.0, 1.0))


def _geometry_soft_penalty(
    raw_match_count: int,
    f_inliers: int,
    f_inlier_ratio: float,
    coverage_4x4: float,
) -> float:
    raw_shortfall = _soft_shortfall(float(raw_match_count), float(MIN_RAW_MATCHES), SOFT_RAW_MATCH_BAND)
    inlier_shortfall = _soft_shortfall(float(f_inliers), float(MIN_INLIERS), SOFT_INLIER_BAND)
    ratio_shortfall = _soft_shortfall(float(f_inlier_ratio), float(MIN_INLIER_RATIO), SOFT_INLIER_RATIO_BAND)
    coverage_shortfall = _soft_shortfall(float(coverage_4x4), float(MIN_COVERAGE), SOFT_COVERAGE_BAND)
    return float(
        np.clip(
            0.15 * raw_shortfall
            + 0.45 * inlier_shortfall
            + 0.20 * ratio_shortfall
            + 0.20 * coverage_shortfall,
            0.0,
            1.0,
        )
    )


def _hard_geometry_fail(
    raw_match_count: int,
    f_inliers: int,
    f_inlier_ratio: float,
    coverage_4x4: float,
    homography_ratio: float,
) -> bool:
    if homography_ratio > 1.25 and coverage_4x4 < 0.35:
        return True
    return bool(
        raw_match_count < (MIN_RAW_MATCHES - 20)
        or f_inliers < (MIN_INLIERS - 8)
        or f_inlier_ratio < (MIN_INLIER_RATIO - 0.05)
        or coverage_4x4 < (MIN_COVERAGE - 0.08)
    )


def verify_pair_precision_first(
    photo_a_id: int,
    photo_b_id: int,
    image_a: Image.Image,
    image_b: Image.Image,
    dinov2_similarity: float,
    position_a: int,
    position_b: int,
    room_label_a: str | None,
    room_label_b: str | None,
    depth_cache: dict[int, np.ndarray],
    preprocessed_cache: dict[int, dict[str, Any]] | None = None,
    repeated_room_cue_cache: dict[int, dict[str, Any]] | None = None,
    semantic_region_cache: dict[int, dict[str, Any]] | None = None,
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
    semantic_a = get_semantic_regions_cached(photo_a_id, image_a, room_label_a, semantic_region_cache)
    semantic_b = get_semantic_regions_cached(photo_b_id, image_b, room_label_b, semantic_region_cache)
    cues_a = None
    cues_b = None
    if repeated_room_cue_cache is not None:
        cues_a = repeated_room_cue_cache.get(photo_a_id)
        if cues_a is None:
            cues_a = extract_repeated_room_cues_with_semantics(image_a, room_label_a, semantic_regions=semantic_a)
            repeated_room_cue_cache[photo_a_id] = cues_a
        cues_b = repeated_room_cue_cache.get(photo_b_id)
        if cues_b is None:
            cues_b = extract_repeated_room_cues_with_semantics(image_b, room_label_b, semantic_regions=semantic_b)
            repeated_room_cue_cache[photo_b_id] = cues_b

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
        room_label_a=room_label_a,
        room_label_b=room_label_b,
        depth_cache=depth_cache,
        num_matches=num_matches,
        num_inliers=num_inliers,
        direction=direction,
        diagnostics=diagnostics,
        cues_a=cues_a,
        cues_b=cues_b,
        semantic_a=semantic_a,
        semantic_b=semantic_b,
    )


def summarize_pair_precision_first(
    photo_a_id: int,
    photo_b_id: int,
    image_a: Image.Image,
    image_b: Image.Image,
    dinov2_similarity: float,
    position_a: int,
    position_b: int,
    room_label_a: str | None,
    room_label_b: str | None,
    depth_cache: dict[int, np.ndarray],
    num_matches: int,
    num_inliers: int,
    direction: tuple[float, float] | None,
    diagnostics: dict[str, Any] | None,
    cues_a: dict[str, Any] | None = None,
    cues_b: dict[str, Any] | None = None,
    semantic_a: dict[str, Any] | None = None,
    semantic_b: dict[str, Any] | None = None,
) -> dict[str, Any]:
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    if semantic_a is None:
        semantic_a = get_semantic_regions_cached(photo_a_id, image_a, room_label_a, None)
    if semantic_b is None:
        semantic_b = get_semantic_regions_cached(photo_b_id, image_b, room_label_b, None)
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

    geometry_soft_penalty = _geometry_soft_penalty(
        raw_match_count=raw_match_count,
        f_inliers=f_inliers,
        f_inlier_ratio=f_inlier_ratio,
        coverage_4x4=coverage_4x4,
    )
    gate_pass = not _hard_geometry_fail(
        raw_match_count=raw_match_count,
        f_inliers=f_inliers,
        f_inlier_ratio=f_inlier_ratio,
        coverage_4x4=coverage_4x4,
        homography_ratio=homography_ratio,
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

    if cues_a is None:
        cues_a = extract_repeated_room_cues_with_semantics(image_a, room_label_a, semantic_regions=semantic_a)
    if cues_b is None:
        cues_b = extract_repeated_room_cues_with_semantics(image_b, room_label_b, semantic_regions=semantic_b)
    raw_matches_semantic = _annotate_match_semantics(raw_matches, semantic_a, semantic_b)
    inlier_matches_semantic = _annotate_match_semantics(inlier_matches, semantic_a, semantic_b)
    semantic_inlier_metrics = _semantic_match_metrics(inlier_matches_semantic)
    repeated_room = compute_repeated_room_instance_penalty(
        cues_a=cues_a,
        cues_b=cues_b,
        room_label_a=room_label_a,
        room_label_b=room_label_b,
        global_similarity=float(dinov2_similarity),
        semantic_match_metrics=semantic_inlier_metrics,
    )
    if repeated_room["hard_reject"] and combined_geometry_score < 0.75:
        rejection_reason = "different_room_instance"

    order_proximity = _order_proximity(position_a, position_b)
    anchor_region_a = cues_a.get("anchor_region") if isinstance(cues_a, dict) else None
    anchor_region_b = cues_b.get("anchor_region") if isinstance(cues_b, dict) else None
    anchor_match_density_a = _match_density_in_region(inlier_matches, "x0", "y0", anchor_region_a)
    anchor_match_density_b = _match_density_in_region(inlier_matches, "x1", "y1", anchor_region_b)
    anchor_support_score = float(min(anchor_match_density_a, anchor_match_density_b))
    window_match_density_a = _match_density_in_regions(inlier_matches, "x0", "y0", WINDOW_PRONE_REGIONS)
    window_match_density_b = _match_density_in_regions(inlier_matches, "x1", "y1", WINDOW_PRONE_REGIONS)
    window_support_score = float(0.5 * (window_match_density_a + window_match_density_b))
    anchor_similarity = repeated_room.get("anchor_similarity")
    anchor_similarity_bonus = 0.0
    if anchor_similarity is not None:
        anchor_similarity_bonus = float(np.clip((float(anchor_similarity) - 0.84) / 0.12, 0.0, 1.0))
    anchor_mismatch_penalty = float(repeated_room.get("anchor_mismatch_penalty") or 0.0)
    anchor_support_bonus = float(
        np.clip(
            0.70 * anchor_support_score + 0.30 * anchor_similarity_bonus,
            0.0,
            1.0,
        )
    )
    anchor_inlier_ratio = float(semantic_inlier_metrics["anchor_inlier_ratio"])
    object_match_ratio = float(semantic_inlier_metrics["object_match_ratio"])
    background_match_ratio = float(semantic_inlier_metrics["background_match_ratio"])
    window_match_ratio = float(semantic_inlier_metrics["window_match_ratio"])
    same_anchor_label_ratio = float(semantic_inlier_metrics["same_anchor_label_ratio"])
    cross_label_object_ratio = float(semantic_inlier_metrics["cross_label_object_ratio"])
    window_dominance_penalty = float(
        np.clip((window_support_score - anchor_support_score - 0.12) / 0.40, 0.0, 1.0)
    )
    semantic_window_penalty = float(np.clip((window_match_ratio - 0.18) / 0.30, 0.0, 1.0))
    semantic_background_penalty = float(np.clip((background_match_ratio - 0.24) / 0.32, 0.0, 1.0))
    semantic_cross_label_penalty = float(np.clip((cross_label_object_ratio - 0.18) / 0.35, 0.0, 1.0))
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
            + 0.05 * anchor_support_bonus
            + 0.05 * anchor_inlier_ratio
            + 0.04 * object_match_ratio
            + 0.04 * same_anchor_label_ratio
            - 0.08 * float(crossing["crossing_penalty"])
            - 0.07 * anchor_mismatch_penalty
            - 0.12 * geometry_soft_penalty
            - 0.04 * window_dominance_penalty
            - 0.05 * semantic_window_penalty
            - 0.04 * semantic_background_penalty
            - 0.05 * semantic_cross_label_penalty
            - 0.03 * homography_penalty
            - 0.02 * low_flow_penalty,
            0.0,
            1.0,
        )
    )
    pair_rank = float(
        np.clip(
            pair_rank
            - float(repeated_room["repeated_room_penalty_weight"] or 0.0)
            * float(repeated_room["repeated_room_instance_penalty"] or 0.0),
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
        "effective_overlap": float(effective_overlap),
        "homography_ratio": float(homography_ratio),
        "median_epipolar_error": float(median_epipolar_error),
        "median_flow_magnitude": float(median_flow_magnitude),
        "combined_geometry_score": float(combined_geometry_score),
        "geometry_soft_penalty": float(geometry_soft_penalty),
        "near_positive_ratio": float(crossing["near_positive_ratio"]),
        "near_negative_ratio": float(crossing["near_negative_ratio"]),
        "split_score": float(crossing["split_score"]),
        "depth_monotonicity_score": float(crossing["depth_monotonicity_score"]),
        "dominant_foreground_side_a": int(crossing["dominant_foreground_side_a"]),
        "dominant_foreground_side_b": int(crossing["dominant_foreground_side_b"]),
        "foreground_support_persistence_penalty": float(crossing["foreground_support_persistence_penalty"]),
        "crossing_penalty": float(crossing["crossing_penalty"]),
        "repeated_room_scope": repeated_room["repeated_room_scope"],
        "repeated_room_penalty_weight": float(repeated_room["repeated_room_penalty_weight"] or 0.0),
        "fixture_disagreement_count": int(repeated_room["fixture_disagreement_count"]),
        "fixture_instance_penalty": float(repeated_room["fixture_instance_penalty"]),
        "mirror_similarity": (
            float(repeated_room["mirror_similarity"])
            if repeated_room["mirror_similarity"] is not None
            else None
        ),
        "mirror_penalty": float(repeated_room["mirror_penalty"]),
        "layout_similarity": (
            float(repeated_room["layout_similarity"])
            if repeated_room["layout_similarity"] is not None
            else None
        ),
        "layout_penalty": float(repeated_room["layout_penalty"]),
        "center_similarity": (
            float(repeated_room["center_similarity"])
            if repeated_room["center_similarity"] is not None
            else None
        ),
        "center_support_bonus": float(repeated_room["center_support_bonus"]),
        "center_mismatch_penalty": float(repeated_room["center_mismatch_penalty"]),
        "anchor_region_name": repeated_room.get("anchor_region_name"),
        "semantic_backend": (
            semantic_a.get("semantic_backend")
            if isinstance(semantic_a, dict) and semantic_a.get("semantic_backend") == semantic_b.get("semantic_backend")
            else (semantic_a.get("semantic_backend") if isinstance(semantic_a, dict) else None)
        ),
        "semantic_regions_available": bool(
            isinstance(semantic_a, dict)
            and isinstance(semantic_b, dict)
            and semantic_a.get("semantic_regions_available")
            and semantic_b.get("semantic_regions_available")
        ),
        "anchor_label_a": semantic_a.get("anchor_label") if isinstance(semantic_a, dict) else None,
        "anchor_label_b": semantic_b.get("anchor_label") if isinstance(semantic_b, dict) else None,
        "anchor_similarity": (
            float(repeated_room["anchor_similarity"])
            if repeated_room["anchor_similarity"] is not None
            else None
        ),
        "object_similarity": (
            float(repeated_room["object_similarity"])
            if repeated_room.get("object_similarity") is not None
            else None
        ),
        "anchor_match_density_a": float(anchor_match_density_a),
        "anchor_match_density_b": float(anchor_match_density_b),
        "anchor_support_score": float(anchor_support_score),
        "anchor_support_bonus": float(anchor_support_bonus),
        "anchor_inlier_ratio": float(anchor_inlier_ratio),
        "object_match_ratio": float(object_match_ratio),
        "background_match_ratio": float(background_match_ratio),
        "window_match_ratio": float(window_match_ratio),
        "same_anchor_label_ratio": float(same_anchor_label_ratio),
        "cross_label_object_ratio": float(cross_label_object_ratio),
        "anchor_mismatch_penalty": float(anchor_mismatch_penalty),
        "window_match_density_a": float(window_match_density_a),
        "window_match_density_b": float(window_match_density_b),
        "window_support_score": float(window_support_score),
        "window_dominance_penalty": float(window_dominance_penalty),
        "semantic_match_counts": semantic_inlier_metrics["semantic_match_counts"],
        "semantic_regions_left": list(semantic_a.get("regions") or []) if isinstance(semantic_a, dict) else [],
        "semantic_regions_right": list(semantic_b.get("regions") or []) if isinstance(semantic_b, dict) else [],
        "repeated_room_instance_penalty": float(repeated_room["repeated_room_instance_penalty"]),
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
        "raw_matches_payload": raw_matches_semantic,
        "inlier_matches_payload": inlier_matches_semantic,
    }
