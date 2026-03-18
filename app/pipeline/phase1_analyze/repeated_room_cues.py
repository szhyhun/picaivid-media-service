"""Heuristic repeated-room instance disambiguation cues."""
from __future__ import annotations

import math
import re
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.models.openclip import openclip_model


OUTDOOR_ROOM_LABELS = {
    "exterior",
    "front yard",
    "backyard",
    "patio",
    "deck",
    "balcony",
    "garage exterior",
    "driveway",
    "aerial view",
    "street view",
}

ROOM_INSTANCE_PENALTY_WEIGHTS = {
    "bathroom": 0.22,
    "laundry room": 0.22,
    "bedroom": 0.16,
    "kitchen": 0.16,
    "living room": 0.16,
    "dining room": 0.16,
    "family room": 0.16,
    "office": 0.14,
    "hallway": 0.14,
    "entryway": 0.14,
}

ROOM_ANCHOR_REGIONS = {
    "bathroom": ("vanity", (0.32, 0.18, 0.98, 0.88)),
    "laundry room": ("appliance_core", (0.18, 0.22, 0.88, 0.92)),
    "bedroom": ("bed", (0.14, 0.28, 0.86, 0.92)),
    "kitchen": ("island_or_counter", (0.14, 0.28, 0.90, 0.92)),
    "living room": ("seating_cluster", (0.14, 0.24, 0.88, 0.92)),
    "dining room": ("table", (0.18, 0.28, 0.86, 0.92)),
    "family room": ("seating_cluster", (0.14, 0.24, 0.88, 0.92)),
    "office": ("desk", (0.18, 0.26, 0.86, 0.90)),
    "hallway": ("passage_core", (0.20, 0.14, 0.80, 0.92)),
    "entryway": ("entry_core", (0.18, 0.18, 0.82, 0.92)),
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def normalize_room_label(room_label: str | None) -> str:
    if not room_label:
        return "unknown"
    label = str(room_label).strip().lower()
    label = re.sub(r"\s+", " ", label)
    label = re.sub(r"\s+\d+$", "", label)
    if not label:
        return "unknown"
    return label


def room_scope_penalty_weight(room_label_a: str | None, room_label_b: str | None) -> tuple[str | None, float]:
    normalized_a = normalize_room_label(room_label_a)
    normalized_b = normalize_room_label(room_label_b)
    if normalized_a != normalized_b:
        return None, 0.0
    if normalized_a in {"unknown", "other"} or normalized_a in OUTDOOR_ROOM_LABELS:
        return None, 0.0
    return normalized_a, float(ROOM_INSTANCE_PENALTY_WEIGHTS.get(normalized_a, 0.12))


def _crop_region(array: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    h, w = array.shape[:2]
    left = int(max(0, min(w - 1, round(x0 * w))))
    top = int(max(0, min(h - 1, round(y0 * h))))
    right = int(max(left + 1, min(w, round(x1 * w))))
    bottom = int(max(top + 1, min(h, round(y1 * h))))
    return array[top:bottom, left:right]


def _descriptor_from_crop(array: np.ndarray) -> list[float]:
    if array.size == 0:
        return []
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    small_gray = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
    hue_hist = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten().astype(np.float32)
    sat_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten().astype(np.float32)
    edges = cv2.Canny(gray, 80, 160)
    edge_small = cv2.resize(edges, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    vector = np.concatenate([small_gray.flatten(), edge_small.flatten(), hue_hist, sat_hist])
    norm = float(np.linalg.norm(vector))
    if norm > 1e-8:
        vector /= norm
    return [float(value) for value in vector]


def _semantic_embedding_from_crop(array: np.ndarray) -> list[float]:
    if array.size == 0:
        return []
    crop_image = Image.fromarray(array)
    embedding = openclip_model.get_embedding(crop_image)
    return [float(value) for value in np.asarray(embedding, dtype=np.float32).flatten().tolist()]


def _dinov2_embedding_from_crop(array: np.ndarray) -> list[float]:
    if array.size == 0:
        return []
    try:
        from app.pipeline.phase1_analyze.learned_matching import compute_dinov2_embeddings

        crop_image = Image.fromarray(array)
        embedding = compute_dinov2_embeddings([crop_image])[0]
        return [float(value) for value in np.asarray(embedding, dtype=np.float32).flatten().tolist()]
    except Exception:
        # Keep cue extraction robust. Ranking can continue without anchor semantics.
        return []


def _cosine_similarity(vec_a: list[float] | None, vec_b: list[float] | None) -> float | None:
    if not vec_a or not vec_b:
        return None
    arr_a = np.asarray(vec_a, dtype=np.float32)
    arr_b = np.asarray(vec_b, dtype=np.float32)
    if arr_a.shape != arr_b.shape or arr_a.size == 0:
        return None
    denom = float(np.linalg.norm(arr_a) * np.linalg.norm(arr_b))
    if denom <= 1e-8:
        return None
    return float(np.clip(np.dot(arr_a, arr_b) / denom, -1.0, 1.0))


def _has_window(array: np.ndarray) -> bool:
    upper = _crop_region(array, 0.05, 0.02, 0.95, 0.72)
    if upper.size == 0:
        return False
    hsv = cv2.cvtColor(upper, cv2.COLOR_RGB2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    mask = ((value >= 210) & (saturation >= 20)).astype(np.uint8) * 255
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    total_area = float(mask.shape[0] * mask.shape[1])
    for label_idx in range(1, num_labels):
        area = float(stats[label_idx, cv2.CC_STAT_AREA]) / max(total_area, 1.0)
        width = float(stats[label_idx, cv2.CC_STAT_WIDTH]) / max(mask.shape[1], 1)
        height = float(stats[label_idx, cv2.CC_STAT_HEIGHT]) / max(mask.shape[0], 1)
        if area >= 0.018 and width >= 0.12 and height >= 0.08:
            return True
    return False


def _has_framed_wall_art(array: np.ndarray) -> bool:
    region = _crop_region(array, 0.02, 0.05, 0.98, 0.78)
    if region.size == 0:
        return False
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = float(region.shape[0] * region.shape[1])
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        area_ratio = float(w * h) / max(img_area, 1.0)
        aspect = float(w) / max(float(h), 1.0)
        if 0.004 <= area_ratio <= 0.12 and 0.55 <= aspect <= 1.8:
            return True
    return False


def _has_shower_curtain(array: np.ndarray) -> bool:
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
    dark = gray < 120
    textured = cv2.Laplacian(gray, cv2.CV_32F).var() > 80.0
    sat = hsv[:, :, 1] > 35
    mask = (dark & sat).astype(np.uint8) * 255
    side_region = np.zeros_like(mask)
    w = mask.shape[1]
    side_region[:, : int(w * 0.32)] = mask[:, : int(w * 0.32)]
    side_region[:, int(w * 0.68) :] = mask[:, int(w * 0.68) :]
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(side_region, 8)
    img_area = float(mask.shape[0] * mask.shape[1])
    for label_idx in range(1, num_labels):
        area_ratio = float(stats[label_idx, cv2.CC_STAT_AREA]) / max(img_area, 1.0)
        comp_w = float(stats[label_idx, cv2.CC_STAT_WIDTH])
        comp_h = float(stats[label_idx, cv2.CC_STAT_HEIGHT])
        if area_ratio >= 0.05 and comp_h / max(comp_w, 1.0) >= 1.6:
            return bool(textured)
    return False


def _has_glass_partition(array: np.ndarray) -> bool:
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 75, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=max(40, gray.shape[0] // 4), maxLineGap=12)
    if lines is None:
        return False
    vertical_lines = 0
    min_x = gray.shape[1] * 0.35
    for line in lines[:, 0]:
        x1, y1, x2, y2 = [int(v) for v in line]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy <= dx * 2:
            continue
        if max(x1, x2) < min_x:
            continue
        vertical_lines += 1
    return vertical_lines >= 2


def _has_tub(array: np.ndarray) -> bool:
    lower_left = _crop_region(array, 0.0, 0.42, 0.58, 0.98)
    if lower_left.size == 0:
        return False
    gray = cv2.cvtColor(lower_left, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 140)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45, minLineLength=max(30, gray.shape[1] // 3), maxLineGap=10)
    if lines is None:
        return False
    horizontal_lines = 0
    for line in lines[:, 0]:
        x1, y1, x2, y2 = [int(v) for v in line]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx <= dy * 3:
            continue
        if min(y1, y2) > gray.shape[0] * 0.7:
            continue
        horizontal_lines += 1
    return horizontal_lines >= 2


def _anchor_region_for_label(room_label: str) -> tuple[str | None, tuple[float, float, float, float] | None]:
    anchor = ROOM_ANCHOR_REGIONS.get(room_label)
    if anchor is None:
        return None, None
    return str(anchor[0]), tuple(float(v) for v in anchor[1])


def extract_repeated_room_cues(image: Image.Image, room_label: str | None) -> dict[str, Any]:
    return extract_repeated_room_cues_with_semantics(image, room_label, semantic_regions=None)


def extract_repeated_room_cues_with_semantics(
    image: Image.Image,
    room_label: str | None,
    semantic_regions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    array = np.asarray(image.convert("RGB"))
    normalized_label = normalize_room_label(room_label)
    fixture_crop = _crop_region(array, 0.08, 0.18, 0.92, 0.96)
    mirror_crop = _crop_region(array, 0.18, 0.08, 0.82, 0.62)
    center_crop = _crop_region(array, 0.2, 0.2, 0.8, 0.8)
    anchor_region_name, anchor_region = _anchor_region_for_label(normalized_label)
    if isinstance(semantic_regions, dict):
        anchor_region_name = semantic_regions.get("anchor_label") or anchor_region_name
        semantic_anchor_bbox = semantic_regions.get("anchor_bbox_norm")
        if (
            isinstance(semantic_anchor_bbox, (list, tuple))
            and len(semantic_anchor_bbox) == 4
        ):
            anchor_region = tuple(float(v) for v in semantic_anchor_bbox)
    anchor_crop = _crop_region(array, *anchor_region) if anchor_region is not None else np.empty((0, 0, 3), dtype=np.uint8)

    cues = {
        "normalized_room_label": normalized_label,
        "anchor_region_name": anchor_region_name,
        "anchor_region": anchor_region,
        "has_tub": False,
        "has_shower_curtain": False,
        "has_glass_partition": False,
        "has_window": _has_window(array),
        "has_framed_wall_art": _has_framed_wall_art(array),
        "mirror_region_present": mirror_crop.size > 0,
        "fixture_layout_embedding": _descriptor_from_crop(fixture_crop),
        "mirror_embedding": _descriptor_from_crop(mirror_crop),
        "center_embedding": _semantic_embedding_from_crop(center_crop),
        "anchor_embedding": (
            list(semantic_regions.get("anchor_embedding") or [])
            if isinstance(semantic_regions, dict) and semantic_regions.get("anchor_embedding")
            else _dinov2_embedding_from_crop(anchor_crop)
        ),
        "object_embedding": (
            list(semantic_regions.get("object_embedding") or [])
            if isinstance(semantic_regions, dict)
            else []
        ),
        "object_labels": (
            list(semantic_regions.get("object_labels") or [])
            if isinstance(semantic_regions, dict)
            else []
        ),
        "semantic_backend": (
            str(semantic_regions.get("semantic_backend"))
            if isinstance(semantic_regions, dict) and semantic_regions.get("semantic_backend")
            else "heuristic_fallback"
        ),
    }

    if normalized_label in {"bathroom", "laundry room"}:
        cues["has_tub"] = _has_tub(array)
        cues["has_shower_curtain"] = _has_shower_curtain(array)
        cues["has_glass_partition"] = _has_glass_partition(array)

    return cues


def compute_repeated_room_instance_penalty(
    cues_a: dict[str, Any],
    cues_b: dict[str, Any],
    room_label_a: str | None,
    room_label_b: str | None,
    global_similarity: float | None = None,
    semantic_match_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scope, base_weight = room_scope_penalty_weight(room_label_a, room_label_b)
    result = {
        "repeated_room_scope": scope,
        "repeated_room_penalty_weight": float(base_weight),
        "anchor_region_name": cues_a.get("anchor_region_name") if cues_a.get("anchor_region_name") == cues_b.get("anchor_region_name") else None,
        "fixture_disagreement_count": 0,
        "fixture_instance_penalty": 0.0,
        "mirror_similarity": None,
        "mirror_penalty": 0.0,
        "layout_similarity": None,
        "layout_penalty": 0.0,
        "center_similarity": None,
        "center_support_bonus": 0.0,
        "center_mismatch_penalty": 0.0,
        "anchor_similarity": None,
        "anchor_support_bonus": 0.0,
        "anchor_mismatch_penalty": 0.0,
        "object_similarity": None,
        "object_mismatch_penalty": 0.0,
        "repeated_room_instance_penalty": 0.0,
        "hard_reject": False,
    }
    if scope is None or base_weight <= 0.0:
        return result

    cue_keys = ["has_window", "has_framed_wall_art"]
    if scope in {"bathroom", "laundry room"}:
        cue_keys.extend(["has_tub", "has_shower_curtain", "has_glass_partition"])

    disagreement_count = sum(bool(cues_a.get(key)) != bool(cues_b.get(key)) for key in cue_keys)
    fixture_instance_penalty = min(float(disagreement_count) / 3.0, 1.0)

    layout_similarity = _cosine_similarity(
        cues_a.get("fixture_layout_embedding"),
        cues_b.get("fixture_layout_embedding"),
    )
    if layout_similarity is None:
        layout_penalty = 0.0
    else:
        layout_penalty = _clamp((0.82 - layout_similarity) / 0.28)

    center_similarity = _cosine_similarity(
        cues_a.get("center_embedding"),
        cues_b.get("center_embedding"),
    )
    center_support_bonus = 0.0
    if center_similarity is not None:
        center_support_bonus = _clamp((center_similarity - 0.88) / 0.10)
    center_mismatch_penalty = 0.0
    if global_similarity is not None and center_similarity is not None:
        center_mismatch_penalty = _clamp((float(global_similarity) - center_similarity - 0.05) / 0.30)

    anchor_similarity = _cosine_similarity(
        cues_a.get("anchor_embedding"),
        cues_b.get("anchor_embedding"),
    )
    anchor_support_bonus = 0.0
    if anchor_similarity is not None:
        anchor_support_bonus = _clamp((anchor_similarity - 0.84) / 0.12)
    anchor_mismatch_penalty = 0.0
    if global_similarity is not None and anchor_similarity is not None:
        anchor_mismatch_penalty = _clamp((float(global_similarity) - anchor_similarity - 0.04) / 0.28)

    object_similarity = _cosine_similarity(
        cues_a.get("object_embedding"),
        cues_b.get("object_embedding"),
    )
    object_mismatch_penalty = 0.0
    if global_similarity is not None and object_similarity is not None:
        object_mismatch_penalty = _clamp((float(global_similarity) - object_similarity - 0.04) / 0.26)

    mirror_similarity = None
    mirror_penalty = 0.0
    if scope in {"bathroom", "laundry room"} and cues_a.get("mirror_region_present") and cues_b.get("mirror_region_present"):
        mirror_similarity = _cosine_similarity(
            cues_a.get("mirror_embedding"),
            cues_b.get("mirror_embedding"),
        )
        if mirror_similarity is not None:
            mirror_penalty = _clamp((0.88 - mirror_similarity) / 0.23)

    repeated_room_instance_penalty = max(
        fixture_instance_penalty,
        mirror_penalty * 0.8,
        layout_penalty * 0.8,
        center_mismatch_penalty * 0.8,
        anchor_mismatch_penalty * 0.95,
        object_mismatch_penalty * 0.9,
    )
    repeated_room_instance_penalty = _clamp(
        repeated_room_instance_penalty
        - 0.08 * center_support_bonus
        - 0.10 * anchor_support_bonus
    )

    # Living/dining/family rooms often share windows and wall structure across
    # strong same-instance orbit pairs. If semantic/layout evidence is already
    # very strong, keep the repeated-room penalty from suppressing valid edges.
    if (
        scope in {"living room", "dining room", "family room"}
        and global_similarity is not None
        and layout_similarity is not None
        and center_similarity is not None
        and anchor_similarity is not None
        and float(global_similarity) >= 0.92
        and float(layout_similarity) >= 0.94
        and float(center_similarity) >= 0.90
        and float(anchor_similarity) >= 0.86
        and disagreement_count <= 1
    ):
        repeated_room_instance_penalty *= 0.25

    if isinstance(semantic_match_metrics, dict):
        same_anchor_label_ratio = float(semantic_match_metrics.get("same_anchor_label_ratio") or 0.0)
        anchor_inlier_ratio = float(semantic_match_metrics.get("anchor_inlier_ratio") or 0.0)
        window_match_ratio = float(semantic_match_metrics.get("window_match_ratio") or 0.0)
        background_match_ratio = float(semantic_match_metrics.get("background_match_ratio") or 0.0)
        if same_anchor_label_ratio >= 0.40 and anchor_inlier_ratio >= 0.18:
            repeated_room_instance_penalty *= 0.35
        elif (
            global_similarity is not None
            and float(global_similarity) >= 0.90
            and same_anchor_label_ratio <= 0.12
            and (window_match_ratio >= 0.30 or background_match_ratio >= 0.35)
        ):
            repeated_room_instance_penalty = _clamp(repeated_room_instance_penalty + 0.12)

    hard_reject = (
        scope in {"bathroom", "laundry room"}
        and repeated_room_instance_penalty >= 0.8
        and disagreement_count >= 2
    )

    result.update(
        {
            "fixture_disagreement_count": int(disagreement_count),
            "fixture_instance_penalty": float(fixture_instance_penalty),
            "mirror_similarity": mirror_similarity,
            "mirror_penalty": float(mirror_penalty),
            "layout_similarity": layout_similarity,
            "layout_penalty": float(layout_penalty),
            "center_similarity": center_similarity,
            "center_support_bonus": float(center_support_bonus),
            "center_mismatch_penalty": float(center_mismatch_penalty),
            "anchor_similarity": anchor_similarity,
            "anchor_support_bonus": float(anchor_support_bonus),
            "anchor_mismatch_penalty": float(anchor_mismatch_penalty),
            "object_similarity": object_similarity,
            "object_mismatch_penalty": float(object_mismatch_penalty),
            "repeated_room_instance_penalty": float(repeated_room_instance_penalty),
            "hard_reject": bool(hard_reject),
        }
    )
    return result
