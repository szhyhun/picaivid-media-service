"""Optimized 3-stage clustering pipeline for real estate photos.

Stage 1 (cheap): DINOv2 embedding clustering
- Group photos by visual similarity
- Keep clusters small (5-20 images)
- Fast: one forward pass per image

Stage 2 (medium): SuperPoint + LightGlue within clusters
- Only match within each DINOv2 cluster
- Build overlap graph
- Medium cost: N*(N-1)/2 matches per cluster

Stage 3 (selective SfM): Optional COLMAP
- Only for clusters with 3+ images
- Only if strong geometric consistency
- Expensive: full SfM reconstruction

This keeps 90% of compute small while getting high-quality results.
"""
import logging
import io
import base64
import copy
import os
import sys
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING, Any
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from app.core.config import settings

if TYPE_CHECKING:
    from app.db.models import JobPhoto

logger = logging.getLogger(__name__)

# ============================================================================
# STAGE 1: DINOv2 Embeddings
# ============================================================================

# DINOv2 model singleton
_dinov2_model = None
_dinov2_transform = None


def _load_dinov2():
    """Load DINOv2 model (lazy initialization)."""
    global _dinov2_model, _dinov2_transform

    if _dinov2_model is not None:
        return _dinov2_model, _dinov2_transform

    try:
        # Use transformers for DINOv2
        from transformers import AutoImageProcessor, AutoModel

        model_name = "facebook/dinov2-base"
        _dinov2_transform = AutoImageProcessor.from_pretrained(model_name)
        _dinov2_model = AutoModel.from_pretrained(model_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _dinov2_model = _dinov2_model.to(device)
        _dinov2_model.eval()

        logger.info(f"Loaded DINOv2 model on {device}")
        return _dinov2_model, _dinov2_transform

    except Exception as e:
        logger.error(f"Failed to load DINOv2: {e}")
        return None, None


def compute_dinov2_embeddings(images: List[Image.Image]) -> np.ndarray:
    """Compute DINOv2 embeddings for a list of images.

    Args:
        images: List of PIL Images

    Returns:
        NxD array of embeddings (D=768 for dinov2-base)
    """
    model, transform = _load_dinov2()

    if model is None:
        # Random embeddings make clustering non-deterministic and unreliable.
        raise RuntimeError(
            "DINOv2 model is unavailable. Refusing random embedding fallback "
            "because it causes inconsistent clustering."
        )

    device = next(model.parameters()).device
    embeddings = []

    with torch.no_grad():
        for img in images:
            # Ensure RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Process image
            inputs = transform(img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get CLS token embedding
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(embedding[0])

    return np.array(embeddings, dtype=np.float32)


def cluster_by_dinov2(
    images: List[Image.Image],
    photo_ids: List[int],
    max_cluster_size: int = 15,
    min_cluster_size: int = 2,
) -> List[List[int]]:
    """Cluster images using DINOv2 embeddings.

    Uses HDBSCAN for robust clustering that automatically determines
    the number of clusters.

    Args:
        images: List of PIL Images
        photo_ids: List of photo IDs
        max_cluster_size: Maximum images per cluster
        min_cluster_size: Minimum images per cluster

    Returns:
        List of photo ID lists (clusters)
    """
    from sklearn.cluster import HDBSCAN
    from sklearn.preprocessing import normalize

    n = len(images)
    if n <= 2:
        return [photo_ids]

    # Compute embeddings
    logger.info(f"Computing DINOv2 embeddings for {n} images...")
    embeddings = compute_dinov2_embeddings(images)

    # Normalize for cosine distance
    embeddings = normalize(embeddings)

    # HDBSCAN clustering
    # min_cluster_size=2 to catch even small overlapping groups
    # min_samples=1 for single-linkage-like behavior
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean",  # On normalized vectors = cosine
        cluster_selection_method="eom",
    )

    labels = clusterer.fit_predict(embeddings)

    # Group by label
    clusters_dict = defaultdict(list)
    noise_photos = []

    for i, label in enumerate(labels):
        if label == -1:
            # Noise point - will be its own cluster
            noise_photos.append(photo_ids[i])
        else:
            clusters_dict[label].append(photo_ids[i])

    # Convert to list
    clusters = list(clusters_dict.values())

    # Add noise points as singleton clusters
    for photo_id in noise_photos:
        clusters.append([photo_id])

    # Split large clusters
    final_clusters = []
    for cluster in clusters:
        if len(cluster) > max_cluster_size:
            # Split into smaller chunks
            for i in range(0, len(cluster), max_cluster_size):
                chunk = cluster[i:i + max_cluster_size]
                if len(chunk) >= min_cluster_size:
                    final_clusters.append(chunk)
                elif final_clusters:
                    # Add small remainder to previous
                    final_clusters[-1].extend(chunk)
                else:
                    final_clusters.append(chunk)
        else:
            final_clusters.append(cluster)

    logger.info(f"DINOv2 clustering: {n} images -> {len(final_clusters)} clusters")
    for i, cluster in enumerate(final_clusters):
        logger.debug(f"  Cluster {i}: {len(cluster)} images")

    return final_clusters


# ============================================================================
# STAGE 2: SuperPoint + LightGlue (within clusters)
# ============================================================================

# Matching thresholds - tuned for real estate photo transitions
# Goal: Group photos of SAME ROOM (even different angles) for smooth transitions
MIN_MATCHES_FOR_OVERLAP = 10      # Minimum matches before geometric verification
MIN_INLIERS_FOR_OVERLAP = 15      # Minimum inliers for pixel-level overlap (raised from 6 to avoid weak false positives)
MIN_INLIERS_FOR_DIRECTION = 3     # Lower threshold - just enough to compute direction vector
RANSAC_REPROJ_THRESHOLD = 1.5     # Tight reprojection threshold (pixels)
OVERLAP_THRESHOLD = 0.15          # Minimum score to connect photos
MIN_GEOMETRIC_SCORE_FOR_EDGE = 0.30
LOW_SCORE_MOTION_COHERENCE_MIN = 0.08
LOW_SCORE_SEGMENT_STRENGTH_MIN = 0.10
FUNDAMENTAL_SAMPSON_THRESHOLD = 0.9  # Extra strict epipolar residual filter (pixels)
ALLOW_HOMOGRAPHY_FALLBACK = False  # Fundamental geometry is preferred for parallax transitions.
MIN_HOMOGRAPHY_FALLBACK_INLIERS = 30

# Room label mismatch - only skip non-adjacent photos with different rooms
# For ADJACENT photos (temporal_dist=1), trust geometry even if room labels differ
# (ML room labels are often wrong, but adjacent photos are usually same room)
MIN_INLIERS_CROSS_ROOM = 30       # Require 30+ inliers for non-adjacent cross-room pairs
MIN_INLIERS_CROSS_ROOM_ADJACENT = 15  # Lower threshold for adjacent photos (ML often mislabels)
CROSS_ROOM_ADJ_LOW_SEMANTIC = 0.20  # Very low semantic for adjacent cross-room is suspicious
CROSS_ROOM_ADJ_MID_SEMANTIC = 0.40  # Moderate semantic still needs stricter geometry
MIN_INLIERS_CROSS_ROOM_ADJ_MID_SEMANTIC = 22
MIN_INLIERS_CROSS_ROOM_ADJ_LOW_SEMANTIC = 30

# Position gap thresholds - photos far apart in sequence need stronger evidence
# (prevents clustering photos from different physical locations with same room label)
POSITION_GAP_THRESHOLD = 3         # If gap >= 3, require higher inlier count
MIN_INLIERS_FAR_APART = 25         # Require 25+ inliers for non-adjacent same-room photos
VERY_FAR_POSITION_GAP_THRESHOLD = 20   # Very distant photos in upload order are unlikely to transition directly
MIN_SEMANTIC_FOR_VERY_FAR = 0.55       # Very-far pairs also need moderate semantic affinity
MIN_INLIERS_VERY_FAR = 35              # Enforce stronger geometry for very-far pairs
MIN_SCORE_VERY_FAR = 0.45              # Reject weak-overlap scores for very-far pairs

# Adaptive geometric matching thresholds
LOFTR_CONFIDENCE_LEVELS = (0.10,)
# Keep 0.5 as anchor but evaluate nearby thresholds and pick best-scoring result.
EFFICIENT_LOFTR_CONFIDENCE_LEVELS = (0.60, 0.50, 0.40, 0.70, 0.30, 0.20, 0.10)
LOFTR_OUTDOOR_FALLBACK_MIN_INLIERS = 12  # Run outdoor checkpoint only when indoor is weak
DEFAULT_LOFTR_INPUT_SIZE = (960, 720)  # width, height
EFFICIENT_LOFTR_MODEL_ID = "zju-community/efficientloftr"
DEFAULT_PRODUCTION_MATCHER = "loftr_kornia_indoor_native"
LOFTR_NATIVE_CONFIDENCE_THRESHOLD = 0.10
ZJU_LOFTR_NATIVE_CONFIDENCE_THRESHOLD = 0.20
ZJU_LOFTR_VARIANTS = (
    "indoor_ds",
    "indoor_ot",
    "outdoor_ds",
    "outdoor_ot",
)
LOFTR_FILTER_LOCAL_WINDOW = 15
LOFTR_FILTER_MIN_GRAD = 0.08
LOFTR_FILTER_MIN_STD = 0.07
LOFTR_FILTER_SKY_TOP_MAX_Y = 0.48
LOFTR_FILTER_CEILING_TOP_MAX_Y = 0.40
LOFTR_FILTER_FLOOR_BOTTOM_MIN_Y = 0.62
LOFTR_FILTER_WALL_MIN_Y = 0.18
LOFTR_FILTER_WALL_MAX_Y = 0.85
LOFTR_FILTER_LOW_SAT = 0.22
LOFTR_FILTER_BRIGHT = 0.55
LOFTR_FILTER_FINAL_KEEP_WEIGHT = 0.30
# Low kept-count should not look strong even if confidence mean is high.
LOFTR_FILTER_KEEP_COUNT_ZERO = 60
LOFTR_FILTER_KEEP_COUNT_TARGET = 240
NATIVE_SCORE_COUNT_ZERO = 80
NATIVE_SCORE_COUNT_TARGET = 260
NATIVE_EDGE_MIN_MATCHES = 160
NATIVE_EDGE_MIN_INLIERS = 35
NATIVE_EDGE_MIN_MEAN = 0.64
NATIVE_EDGE_MIN_MEDIAN = 0.63
NATIVE_EDGE_ALLOWED_GEOMETRY_MODELS = {"fundamental_magsac", "fundamental_ransac"}
# If native forward pass is weak, retry reverse orientation and keep the stronger result.
# This stabilizes pair-debug and clustering against directional LoFTR asymmetry.
NATIVE_REVERSE_RETRY_ENABLED = True
NATIVE_REVERSE_RETRY_MATCH_THRESHOLD = 120
NATIVE_REVERSE_RETRY_INLIER_THRESHOLD = 25
ENABLE_PHOTOMETRIC_PREFILTER = False
PHOTOMETRIC_PATCH_RADIUS = 4
PHOTOMETRIC_MIN_NCC = 0.55
PHOTOMETRIC_MIN_GRAD = 0.02

# Segment-aware overlap scoring (horizontal bands)
SEGMENT_LEFT_START = 0.25   # Left transition band starts at 25% width
SEGMENT_LEFT_END = 0.50     # Left transition band ends at 50% width
SEGMENT_RIGHT_START = 0.50  # Right transition band starts at 50% width
SEGMENT_RIGHT_END = 0.75    # Right transition band ends at 75% width
SEGMENT_CENTER_START = 0.40
SEGMENT_CENTER_END = 0.60
SEGMENT_STRONG_OVERLAP_TARGET = 0.20  # 20% inlier support in a transition segment is strong
SPATIAL_COVERAGE_TARGET = 0.12        # 12% inlier bbox coverage is treated as robust


def _safe_inlier_mask(mask: np.ndarray | None, expected_len: int) -> np.ndarray | None:
    if mask is None:
        return None
    inlier_mask = np.asarray(mask).ravel().astype(bool)
    if inlier_mask.shape[0] != expected_len:
        return None
    return inlier_mask


def _refine_fundamental_inliers_sampson(
    points0: np.ndarray,
    points1: np.ndarray,
    fundamental: np.ndarray | None,
    inlier_mask: np.ndarray,
    threshold: float = FUNDAMENTAL_SAMPSON_THRESHOLD,
) -> np.ndarray:
    """Refine fundamental-matrix inliers via Sampson residual."""
    if fundamental is None:
        return inlier_mask
    f = np.asarray(fundamental, dtype=np.float64)
    if f.shape != (3, 3) or not np.isfinite(f).all():
        return inlier_mask

    mask = inlier_mask.ravel().astype(bool)
    if int(mask.sum()) < 8:
        return mask

    p0 = points0[mask].astype(np.float64)
    p1 = points1[mask].astype(np.float64)
    ones = np.ones((p0.shape[0], 1), dtype=np.float64)
    x0 = np.hstack([p0, ones])  # Nx3
    x1 = np.hstack([p1, ones])  # Nx3

    fx0 = (f @ x0.T).T
    ftx1 = (f.T @ x1.T).T
    numer = np.square(np.sum(x1 * fx0, axis=1))
    denom = (
        np.square(fx0[:, 0])
        + np.square(fx0[:, 1])
        + np.square(ftx1[:, 0])
        + np.square(ftx1[:, 1])
        + 1e-12
    )
    sampson = numer / denom
    refined_local = sampson <= float(threshold * threshold)

    refined = np.zeros_like(mask, dtype=bool)
    refined_indices = np.where(mask)[0]
    refined[refined_indices] = refined_local
    return refined


def _compute_gradient_magnitude(image_gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(image_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image_gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _photometric_consistency_mask(
    points0: np.ndarray,
    points1: np.ndarray,
    image0: np.ndarray,
    image1: np.ndarray,
    grad0: np.ndarray,
    grad1: np.ndarray,
    patch_radius: int = PHOTOMETRIC_PATCH_RADIUS,
    min_ncc: float = PHOTOMETRIC_MIN_NCC,
    min_grad: float = PHOTOMETRIC_MIN_GRAD,
) -> np.ndarray:
    """Keep correspondences that agree in local appearance (NCC) and texture."""
    n = int(points0.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=bool)

    h0, w0 = image0.shape[:2]
    h1, w1 = image1.shape[:2]
    kept = np.zeros((n,), dtype=bool)

    for i in range(n):
        x0, y0 = points0[i]
        x1, y1 = points1[i]
        xi0 = int(round(float(x0)))
        yi0 = int(round(float(y0)))
        xi1 = int(round(float(x1)))
        yi1 = int(round(float(y1)))

        if (
            xi0 < patch_radius
            or yi0 < patch_radius
            or xi0 >= (w0 - patch_radius)
            or yi0 >= (h0 - patch_radius)
            or xi1 < patch_radius
            or yi1 < patch_radius
            or xi1 >= (w1 - patch_radius)
            or yi1 >= (h1 - patch_radius)
        ):
            continue

        if float(grad0[yi0, xi0]) < min_grad or float(grad1[yi1, xi1]) < min_grad:
            continue

        p0 = image0[
            yi0 - patch_radius: yi0 + patch_radius + 1,
            xi0 - patch_radius: xi0 + patch_radius + 1,
        ].astype(np.float32)
        p1 = image1[
            yi1 - patch_radius: yi1 + patch_radius + 1,
            xi1 - patch_radius: xi1 + patch_radius + 1,
        ].astype(np.float32)

        p0 = p0 - float(p0.mean())
        p1 = p1 - float(p1.mean())
        n0 = float(np.linalg.norm(p0))
        n1 = float(np.linalg.norm(p1))
        if n0 <= 1e-6 or n1 <= 1e-6:
            continue

        ncc = float((p0 * p1).sum() / (n0 * n1))
        if ncc >= min_ncc:
            kept[i] = True

    return kept


def _estimate_geometric_inliers(
    points0: np.ndarray,
    points1: np.ndarray,
    reproj_threshold: float = RANSAC_REPROJ_THRESHOLD,
    homography_reproj_threshold: float = 3.0,
) -> Tuple[np.ndarray | None, str]:
    """Estimate inlier mask using robust geometry, preferring stronger consensus."""
    best_fundamental_mask = None
    best_fundamental_count = -1
    best_fundamental_model = "none"

    # Fundamental matrix with MAGSAC if available.
    f_methods = []
    if hasattr(cv2, "USAC_MAGSAC"):
        f_methods.append(("fundamental_magsac", cv2.USAC_MAGSAC))
    f_methods.append(("fundamental_ransac", cv2.FM_RANSAC))

    for model_name, method in f_methods:
        try:
            fundamental, mask = cv2.findFundamentalMat(
                points0,
                points1,
                method,
                float(reproj_threshold),
                0.999,
            )
            inlier_mask = _safe_inlier_mask(mask, len(points0))
            if inlier_mask is None:
                continue
            inlier_mask = _refine_fundamental_inliers_sampson(
                points0=points0,
                points1=points1,
                fundamental=fundamental,
                inlier_mask=inlier_mask,
            )
            count = int(inlier_mask.sum())
            if count > best_fundamental_count:
                best_fundamental_mask = inlier_mask
                best_fundamental_count = count
                best_fundamental_model = model_name
        except cv2.error:
            continue

    # Fundamental inliers are preferred for real camera-motion consistency.
    if best_fundamental_mask is not None and best_fundamental_count > 0:
        return best_fundamental_mask, best_fundamental_model

    # Homography fallback (optional, strict) for nearly-planar scenes.
    if ALLOW_HOMOGRAPHY_FALLBACK:
        h_method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
        try:
            _, mask_h = cv2.findHomography(points0, points1, h_method, float(homography_reproj_threshold))
            inlier_mask_h = _safe_inlier_mask(mask_h, len(points0))
            if inlier_mask_h is not None:
                count_h = int(inlier_mask_h.sum())
                if count_h >= MIN_HOMOGRAPHY_FALLBACK_INLIERS:
                    return inlier_mask_h, "homography"
        except cv2.error:
            pass

    return None, "none"


def _resize_by_longest_side_and_pad(
    image_gray: np.ndarray,
    target_long_side: int,
    multiple: int = 8,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Resize preserving aspect ratio and pad (right/bottom) to multiple-of-N."""
    h0, w0 = image_gray.shape[:2]
    if h0 <= 0 or w0 <= 0:
        return image_gray, {"content_w": max(1, w0), "content_h": max(1, h0), "pad_w": 0, "pad_h": 0}

    long_side = max(h0, w0)
    target = max(64, int(target_long_side))
    scale = float(target) / float(long_side)

    new_w = max(1, int(round(w0 * scale)))
    new_h = max(1, int(round(h0 * scale)))
    interp = cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(image_gray, (new_w, new_h), interpolation=interp)

    pad_w = (multiple - (new_w % multiple)) % multiple
    pad_h = (multiple - (new_h % multiple)) % multiple
    padded = cv2.copyMakeBorder(
        resized,
        0,
        int(pad_h),
        0,
        int(pad_w),
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )
    return padded, {
        "content_w": int(new_w),
        "content_h": int(new_h),
        "pad_w": int(pad_w),
        "pad_h": int(pad_h),
    }


def _resize_rgb_by_longest_side_and_pad(
    image_rgb: np.ndarray,
    target_long_side: int,
    multiple: int = 8,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """RGB variant of longest-side resize + right/bottom padding."""
    h0, w0 = image_rgb.shape[:2]
    if h0 <= 0 or w0 <= 0:
        return image_rgb, {"content_w": max(1, w0), "content_h": max(1, h0), "pad_w": 0, "pad_h": 0}

    long_side = max(h0, w0)
    target = max(64, int(target_long_side))
    scale = float(target) / float(long_side)

    new_w = max(1, int(round(w0 * scale)))
    new_h = max(1, int(round(h0 * scale)))
    interp = cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=interp)

    pad_w = (multiple - (new_w % multiple)) % multiple
    pad_h = (multiple - (new_h % multiple)) % multiple
    padded = cv2.copyMakeBorder(
        resized,
        0,
        int(pad_h),
        0,
        int(pad_w),
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return padded, {
        "content_w": int(new_w),
        "content_h": int(new_h),
        "pad_w": int(pad_w),
        "pad_h": int(pad_h),
    }


def _compute_segment_scores(
    inlier_points0: np.ndarray,
    inlier_points1: np.ndarray,
    width0: int,
    height0: int,
    width1: int,
    height1: int,
) -> Dict[str, float]:
    """Compute left/right segment strength using inlier correspondences."""
    n = int(len(inlier_points0))
    if n == 0:
        return {
            "from_left_25_50": 0.0,
            "from_right_50_75": 0.0,
            "to_left_25_50": 0.0,
            "to_right_50_75": 0.0,
            "cross_left_to_right": 0.0,
            "cross_right_to_left": 0.0,
            "cross_center_to_center": 0.0,
        }

    x0 = inlier_points0[:, 0]
    x1 = inlier_points1[:, 0]

    left0 = (x0 >= SEGMENT_LEFT_START * width0) & (x0 < SEGMENT_LEFT_END * width0)
    right0 = (x0 >= SEGMENT_RIGHT_START * width0) & (x0 < SEGMENT_RIGHT_END * width0)
    left1 = (x1 >= SEGMENT_LEFT_START * width1) & (x1 < SEGMENT_LEFT_END * width1)
    right1 = (x1 >= SEGMENT_RIGHT_START * width1) & (x1 < SEGMENT_RIGHT_END * width1)
    center0 = (x0 >= SEGMENT_CENTER_START * width0) & (x0 < SEGMENT_CENTER_END * width0)
    center1 = (x1 >= SEGMENT_CENTER_START * width1) & (x1 < SEGMENT_CENTER_END * width1)

    def frac(mask: np.ndarray) -> float:
        return float(mask.sum() / n)

    return {
        "from_left_25_50": frac(left0),
        "from_right_50_75": frac(right0),
        "to_left_25_50": frac(left1),
        "to_right_50_75": frac(right1),
        "cross_left_to_right": frac(left0 & right1),
        "cross_right_to_left": frac(right0 & left1),
        "cross_center_to_center": frac(center0 & center1),
    }


def _normalized_convex_hull_area(points: np.ndarray, width: int, height: int) -> float:
    """Return convex hull coverage ratio in [0,1] for a 2D point set."""
    if points.shape[0] < 3:
        return 0.0
    pts = np.ascontiguousarray(points.astype(np.float32))
    hull = cv2.convexHull(pts)
    if hull is None or len(hull) < 3:
        return 0.0
    area = float(cv2.contourArea(hull))
    frame_area = float(max(1, width * height))
    return float(np.clip(area / frame_area, 0.0, 1.0))


def _compute_transition_geometry_components(
    num_matches: int,
    num_inliers: int,
    inlier_points0: np.ndarray,
    inlier_points1: np.ndarray,
    width0: int,
    height0: int,
    width1: int,
    height1: int,
    segment_scores: Dict[str, float],
    inlier_ratio_zero_point: float = 0.20,
    inlier_ratio_full_point: float = 1.00,
) -> Dict[str, float]:
    """Return detailed score components used by transition geometry scoring."""
    if num_matches <= 0 or num_inliers <= 0:
        return {
            "inlier_ratio": 0.0,
            "inlier_ratio_term": 0.0,
            "inlier_volume_bonus": 0.0,
            "spread_area0": 0.0,
            "spread_area1": 0.0,
            "spread_bonus": 0.0,
            "segment_strength": 0.0,
            "segment_bonus": 0.0,
            "motion_coherence": 0.0,
            "final_score": 0.0,
        }

    inlier_ratio = float(num_inliers / max(1, num_matches))
    zero_point = float(np.clip(inlier_ratio_zero_point, 0.0, 0.99))
    full_point = float(np.clip(inlier_ratio_full_point, zero_point + 1e-6, 1.0))
    inlier_ratio_term = float(np.clip((inlier_ratio - zero_point) / (full_point - zero_point), 0.0, 1.0))
    inlier_volume_bonus = min(1.0, float(num_inliers) / 140.0)

    area0 = _normalized_convex_hull_area(inlier_points0, width=width0, height=height0)
    area1 = _normalized_convex_hull_area(inlier_points1, width=width1, height=height1)
    spread_bonus = min(1.0, min(area0, area1) / SPATIAL_COVERAGE_TARGET)

    transition_segment_strength = max(
        float(segment_scores.get("cross_left_to_right", 0.0)),
        float(segment_scores.get("cross_right_to_left", 0.0)),
        float(segment_scores.get("cross_center_to_center", 0.0)),
    )
    segment_bonus = min(1.0, transition_segment_strength / SEGMENT_STRONG_OVERLAP_TARGET)

    deltas = inlier_points1 - inlier_points0
    norms = np.linalg.norm(deltas, axis=1)
    valid = norms > 1e-6
    if int(valid.sum()) >= 3:
        unit = deltas[valid] / norms[valid][:, None]
        motion_coherence = float(np.linalg.norm(unit.mean(axis=0)))
    else:
        motion_coherence = 0.0

    score = (
        inlier_ratio_term
        * (0.45 + 0.55 * inlier_volume_bonus)
        * (0.35 + 0.65 * spread_bonus)
        * (0.40 + 0.60 * segment_bonus)
        * (0.45 + 0.55 * motion_coherence)
    )
    final_score = float(np.clip(score, 0.0, 1.0))

    return {
        "inlier_ratio": float(inlier_ratio),
        "inlier_ratio_term": float(inlier_ratio_term),
        "inlier_ratio_zero_point": float(zero_point),
        "inlier_ratio_full_point": float(full_point),
        "inlier_volume_bonus": float(inlier_volume_bonus),
        "spread_area0": float(area0),
        "spread_area1": float(area1),
        "spread_bonus": float(spread_bonus),
        "segment_strength": float(transition_segment_strength),
        "segment_bonus": float(segment_bonus),
        "motion_coherence": float(motion_coherence),
        "final_score": float(final_score),
    }


def _compute_transition_geometry_score(
    num_matches: int,
    num_inliers: int,
    inlier_points0: np.ndarray,
    inlier_points1: np.ndarray,
    width0: int,
    height0: int,
    width1: int,
    height1: int,
    segment_scores: Dict[str, float],
) -> float:
    """Compute transition-safe geometric score using robust spatial/motion evidence."""
    components = _compute_transition_geometry_components(
        num_matches=num_matches,
        num_inliers=num_inliers,
        inlier_points0=inlier_points0,
        inlier_points1=inlier_points1,
        width0=width0,
        height0=height0,
        width1=width1,
        height1=height1,
        segment_scores=segment_scores,
    )
    return float(components["final_score"])


def _sample_normalized_matches(
    points0: np.ndarray,
    points1: np.ndarray,
    width0: int,
    height0: int,
    width1: int,
    height1: int,
    max_points: int | None = 300,
) -> List[Dict[str, float]]:
    """Create normalized correspondence payload for debug visualization."""
    n = min(len(points0), len(points1))
    if n <= 0 or width0 <= 0 or height0 <= 0 or width1 <= 0 or height1 <= 0:
        return []

    if max_points is None or max_points <= 0 or n <= max_points:
        indices = np.arange(n, dtype=np.int32)
    else:
        indices = np.linspace(0, n - 1, num=max_points, dtype=np.int32)

    points0 = points0[indices]
    points1 = points1[indices]
    width0_f = float(width0)
    height0_f = float(height0)
    width1_f = float(width1)
    height1_f = float(height1)

    payload: List[Dict[str, float]] = []
    for p0, p1 in zip(points0, points1):
        x0 = float(np.clip(p0[0] / width0_f, 0.0, 1.0))
        y0 = float(np.clip(p0[1] / height0_f, 0.0, 1.0))
        x1 = float(np.clip(p1[0] / width1_f, 0.0, 1.0))
        y1 = float(np.clip(p1[1] / height1_f, 0.0, 1.0))
        payload.append({
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "dx": float(x1 - x0),
            "dy": float(y1 - y0),
        })
    return payload


def cross_room_min_inliers_required(temporal_dist: int, sem_sim: float) -> int:
    """Adaptive threshold for cross-room temporal pairs."""
    if temporal_dist != 1:
        return MIN_INLIERS_CROSS_ROOM
    if sem_sim < CROSS_ROOM_ADJ_LOW_SEMANTIC:
        return MIN_INLIERS_CROSS_ROOM_ADJ_LOW_SEMANTIC
    if sem_sim < CROSS_ROOM_ADJ_MID_SEMANTIC:
        return MIN_INLIERS_CROSS_ROOM_ADJ_MID_SEMANTIC
    return MIN_INLIERS_CROSS_ROOM_ADJACENT


def accepts_very_far_pair(position_gap: int, sem_sim: float, num_inliers: int, score: float) -> bool:
    """Apply very-far sequence gap safety constraints."""
    if position_gap < VERY_FAR_POSITION_GAP_THRESHOLD:
        return True
    if sem_sim < MIN_SEMANTIC_FOR_VERY_FAR:
        return False
    if num_inliers < MIN_INLIERS_VERY_FAR:
        return False
    if score < MIN_SCORE_VERY_FAR:
        return False
    return True


def blend_geometric_semantic_score(geometric_score: float, semantic_score: float) -> float:
    """Blend geometry + semantic affinity while keeping geometry dominant."""
    geo = float(np.clip(geometric_score, 0.0, 1.0))
    sem = float(np.clip(semantic_score, 0.0, 1.0))
    w_geo = max(0.0, float(GEOMETRIC_SCORE_WEIGHT))
    w_sem = max(0.0, float(SEMANTIC_SCORE_WEIGHT))
    if (w_geo + w_sem) <= 1e-8:
        return geo
    return float(np.clip(((w_geo * geo) + (w_sem * sem)) / (w_geo + w_sem), 0.0, 1.0))


def geometry_quality_gate(
    geometric_score: float | None,
    diagnostics: Dict[str, Any] | None,
    min_score: float = MIN_GEOMETRIC_SCORE_FOR_EDGE,
) -> bool:
    """Accept only geometry with sufficient quality, not inlier count alone."""
    if geometric_score is None:
        return False
    score = float(geometric_score)
    if score < min_score:
        return False

    if not isinstance(diagnostics, dict):
        return True

    score_components = diagnostics.get("score_components")
    if not isinstance(score_components, dict):
        return True

    motion = float(score_components.get("motion_coherence", 0.0) or 0.0)
    segment_strength = float(score_components.get("segment_strength", 0.0) or 0.0)

    # Low-mid scores are fragile on repetitive textures; require stronger structure/motion.
    if score < 0.40 and motion < LOW_SCORE_MOTION_COHERENCE_MIN and segment_strength < LOW_SCORE_SEGMENT_STRENGTH_MIN:
        return False
    return True


def strict_geometry_edge_gate(
    num_matches: int | None,
    num_inliers: int | None,
    geometric_score: float | None,
    diagnostics: Dict[str, Any] | None,
    min_inliers_required: int,
) -> bool:
    """Strict production gate for cluster edges.

    Requires:
    - robust fundamental geometry model
    - enough matches/inliers
    - strong native confidence distribution
    - geometry quality score
    """
    if num_matches is None or num_inliers is None:
        return False
    if num_matches < NATIVE_EDGE_MIN_MATCHES:
        return False
    required_inliers = max(int(min_inliers_required), int(NATIVE_EDGE_MIN_INLIERS))
    if num_inliers < required_inliers:
        return False
    if not geometry_quality_gate(geometric_score, diagnostics):
        return False
    if not isinstance(diagnostics, dict):
        return False

    geom_model = str(diagnostics.get("geometry_model") or "").strip().lower()
    if geom_model not in NATIVE_EDGE_ALLOWED_GEOMETRY_MODELS:
        return False

    native_scores = diagnostics.get("native_matching_scores")
    if not isinstance(native_scores, dict):
        return False
    native_mean = float(native_scores.get("mean", 0.0) or 0.0)
    native_median = float(native_scores.get("median", 0.0) or 0.0)
    if native_mean < NATIVE_EDGE_MIN_MEAN:
        return False
    if native_median < NATIVE_EDGE_MIN_MEDIAN:
        return False
    return True


def rooms_are_different(room1: str, room2: str) -> bool:
    """Check if two room labels represent different rooms.

    Returns True if the rooms are clearly different types.
    Returns False if they're the same or unknown.
    """
    if room1 is None or room2 is None:
        return False
    if room1 == "unknown" or room2 == "unknown":
        return False
    if room1 == room2:
        return False

    # Normalize labels
    r1 = room1.lower().strip()
    r2 = room2.lower().strip()

    # Same label after normalization
    if r1 == r2:
        return False

    # Check for similar room types that should be allowed to connect
    # (e.g., "living_room" and "living room" are the same)
    r1_words = set(r1.replace("_", " ").split())
    r2_words = set(r2.replace("_", " ").split())

    # If they share key room words, they might be same room with slight labeling difference
    key_words = {"living", "dining", "kitchen", "bedroom", "bathroom", "patio", "exterior", "outdoor", "yard", "pool"}
    r1_keys = r1_words & key_words
    r2_keys = r2_words & key_words

    # If they share key words, they're likely the same room type
    if r1_keys and r2_keys and r1_keys == r2_keys:
        return False

    # Otherwise they're different room types
    return True

# Temporal + semantic matching for same-room different-angle shots
# Adjacent photos in upload order are usually the same room
TEMPORAL_WINDOW = 2               # Check photos within ±2 positions
TEMPORAL_SEMANTIC_THRESHOLD = 0.88  # High confidence same room (balanced - not too strict)
TEMPORAL_GEOMETRIC_THRESHOLD = 0.60  # Lower threshold if we also verify geometrically
NEIGHBOR_TRUST_THRESHOLD = 0.63   # Trust immediate neighbors with strong semantic support
HIGH_CONFIDENCE_NEIGHBOR_TRUST = 0.83  # Very strong adjacent semantic support (can pass without extra context)
NEIGHBOR_SUPPORT_THRESHOLD = 0.70  # Support from nearby photos for semantic-only adjacent fallback
SAME_LABEL_NEIGHBOR_TRUST = 0.70  # Adjacent pairs with same room label can pass with moderate semantic
AMBIGUOUS_SAME_LABEL_TRUST = 0.84  # Stricter threshold for ambiguous repeated rooms
DIST2_TRUST_THRESHOLD = 0.75      # Conservative trust threshold for temporal distance=2 pairs
SAME_LABEL_DIST2_TRUST = 0.70     # Dist-2 same-label fallback to reduce over-splitting
AMBIGUOUS_DIST2_TRUST = 0.82      # Keep stricter dist-2 fallback for ambiguous rooms
CROSS_ROOM_RECOVERY_THRESHOLD = 0.72  # Adjacent cross-room fallback for known room-family confusions
SERVICE_ROOM_RECOVERY_THRESHOLD = 0.66  # Slightly lower for bathroom<->laundry confusions
CROSS_ROOM_RECOVERY_TOPK = 2       # Require mutual top-K semantic affinity for cross-room fallback
EXTERIOR_LONG_GAP_SEMANTIC_TRUST = 0.78  # Recover long-gap exterior pairs when geometry fails
EXTERIOR_LONG_GAP_MIN = 5          # Minimum sequence gap for long-gap exterior semantic recovery

# ORB pre-filter thresholds (for performance - ORB is ~10x faster than LoFTR)
ORB_QUICK_REJECT_INLIERS = 2      # If ORB finds ≤2 inliers, skip LoFTR (definitely no overlap)
ORB_QUICK_ACCEPT_INLIERS = 15     # If ORB finds ≥15 inliers, skip LoFTR (definitely overlap)

# Minimum DINOv2 similarity to even consider geometric verification
# Filters out obviously unrelated photos (aerial vs interior = ~0.05)
MIN_SEMANTIC_FOR_GEOMETRIC = 0.15  # Skip geometric check if semantic < 15%

# Cluster ordering thresholds
MIN_TRANSITION_SCORE = 0.20  # Minimum overlap score to keep photo in ordered chain
DIRECTION_CONSISTENCY_THRESHOLD = 0.5  # Cos similarity for direction vectors to be "consistent"
SEMANTIC_BRIDGE_SUPPORT_THRESHOLD = 0.70  # Remove weak semantic-only bridges if cross-support is low
HARD_TRANSITION_MIN_OVERLAP = 0.25  # Require at least ~25% overlap-like score for final transitions
HARD_TRANSITION_MIN_INLIERS = 8  # Reject transition edges without enough geometric correspondences
HARD_TRANSITION_MIN_SIDE_OVERLAP = settings.HARD_TRANSITION_MIN_SIDE_OVERLAP
HARD_TRANSITION_MIN_CENTER_OVERLAP = settings.HARD_TRANSITION_MIN_CENTER_OVERLAP
HARD_TRANSITION_MIN_OVERLAP_RATIO = settings.HARD_TRANSITION_MIN_OVERLAP_RATIO
HARD_TRANSITION_ADJACENT_SEMANTIC_TRUST = NEIGHBOR_TRUST_THRESHOLD  # Keep trusted adjacent semantic links
HARD_TRANSITION_DIST2_SEMANTIC_TRUST = DIST2_TRUST_THRESHOLD  # Keep trusted distance-2 semantic links
HARD_TRANSITION_MAX_SEMANTIC_GAP = 2  # Split semantic-only transitions that jump too far in capture order
HARD_TRANSITION_RECOVERY_SEMANTIC_TRUST = 0.86  # semantic-recovery edges require much stronger confidence
HARD_TRANSITION_FRONT_RECOVERY_SEMANTIC_TRUST = 0.78  # Keep strong long-gap front-exterior recovery links
HARD_TRANSITION_REQUIRE_GEOMETRY = settings.REQUIRE_GEOMETRIC_TRANSITIONS
HARD_TRANSITION_REQUIRE_DIRECTION = settings.REQUIRE_DIRECTION_FOR_TRANSITIONS
GEOMETRIC_SCORE_WEIGHT = settings.GEOMETRIC_SCORE_WEIGHT
SEMANTIC_SCORE_WEIGHT = settings.SEMANTIC_SCORE_WEIGHT

# Component graph safety gates for semantic-only links.
STRICT_SEMANTIC_COMPONENT_CONNECTIVITY = settings.STRICT_SEMANTIC_COMPONENT_CONNECTIVITY
GEOMETRY_ONLY_CLUSTER_MEMBERSHIP = settings.GEOMETRY_ONLY_CLUSTER_MEMBERSHIP
COMPONENT_SEMANTIC_ADJ_MIN = settings.COMPONENT_SEMANTIC_ADJ_MIN
COMPONENT_SEMANTIC_DIST2_MIN = settings.COMPONENT_SEMANTIC_DIST2_MIN
COMPONENT_SEMANTIC_RECOVERY_MIN = settings.COMPONENT_SEMANTIC_RECOVERY_MIN
COMPONENT_FRONT_RECOVERY_MIN = settings.COMPONENT_FRONT_RECOVERY_MIN
COMPONENT_SEMANTIC_MAX_GAP = settings.COMPONENT_SEMANTIC_MAX_GAP
COMPONENT_SAME_LABEL_ADJ_MIN = settings.COMPONENT_SAME_LABEL_ADJ_MIN
COMPONENT_SAME_LABEL_DIST2_MIN = settings.COMPONENT_SAME_LABEL_DIST2_MIN
COMPONENT_AMBIGUOUS_SAME_LABEL_MIN = settings.COMPONENT_AMBIGUOUS_SAME_LABEL_MIN
COMPONENT_CROSS_LABEL_ADJ_MIN = settings.COMPONENT_CROSS_LABEL_ADJ_MIN
COMPONENT_CROSS_LABEL_DIST2_MIN = settings.COMPONENT_CROSS_LABEL_DIST2_MIN

# Deduplication thresholds
# Only consecutive photos with very high similarity are considered "same angle" duplicates
# Different angles of the same room typically have 0.85-0.92 similarity
DUPLICATE_SIMILARITY_THRESHOLD = 0.97  # Conservative dedup to avoid dropping transition-valuable near-duplicates
DUPLICATE_GEOMETRIC_SEMANTIC_THRESHOLD = 0.95  # Allow slightly lower semantic threshold when geometry is near-identical
DUPLICATE_GEOMETRIC_OVERLAP_THRESHOLD = 0.90  # Require very high overlap score for geometric-backed dedupe
KEEP_DUPLICATE_SINGLETON_CLUSTERS = not settings.DELETE_OBVIOUS_DUPLICATES
SPLIT_LONG_GAP_DECAY = 0.25  # Prefer cutting oversized clusters at large capture-order jumps


def compute_direction_vector(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    inlier_mask: np.ndarray,
) -> Tuple[float, float]:
    """Compute camera motion direction from matched keypoints.

    Returns the centroid shift vector (dx, dy) indicating how content
    moved from image 0 to image 1.

    If content shifted LEFT (negative dx), camera moved RIGHT.
    If content shifted UP (negative dy), camera moved DOWN.

    Returns:
        (dx, dy) normalized direction vector, or (0, 0) if insufficient matches
    """
    if inlier_mask is None or inlier_mask.sum() < 4:
        return (0.0, 0.0)

    # Get inlier points only
    inliers = inlier_mask.ravel().astype(bool)
    pts0 = mkpts0[inliers]
    pts1 = mkpts1[inliers]

    # Compute centroids
    centroid0 = pts0.mean(axis=0)
    centroid1 = pts1.mean(axis=0)

    # Direction = how content shifted from img0 to img1
    dx = centroid1[0] - centroid0[0]
    dy = centroid1[1] - centroid0[1]

    # Normalize
    magnitude = np.sqrt(dx * dx + dy * dy)
    if magnitude < 1e-6:
        return (0.0, 0.0)

    return (dx / magnitude, dy / magnitude)


def directions_consistent(dir1: Tuple[float, float], dir2: Tuple[float, float]) -> bool:
    """Check if two direction vectors are roughly consistent.

    Consistent means they point in similar or same direction (not opposite).
    Uses cosine similarity: > 0 means same general direction.
    """
    if dir1 == (0.0, 0.0) or dir2 == (0.0, 0.0):
        return True  # Unknown direction - assume OK

    cos_sim = dir1[0] * dir2[0] + dir1[1] * dir2[1]
    return cos_sim > DIRECTION_CONSISTENCY_THRESHOLD


def has_local_semantic_support(i: int, j: int, similarity: np.ndarray) -> bool:
    """Check whether an adjacent pair has local sequence context support.

    Semantic-only fallback is safer when at least one nearby photo also has
    reasonably high similarity with one of the pair endpoints.
    """
    n = similarity.shape[0]
    neighbor_indices = {i - 1, i + 1, j - 1, j + 1}
    support_scores = []
    for k in neighbor_indices:
        if k < 0 or k >= n or k == i or k == j:
            continue
        support_scores.append(max(float(similarity[i, k]), float(similarity[j, k])))

    return bool(support_scores) and max(support_scores) >= NEIGHBOR_SUPPORT_THRESHOLD


def normalize_room_label(room: str | None) -> str:
    return (room or "").strip().lower().replace("_", " ")


def room_family(room: str | None) -> str:
    room_norm = normalize_room_label(room)
    if not room_norm or room_norm == "unknown":
        return "unknown"

    if any(token in room_norm for token in ("living", "dining", "kitchen", "entrance", "foyer", "hallway")):
        return "social"
    if any(token in room_norm for token in ("bathroom", "laundry", "powder")):
        return "service"
    if any(token in room_norm for token in ("front yard", "exterior front", "driveway", "curb", "porch")):
        return "front_exterior"
    if any(token in room_norm for token in ("patio", "deck", "backyard", "garden", "pool", "exterior")):
        return "exterior"
    return room_norm


def rooms_allow_adjacent_semantic_bridge(room1: str | None, room2: str | None) -> bool:
    fam1 = room_family(room1)
    fam2 = room_family(room2)
    if fam1 == "unknown" or fam2 == "unknown":
        return False
    if fam1 != fam2:
        return False
    return fam1 in {"social", "service", "front_exterior"}


def is_exterior_like(room: str | None) -> bool:
    return room_family(room) in {"front_exterior", "exterior"}


def is_mutual_top_semantic_neighbor(i: int, j: int, similarity: np.ndarray, top_k: int = 2) -> bool:
    """Require mutual top semantic affinity to avoid weak cross-room fallbacks."""
    sorted_i = [idx for idx in np.argsort(-similarity[i]) if idx != i][:top_k]
    sorted_j = [idx for idx in np.argsort(-similarity[j]) if idx != j][:top_k]
    return (j in sorted_i) and (i in sorted_j)


def order_cluster_for_transitions(
    cluster_indices: List[int],
    photo_ids: List[int],
    adjacency: np.ndarray,
    directions: dict,
    min_score: float = MIN_TRANSITION_SCORE,
) -> Tuple[List[int], List[int]]:
    """Order photos within a cluster for optimal video transitions.

    Uses direction-aware path-finding to create a chain where:
    1. Each consecutive pair has good visual overlap
    2. Camera motion direction is consistent (no sudden reversals)
    3. Starts from an ENDPOINT (leftmost/rightmost in spatial order)

    Args:
        cluster_indices: Indices into photo_ids for this cluster
        photo_ids: Full list of photo IDs
        adjacency: Full NxN adjacency matrix (overlap scores)
        directions: Dict of (i,j) -> (dx, dy) direction vectors
        min_score: Minimum overlap score for valid transition

    Returns:
        Tuple of (ordered_main, isolated):
        - ordered_main: Ordered list of photo IDs for the main path
        - isolated: List of photo IDs not connected to main path
    """
    if len(cluster_indices) <= 1:
        return ([photo_ids[i] for i in cluster_indices], [])

    if len(cluster_indices) == 2:
        return ([photo_ids[i] for i in cluster_indices], [])

    n = len(cluster_indices)

    # Find ENDPOINT to start from (not the center!)
    # An endpoint has directions pointing mostly one way (it's at an edge of the scene)
    # A center photo has directions pointing both ways
    endpoint_scores = np.zeros(n)

    for i, idx_i in enumerate(cluster_indices):
        # Collect all direction vectors FROM this photo TO its neighbors
        outgoing_directions = []
        for j, idx_j in enumerate(cluster_indices):
            if i == j:
                continue
            if adjacency[idx_i, idx_j] < min_score:
                continue

            pair_key = (min(idx_i, idx_j), max(idx_i, idx_j))
            pair_dir = directions.get(pair_key, (0.0, 0.0))

            # Flip if we stored direction for the reverse pair
            if idx_i > idx_j:
                pair_dir = (-pair_dir[0], -pair_dir[1])

            if pair_dir != (0.0, 0.0):
                outgoing_directions.append(pair_dir)

        if len(outgoing_directions) >= 1:
            # Calculate how consistent the outgoing directions are
            # Endpoints have all directions pointing the same way
            # Centers have directions pointing opposite ways (cancel out)
            avg_dx = sum(d[0] for d in outgoing_directions) / len(outgoing_directions)
            avg_dy = sum(d[1] for d in outgoing_directions) / len(outgoing_directions)
            # Magnitude of average = consistency (1.0 = all same direction, 0.0 = cancel out)
            consistency = np.sqrt(avg_dx * avg_dx + avg_dy * avg_dy)
            endpoint_scores[i] = consistency
        else:
            # No direction info - use connection count as fallback
            endpoint_scores[i] = 0.5

    # Start from the best endpoint (highest consistency = most "edge-like")
    # If tie, prefer photo with fewer connections (more likely to be endpoint)
    connection_counts = np.zeros(n)
    for i, idx_i in enumerate(cluster_indices):
        for j, idx_j in enumerate(cluster_indices):
            if i != j and adjacency[idx_i, idx_j] >= min_score:
                connection_counts[i] += 1

    # Combine: high endpoint score, low connection count
    # Normalize and combine
    if endpoint_scores.max() > 0:
        endpoint_scores_norm = endpoint_scores / endpoint_scores.max()
    else:
        endpoint_scores_norm = endpoint_scores

    if connection_counts.max() > 0:
        connection_penalty = connection_counts / connection_counts.max()
    else:
        connection_penalty = connection_counts

    combined_score = endpoint_scores_norm - 0.3 * connection_penalty
    start_local = int(np.argmax(combined_score))

    endpoint_info = [(photo_ids[cluster_indices[i]], round(endpoint_scores[i], 2)) for i in range(n)]
    logger.debug(f"Endpoint scores: {endpoint_info}")
    logger.debug(f"Starting from endpoint: {photo_ids[cluster_indices[start_local]]}")

    # Greedy path building with direction awareness
    ordered_local = [start_local]
    remaining = set(range(n)) - {start_local}
    current_direction = (0.0, 0.0)  # Unknown initially

    while remaining:
        current_local = ordered_local[-1]
        current_idx = cluster_indices[current_local]

        # Find best next photo considering overlap AND direction
        best_next = None
        best_score = -1
        best_direction = (0.0, 0.0)

        for candidate_local in remaining:
            candidate_idx = cluster_indices[candidate_local]
            score = adjacency[current_idx, candidate_idx]

            if score < min_score:
                continue

            # Get direction for this transition
            pair_key = (min(current_idx, candidate_idx), max(current_idx, candidate_idx))
            pair_dir = directions.get(pair_key, (0.0, 0.0))

            # Flip direction if we're going in reverse order
            if current_idx > candidate_idx:
                pair_dir = (-pair_dir[0], -pair_dir[1])

            # Check direction consistency
            if current_direction != (0.0, 0.0) and pair_dir != (0.0, 0.0):
                if not directions_consistent(current_direction, pair_dir):
                    # Direction reversal - penalize score
                    score *= 0.5
                    logger.debug(f"Direction reversal penalty: {photo_ids[current_idx]}->{photo_ids[candidate_idx]}")

            if score > best_score:
                best_score = score
                best_next = candidate_local
                best_direction = pair_dir

        if best_next is None:
            # No good connection - try extending from the other end
            if len(ordered_local) > 1:
                first_local = ordered_local[0]
                first_idx = cluster_indices[first_local]

                for candidate_local in remaining:
                    candidate_idx = cluster_indices[candidate_local]
                    score = adjacency[first_idx, candidate_idx]

                    if score > best_score and score >= min_score:
                        best_score = score
                        best_next = candidate_local
                        # Will prepend, so this becomes the new first

                if best_next is not None:
                    ordered_local.insert(0, best_next)
                    remaining.remove(best_next)
                    continue

            # Fallback: insert by strongest connection to ANY point in the chain.
            # This prevents unnecessary singleton isolation in connected components
            # when endpoint extension gets stuck.
            insert_candidate = None
            insert_anchor = None
            insert_score = -1.0
            for candidate_local in remaining:
                candidate_idx = cluster_indices[candidate_local]
                for anchor_local in ordered_local:
                    anchor_idx = cluster_indices[anchor_local]
                    score = float(adjacency[anchor_idx, candidate_idx])
                    if score > insert_score:
                        insert_score = score
                        insert_candidate = candidate_local
                        insert_anchor = anchor_local

            relaxed_threshold = min_score * 0.8
            if (
                insert_candidate is not None
                and insert_anchor is not None
                and insert_score >= relaxed_threshold
            ):
                anchor_pos = ordered_local.index(insert_anchor)
                if anchor_pos <= len(ordered_local) // 2:
                    ordered_local.insert(0, insert_candidate)
                else:
                    ordered_local.append(insert_candidate)
                remaining.remove(insert_candidate)
                logger.debug(
                    "Inserted non-endpoint candidate %s with relaxed score %.3f",
                    photo_ids[cluster_indices[insert_candidate]],
                    insert_score,
                )
                continue

            # Still no good connection - these are isolated photos
            # Don't drop them - they'll be handled as separate mini-clusters
            logger.info(f"Cannot connect {len(remaining)} remaining photos - will create separate clusters")
            break

        ordered_local.append(best_next)
        remaining.remove(best_next)
        current_direction = best_direction

    # Convert back to photo IDs
    ordered_photo_ids = [photo_ids[cluster_indices[i]] for i in ordered_local]

    # Also return remaining photos as separate singleton clusters
    # They couldn't be ordered but they're still part of this connected component
    remaining_photo_ids = [photo_ids[cluster_indices[i]] for i in remaining]

    if remaining_photo_ids:
        logger.info(f"Cluster ordering: main chain={len(ordered_photo_ids)}, isolated={len(remaining_photo_ids)}")
        # Return as tuple: (ordered_main_chain, list_of_isolated_photos)
        return (ordered_photo_ids, remaining_photo_ids)

    return (ordered_photo_ids, [])


def deduplicate_and_split_cluster(
    cluster_photo_ids: List[int],
    photo_ids: List[int],
    embeddings: np.ndarray,
    adjacency: np.ndarray,
    max_size: int = 3,
    keep_duplicate_singletons: bool = True,
) -> Tuple[List[List[int]], Dict[int, int]]:
    """Remove duplicates and split large clusters into smaller ones.

    Photos are ALREADY ORDERED for optimal transitions. This function:
    1. Identifies duplicates (same angle shots) based on semantic similarity
    2. Removes duplicates that contribute LESS to transition smoothness
    3. If still > max_size, SPLITS into multiple clusters instead of dropping

    Key insight: If we have 6 good photos, we should make 2 clusters of 3,
    not drop 3 photos. This preserves content while keeping clusters manageable.

    Args:
        cluster_photo_ids: Photo IDs in this cluster (already ordered for transitions)
        photo_ids: Full list of all photo IDs (for indexing)
        embeddings: NxD normalized embedding matrix for all photos
        adjacency: NxN geometric overlap matrix (higher = better overlap)
        max_size: Maximum photos per cluster
        keep_duplicate_singletons: When True, duplicate photos are returned as
            singleton clusters. When False, duplicate photos are dropped.

    Returns:
        Tuple:
        - List of photo ID lists (one or more clusters, each with max_size or fewer photos)
        - duplicate_of map: duplicate_photo_id -> canonical_photo_id
    """
    if not cluster_photo_ids:
        return [], {}

    # Map photo IDs to matrix indices
    pid_to_idx = {pid: i for i, pid in enumerate(photo_ids)}
    cluster_indices = [pid_to_idx[pid] for pid in cluster_photo_ids if pid in pid_to_idx]

    if not cluster_indices:
        return [cluster_photo_ids], {}

    n = len(cluster_indices)
    if n == 1:
        return [cluster_photo_ids], {}

    # Compute semantic similarity within cluster
    cluster_embeddings = embeddings[cluster_indices]
    sem_sim = cluster_embeddings @ cluster_embeddings.T

    # Compute TRANSITION SCORE for each photo (overlap with prev + next in sequence)
    transition_scores = np.zeros(n)
    for i in range(n):
        idx_i = cluster_indices[i]
        if i > 0:
            idx_prev = cluster_indices[i - 1]
            transition_scores[i] += adjacency[idx_i, idx_prev]
        if i < n - 1:
            idx_next = cluster_indices[i + 1]
            transition_scores[i] += adjacency[idx_i, idx_next]

    # Endpoints get bonus
    transition_scores[0] += 0.5
    transition_scores[n - 1] += 0.5

    # Find CONSECUTIVE duplicates only
    to_remove = set()
    duplicate_of: Dict[int, int] = {}
    for i in range(n - 1):
        if i in to_remove:
            continue
        j = i + 1
        if j in to_remove:
            continue

        idx_i = cluster_indices[i]
        idx_j = cluster_indices[j]
        semantic_score = float(sem_sim[i, j])
        geometric_overlap = float(adjacency[idx_i, idx_j])
        is_duplicate_pair = (
            semantic_score >= DUPLICATE_SIMILARITY_THRESHOLD
            or (
                semantic_score >= DUPLICATE_GEOMETRIC_SEMANTIC_THRESHOLD
                and geometric_overlap >= DUPLICATE_GEOMETRIC_OVERLAP_THRESHOLD
            )
        )

        if is_duplicate_pair:
            if transition_scores[i] >= transition_scores[j]:
                to_remove.add(j)
                duplicate_of[cluster_photo_ids[j]] = cluster_photo_ids[i]
                logger.debug(
                    "Marking consecutive duplicate %s -> %s (sem=%.3f, overlap=%.3f)",
                    cluster_photo_ids[j],
                    cluster_photo_ids[i],
                    semantic_score,
                    geometric_overlap,
                )
            else:
                to_remove.add(i)
                duplicate_of[cluster_photo_ids[i]] = cluster_photo_ids[j]
                logger.debug(
                    "Marking consecutive duplicate %s -> %s (sem=%.3f, overlap=%.3f)",
                    cluster_photo_ids[i],
                    cluster_photo_ids[j],
                    semantic_score,
                    geometric_overlap,
                )

    # Keep canonical photos in order, split if needed.
    remaining_indices = [i for i in range(n) if i not in to_remove]
    remaining = [cluster_photo_ids[i] for i in remaining_indices]
    duplicate_clusters = [[cluster_photo_ids[i]] for i in range(n) if i in to_remove]
    emitted_duplicate_clusters = duplicate_clusters if keep_duplicate_singletons else []

    if len(remaining) <= max_size:
        if len(remaining) < len(cluster_photo_ids):
            if keep_duplicate_singletons:
                logger.info(
                    "Deduplicated: %s -> %s canonical (+%s duplicate singleton clusters)",
                    len(cluster_photo_ids),
                    len(remaining),
                    len(duplicate_clusters),
                )
            else:
                logger.info(
                    "Deduplicated: %s -> %s canonical (dropped %s duplicates)",
                    len(cluster_photo_ids),
                    len(remaining),
                    len(duplicate_clusters),
                )
        if not remaining:
            return emitted_duplicate_clusters, duplicate_of
        return [remaining] + emitted_duplicate_clusters, duplicate_of

    # Too many photos - SPLIT into multiple clusters instead of dropping.
    # Use constrained partitioning to GUARANTEE each cluster size <= max_size.
    m = len(remaining)
    num_clusters = (m + max_size - 1) // max_size  # Minimum required clusters

    # Find transition strengths between consecutive photos
    remaining_cluster_indices = [pid_to_idx[pid] for pid in remaining]
    transition_strengths = []
    for i in range(m - 1):
        idx_i = remaining_cluster_indices[i]
        idx_j = remaining_cluster_indices[i + 1]
        strength = adjacency[idx_i, idx_j]
        transition_strengths.append(float(strength))

    # Dynamic programming with exact cluster count (minimum needed), while
    # enforcing each segment length <= max_size and preferring weak boundaries.
    inf = float("inf")
    dp = [[inf] * (m + 1) for _ in range(num_clusters + 1)]
    prev = [[None] * (m + 1) for _ in range(num_clusters + 1)]
    dp[0][0] = 0.0

    for c in range(1, num_clusters + 1):
        min_pos = c  # at least one photo per cluster
        max_pos = min(m, c * max_size)
        for pos in range(min_pos, max_pos + 1):
            for seg_len in range(1, max_size + 1):
                start = pos - seg_len
                if start < (c - 1):
                    continue
                prev_cost = dp[c - 1][start]
                if prev_cost == inf:
                    continue

                # Cost of adding boundary before this segment.
                # Prefer splitting across large capture-order gaps, even when raw
                # geometric score is moderate, because these are often different
                # room instances with similar semantics.
                if start == 0:
                    boundary_cost = 0.0
                else:
                    boundary_strength = float(transition_strengths[start - 1])
                    left_idx = int(remaining_cluster_indices[start - 1])
                    right_idx = int(remaining_cluster_indices[start])
                    order_gap = abs(right_idx - left_idx)
                    if order_gap > 1:
                        boundary_strength /= (1.0 + (order_gap - 1) * SPLIT_LONG_GAP_DECAY)
                    boundary_cost = boundary_strength
                cand = prev_cost + boundary_cost
                if cand < dp[c][pos]:
                    dp[c][pos] = cand
                    prev[c][pos] = start

    # Reconstruct optimal partition; fallback to fixed chunking if unreachable.
    result_clusters = []
    if dp[num_clusters][m] != inf:
        boundaries = []
        c = num_clusters
        pos = m
        while c > 0:
            start = prev[c][pos]
            if start is None:
                boundaries = []
                break
            boundaries.append((start, pos))
            pos = start
            c -= 1

        boundaries.reverse()
        for start, end in boundaries:
            result_clusters.append(remaining[start:end])

    if not result_clusters:
        # Deterministic hard fallback: contiguous chunks, each <= max_size.
        for i in range(0, m, max_size):
            result_clusters.append(remaining[i:i + max_size])

    # Safety invariant: never return oversize clusters.
    oversized = [cluster for cluster in result_clusters if len(cluster) > max_size]
    if oversized:
        logger.warning(
            "Oversized clusters after split (%s). Applying hard chunk fallback.",
            [len(c) for c in oversized],
        )
        result_clusters = []
        for i in range(0, m, max_size):
            result_clusters.append(remaining[i:i + max_size])

    # Log the split
    sizes = [len(c) for c in result_clusters]
    if keep_duplicate_singletons:
        logger.info(
            "Split cluster: %s photos -> %s canonical clusters (sizes: %s) + %s duplicate singleton clusters",
            len(cluster_photo_ids),
            len(result_clusters),
            sizes,
            len(duplicate_clusters),
        )
    else:
        logger.info(
            "Split cluster: %s photos -> %s canonical clusters (sizes: %s), dropped %s duplicates",
            len(cluster_photo_ids),
            len(result_clusters),
            sizes,
            len(duplicate_clusters),
        )

    return result_clusters + emitted_duplicate_clusters, duplicate_of


def _build_similarity_lookup(
    similarity_records: List[Dict[str, object]],
) -> Dict[Tuple[int, int], Dict[str, object]]:
    """Build quick lookup for per-pair metrics collected during stage 2."""
    lookup: Dict[Tuple[int, int], Dict[str, object]] = {}
    for record in similarity_records:
        photo_a = record.get("photo_a_id")
        photo_b = record.get("photo_b_id")
        if photo_a is None or photo_b is None:
            continue
        key = (int(min(photo_a, photo_b)), int(max(photo_a, photo_b)))
        lookup[key] = record
    return lookup


def _float_or_none(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _int_or_none(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _segment_scores_from_diagnostics(diagnostics: Dict[str, Any] | None) -> Dict[str, float | None]:
    segment_scores = diagnostics.get("segment_scores") if diagnostics else {}
    raw = segment_scores if isinstance(segment_scores, dict) else {}
    return {
        "from_left_25_50": _float_or_none(raw.get("from_left_25_50")),
        "from_right_50_75": _float_or_none(raw.get("from_right_50_75")),
        "to_left_25_50": _float_or_none(raw.get("to_left_25_50")),
        "to_right_50_75": _float_or_none(raw.get("to_right_50_75")),
        "cross_left_to_right": _float_or_none(raw.get("cross_left_to_right")),
        "cross_right_to_left": _float_or_none(raw.get("cross_right_to_left")),
        "cross_center_to_center": _float_or_none(raw.get("cross_center_to_center")),
    }


def _oracle_metrics_from_diagnostics(diagnostics: Dict[str, Any] | None) -> Dict[str, float | int | None]:
    oracle = diagnostics.get("oracle") if diagnostics else {}
    raw = oracle if isinstance(oracle, dict) else {}
    transition_overlap_ok = raw.get("transition_overlap_ok")
    transition_overlap_ok_int: int | None = None
    if transition_overlap_ok is not None:
        transition_overlap_ok_int = 1 if bool(transition_overlap_ok) else 0
    return {
        "kornia_overlap_ratio": _float_or_none(raw.get("overlap_ratio")),
        "kornia_side_overlap": _float_or_none(raw.get("side_overlap")),
        "kornia_center_overlap": _float_or_none(raw.get("center_overlap")),
        "kornia_inlier_ratio": _float_or_none(raw.get("inlier_ratio")),
        "kornia_transition_overlap_ok": transition_overlap_ok_int,
    }


def _build_pair_metrics_payload(diagnostics: Dict[str, Any] | None) -> Dict[str, float | int | None]:
    payload: Dict[str, float | int | None] = {}
    payload.update(_segment_scores_from_diagnostics(diagnostics))
    payload.update(_oracle_metrics_from_diagnostics(diagnostics))
    return payload


def enforce_transition_quality(
    ordered_clusters: List[List[int]],
    photo_ids: List[int],
    adjacency: np.ndarray,
    similarity: np.ndarray,
    edge_has_geometry: np.ndarray,
    similarity_records: List[Dict[str, object]],
    room_labels: Optional[List[str]] = None,
    min_overlap: float = HARD_TRANSITION_MIN_OVERLAP,
    min_inliers: int = HARD_TRANSITION_MIN_INLIERS,
    min_side_overlap: float = HARD_TRANSITION_MIN_SIDE_OVERLAP,
    min_center_overlap: float = HARD_TRANSITION_MIN_CENTER_OVERLAP,
    min_overlap_ratio: float = HARD_TRANSITION_MIN_OVERLAP_RATIO,
    adjacent_semantic_trust: float = HARD_TRANSITION_ADJACENT_SEMANTIC_TRUST,
    dist2_semantic_trust: float = HARD_TRANSITION_DIST2_SEMANTIC_TRUST,
    max_semantic_gap: int = HARD_TRANSITION_MAX_SEMANTIC_GAP,
    recovery_semantic_trust: float = HARD_TRANSITION_RECOVERY_SEMANTIC_TRUST,
    front_recovery_semantic_trust: float = HARD_TRANSITION_FRONT_RECOVERY_SEMANTIC_TRUST,
    require_geometry: bool = HARD_TRANSITION_REQUIRE_GEOMETRY,
    require_direction: bool = HARD_TRANSITION_REQUIRE_DIRECTION,
) -> List[List[int]]:
    """Split ordered clusters at transition edges that are too weak for video cuts.

    This is a final safety gate: connected-components can still keep semantic-only
    links that are visually related but not transition-safe.
    By default this service enforces geometry-only transitions for reliability.
    """
    if not ordered_clusters:
        return []

    pid_to_idx = {pid: idx for idx, pid in enumerate(photo_ids)}
    pair_lookup = _build_similarity_lookup(similarity_records)

    refined_clusters: List[List[int]] = []
    split_count = 0

    for cluster in ordered_clusters:
        if len(cluster) <= 1:
            refined_clusters.append(cluster)
            continue

        current_chain = [cluster[0]]

        for left_pid, right_pid in zip(cluster, cluster[1:]):
            idx_left = pid_to_idx.get(left_pid)
            idx_right = pid_to_idx.get(right_pid)
            if idx_left is None or idx_right is None:
                current_chain.append(right_pid)
                continue

            overlap_score = float(adjacency[idx_left, idx_right])
            semantic_score = float(similarity[idx_left, idx_right])
            has_geometry = bool(edge_has_geometry[idx_left, idx_right])
            seq_gap = abs(idx_right - idx_left)

            record = pair_lookup.get((min(left_pid, right_pid), max(left_pid, right_pid)), {})
            raw_inliers = record.get("geometric_inliers")
            try:
                inliers = int(raw_inliers) if raw_inliers is not None else 0
            except (TypeError, ValueError):
                inliers = 0
            side_overlap = max(
                _float_or_none(record.get("cross_left_to_right")) or 0.0,
                _float_or_none(record.get("cross_right_to_left")) or 0.0,
            )
            center_overlap = _float_or_none(record.get("cross_center_to_center")) or 0.0
            overlap_ratio = _float_or_none(record.get("kornia_overlap_ratio"))
            transition_overlap_ok = (
                side_overlap >= min_side_overlap
                or center_overlap >= min_center_overlap
            )
            overlap_ratio_ok = (
                overlap_ratio is None
                or overlap_ratio >= min_overlap_ratio
            )
            raw_dx = record.get("direction_dx")
            raw_dy = record.get("direction_dy")
            try:
                direction_dx = float(raw_dx) if raw_dx is not None else None
                direction_dy = float(raw_dy) if raw_dy is not None else None
            except (TypeError, ValueError):
                direction_dx = None
                direction_dy = None
            has_direction = (
                direction_dx is not None
                and direction_dy is not None
                and ((direction_dx * direction_dx) + (direction_dy * direction_dy)) > 1e-8
            )
            pair_source = str(record.get("pair_source") or "")
            room_left = room_labels[idx_left] if room_labels and idx_left < len(room_labels) else None
            room_right = room_labels[idx_right] if room_labels and idx_right < len(room_labels) else None
            both_front_exterior = (
                room_family(room_left) == "front_exterior"
                and room_family(room_right) == "front_exterior"
            )

            keep_edge = False
            keep_reason = ""

            if require_geometry:
                geometry_strong_enough = (
                    has_geometry
                    and inliers >= min_inliers
                    and overlap_score >= min_overlap
                    and transition_overlap_ok
                    and overlap_ratio_ok
                )
                direction_ok = (not require_direction) or has_direction
                if geometry_strong_enough and direction_ok:
                    keep_edge = True
                    keep_reason = "geometry_strict"
            else:
                if has_geometry and (inliers >= min_inliers or overlap_score >= min_overlap):
                    keep_edge = True
                    keep_reason = "geometry"
                elif "semantic_recovery" in pair_source:
                    if (
                        both_front_exterior
                        and semantic_score >= front_recovery_semantic_trust
                    ):
                        keep_edge = True
                        keep_reason = "front_recovery"
                    elif seq_gap <= max_semantic_gap and semantic_score >= recovery_semantic_trust:
                        keep_edge = True
                        keep_reason = "semantic_recovery"
                elif seq_gap <= 1 and semantic_score >= adjacent_semantic_trust:
                    keep_edge = True
                    keep_reason = "adjacent_semantic"
                elif seq_gap <= 2 and semantic_score >= dist2_semantic_trust:
                    keep_edge = True
                    keep_reason = "dist2_semantic"

            if keep_edge:
                current_chain.append(right_pid)
                logger.debug(
                    "Transition kept %s -> %s (%s: overlap=%.3f, inliers=%s, sem=%.3f, "
                    "side=%.3f, center=%.3f, overlap_ratio=%s, gap=%s, source=%s)",
                    left_pid,
                    right_pid,
                    keep_reason,
                    overlap_score,
                    inliers,
                    semantic_score,
                    side_overlap,
                    center_overlap,
                    f"{overlap_ratio:.3f}" if overlap_ratio is not None else "n/a",
                    seq_gap,
                    pair_source,
                )
                continue

            split_count += 1
            logger.info(
                "Transition split %s -> %s (overlap=%.3f, inliers=%s, sem=%.3f, "
                "side=%.3f, center=%.3f, overlap_ratio=%s, overlap_ok=%s, "
                "has_geometry=%s, has_direction=%s, gap=%s, source=%s)",
                left_pid,
                right_pid,
                overlap_score,
                inliers,
                semantic_score,
                side_overlap,
                center_overlap,
                f"{overlap_ratio:.3f}" if overlap_ratio is not None else "n/a",
                "yes" if transition_overlap_ok else "no",
                "yes" if has_geometry else "no",
                "yes" if has_direction else "no",
                seq_gap,
                pair_source,
            )
            refined_clusters.append(current_chain)
            current_chain = [right_pid]

        if current_chain:
            refined_clusters.append(current_chain)

    logger.info(
        "Transition quality enforcement: %s ordered clusters -> %s refined clusters "
        "(splits=%s, min_overlap=%.2f, min_inliers=%s, min_side=%.2f, min_center=%.2f, min_ratio=%.2f, strict_geometry=%s, require_direction=%s, adjacent_sem>=%.2f, dist2_sem>=%.2f, recovery_sem>=%.2f, front_recovery_sem>=%.2f, max_sem_gap=%s)",
        len(ordered_clusters),
        len(refined_clusters),
        split_count,
        min_overlap,
        min_inliers,
        min_side_overlap,
        min_center_overlap,
        min_overlap_ratio,
        "yes" if require_geometry else "no",
        "yes" if require_direction else "no",
        adjacent_semantic_trust,
        dist2_semantic_trust,
        recovery_semantic_trust,
        front_recovery_semantic_trust,
        max_semantic_gap,
    )

    return refined_clusters


# LightGlue/LoFTR model singleton
_matcher_type = None
_loftr_matchers: Dict[str, Any] = {}
_efficient_loftr_processor = None
_efficient_loftr_model = None
_zju_loftr_variant_models: Dict[str, Any] = {}
_zju_loftr_variant_meta: Dict[str, Dict[str, Any]] = {}
_zju_legacy_matcher = None
_zju_legacy_checkpoint_path: str | None = None
_kornia_oracle_ransac = None


def _normalized_oracle_mode() -> str:
    mode = (settings.KORNIA_ORACLE_MODE or "off").strip().lower()
    if mode not in {"off", "shadow", "gate"}:
        logger.warning("Unknown KORNIA_ORACLE_MODE=%s (using off)", settings.KORNIA_ORACLE_MODE)
        return "off"
    return mode


def _load_kornia_oracle_ransac():
    global _kornia_oracle_ransac
    if _kornia_oracle_ransac is not None:
        return _kornia_oracle_ransac

    from kornia.geometry.ransac import RANSAC

    _kornia_oracle_ransac = RANSAC(
        model_type="homography",
        inl_th=float(settings.KORNIA_ORACLE_INLIER_THRESHOLD_PX),
        max_iter=10,
        confidence=0.999,
    )
    return _kornia_oracle_ransac


def _compute_homography_overlap_ratio(
    homography: np.ndarray,
    width: int,
    height: int,
) -> float:
    if homography is None or homography.shape != (3, 3) or not np.isfinite(homography).all():
        return 0.0

    corners = np.array(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float32,
    )
    projected = cv2.perspectiveTransform(corners.reshape(1, -1, 2), homography.astype(np.float32))[0]
    target_rect = corners

    # cv2.intersectConvexConvex returns (intersection_area, intersection_polygon)
    try:
        intersection_area, _ = cv2.intersectConvexConvex(projected, target_rect)
    except cv2.error:
        return 0.0

    if not np.isfinite(intersection_area) or intersection_area <= 0.0:
        return 0.0

    frame_area = float(max(1, width * height))
    return float(np.clip(float(intersection_area) / frame_area, 0.0, 1.0))


def _evaluate_kornia_oracle(
    inlier_points0: np.ndarray,
    inlier_points1: np.ndarray,
    width: int,
    height: int,
    segment_scores: Dict[str, float],
) -> Dict[str, Any]:
    mode = _normalized_oracle_mode()
    if mode == "off":
        return {
            "mode": mode,
            "evaluated": False,
            "passed": True,
            "decision": "disabled",
        }

    if len(inlier_points0) < 8 or len(inlier_points1) < 8:
        passed = mode != "gate"
        return {
            "mode": mode,
            "evaluated": False,
            "passed": passed,
            "decision": "insufficient_inliers",
            "overlap_ratio": 0.0,
            "side_overlap": 0.0,
            "inlier_ratio": 0.0,
        }

    try:
        ransac = _load_kornia_oracle_ransac()
        pts0_t = torch.from_numpy(np.ascontiguousarray(inlier_points0, dtype=np.float32))
        pts1_t = torch.from_numpy(np.ascontiguousarray(inlier_points1, dtype=np.float32))
        homography_t, inlier_mask_t = ransac(pts0_t, pts1_t)
        homography = homography_t.detach().cpu().numpy()
        inlier_mask = inlier_mask_t.detach().cpu().numpy().astype(bool)

        kornia_inliers = int(inlier_mask.sum())
        kornia_inlier_ratio = float(kornia_inliers / max(1, len(inlier_points0)))
        overlap_ratio = _compute_homography_overlap_ratio(homography, width=width, height=height)
        side_overlap = max(
            float(segment_scores.get("cross_left_to_right", 0.0)),
            float(segment_scores.get("cross_right_to_left", 0.0)),
        )
        center_overlap = float(segment_scores.get("cross_center_to_center", 0.0))
        transition_overlap_ok = (
            side_overlap >= float(settings.KORNIA_ORACLE_MIN_SIDE_OVERLAP)
            or center_overlap >= float(settings.KORNIA_ORACLE_MIN_CENTER_OVERLAP)
        )

        passed = (
            overlap_ratio >= float(settings.KORNIA_ORACLE_MIN_OVERLAP_RATIO)
            and transition_overlap_ok
            and kornia_inlier_ratio >= float(settings.KORNIA_ORACLE_MIN_INLIER_RATIO)
        )

        return {
            "mode": mode,
            "evaluated": True,
            "passed": passed,
            "decision": "pass" if passed else "reject",
            "overlap_ratio": overlap_ratio,
            "side_overlap": side_overlap,
            "center_overlap": center_overlap,
            "transition_overlap_ok": transition_overlap_ok,
            "inlier_ratio": kornia_inlier_ratio,
            "kornia_inliers": kornia_inliers,
        }
    except Exception as err:
        logger.debug("Kornia oracle evaluation failed: %s", err)
        return {
            "mode": mode,
            "evaluated": False,
            "passed": True,  # fail-open to keep pipeline stable
            "decision": "error_fail_open",
            "error": str(err),
        }


def _apply_kornia_oracle_to_match(
    num_matches: int,
    num_inliers: int,
    score: float,
    direction: Tuple[float, float],
    inlier_points0: np.ndarray,
    inlier_points1: np.ndarray,
    width: int,
    height: int,
    segment_scores: Dict[str, float],
) -> Tuple[int, int, float, Tuple[float, float], Dict[str, Any]]:
    oracle = _evaluate_kornia_oracle(
        inlier_points0=inlier_points0,
        inlier_points1=inlier_points1,
        width=width,
        height=height,
        segment_scores=segment_scores,
    )
    mode = str(oracle.get("mode", "off"))

    if mode == "gate" and not bool(oracle.get("passed", True)):
        return num_matches, 0, 0.0, (0.0, 0.0), oracle
    return num_matches, num_inliers, score, direction, oracle


def _annotate_pair_source_with_oracle(pair_source: str, diagnostics: Dict[str, Any] | None) -> str:
    if not diagnostics:
        return pair_source
    oracle = diagnostics.get("oracle")
    if not isinstance(oracle, dict):
        return pair_source

    mode = str(oracle.get("mode", "off"))
    decision = str(oracle.get("decision", ""))
    if mode == "shadow":
        suffix = "|koS"
    elif mode == "gate" and decision == "pass":
        suffix = "|koG"
    elif mode == "gate" and decision in {"reject", "insufficient_inliers"}:
        suffix = "|koR"
    else:
        return pair_source

    annotated = f"{pair_source}{suffix}"
    return annotated[:50]


def _load_loftr_checkpoint(checkpoint: str):
    """Load and cache a specific LoFTR checkpoint."""
    if checkpoint in _loftr_matchers:
        return _loftr_matchers[checkpoint]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from kornia.feature import LoFTR

    matcher = LoFTR(pretrained=checkpoint)
    matcher = matcher.to(device)
    matcher.eval()
    _loftr_matchers[checkpoint] = matcher
    logger.info("Loaded LoFTR matcher (%s) on %s", checkpoint, device)
    return matcher


def _zju_variant_checkpoint_env_key(variant: str) -> str:
    variant = str(variant).strip().lower()
    env_keys = {
        "indoor_ds": "LOFTR_ZJU_INDOOR_DS_CKPT",
        "indoor_ot": "LOFTR_ZJU_INDOOR_OT_CKPT",
        "outdoor_ds": "LOFTR_ZJU_OUTDOOR_DS_CKPT",
        "outdoor_ot": "LOFTR_ZJU_OUTDOOR_OT_CKPT",
    }
    if variant not in env_keys:
        raise RuntimeError(f"Unknown ZJU LoFTR variant: {variant}")
    return env_keys[variant]


def _zju_variant_checkpoint_path(variant: str) -> str:
    variant = str(variant).strip().lower()
    env_key = _zju_variant_checkpoint_env_key(variant)
    settings_attr_map = {
        "indoor_ds": "LOFTR_ZJU_INDOOR_DS_CKPT",
        "indoor_ot": "LOFTR_ZJU_INDOOR_OT_CKPT",
        "outdoor_ds": "LOFTR_ZJU_OUTDOOR_DS_CKPT",
        "outdoor_ot": "LOFTR_ZJU_OUTDOOR_OT_CKPT",
    }
    settings_attr = settings_attr_map.get(variant)
    configured = getattr(settings, settings_attr, None) if settings_attr else None
    if not configured:
        configured = os.getenv(env_key)

    if not configured:
        raise RuntimeError(
            f"ZJU model id not found for variant '{variant}': missing env {env_key}"
        )

    ckpt_path = os.path.expanduser(str(configured).strip())
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise RuntimeError(
            f"ZJU model id not found for variant '{variant}': "
            f"{env_key}='{configured}' does not exist"
        )
    return ckpt_path


def _load_zju_loftr_variant(variant: str):
    """Load ZJU LoFTR variant checkpoint with strict repo-based implementation (no fallback)."""
    variant = str(variant).strip().lower()
    if variant not in ZJU_LOFTR_VARIANTS:
        raise RuntimeError(f"Unknown ZJU LoFTR variant: {variant}")
    if variant in _zju_loftr_variant_models:
        return _zju_loftr_variant_models[variant]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_dir = os.path.expanduser((settings.LOFTR_ZJU_REPO_DIR or "").strip())
    ckpt_path = _zju_variant_checkpoint_path(variant)
    if not (repo_dir and os.path.isdir(repo_dir)):
        raise RuntimeError(
            f"ZJU model id not found for variant '{variant}': "
            "LOFTR_ZJU_REPO_DIR is missing or invalid"
        )

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    try:
        from src.loftr import LoFTR as ZJULoFTR, default_cfg as zju_default_cfg
    except Exception as err:
        raise RuntimeError(
            f"ZJU model id not found for variant '{variant}': failed to import repo modules ({err})"
        ) from err

    cfg = copy.deepcopy(zju_default_cfg)
    # zju3dv LoFTR repo has config variants where sparse_spvs is defined in
    # training configs but absent in cvpr_ds_config default. CoarseMatching
    # expects this key at runtime, so set an explicit inference-safe default.
    cfg.setdefault("match_coarse", {})
    cfg["match_coarse"].setdefault("sparse_spvs", False)
    if variant.endswith("_ot"):
        cfg["match_coarse"]["match_type"] = "sinkhorn"
        cfg["match_coarse"]["skh_iters"] = 3
        cfg["match_coarse"]["skh_init_bin_score"] = 1.0
        cfg["match_coarse"]["skh_prefilter"] = True

    matcher = ZJULoFTR(config=cfg)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unexpected checkpoint format for {ckpt_path}")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = str(key)
        for prefix in ("module.", "matcher."):
            if cleaned_key.startswith(prefix):
                cleaned_key = cleaned_key[len(prefix):]
        cleaned_state_dict[cleaned_key] = value

    missing, unexpected = matcher.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        logger.debug("ZJU %s missing keys: %s", variant, missing[:10])
    if unexpected:
        logger.debug("ZJU %s unexpected keys: %s", variant, unexpected[:10])

    matcher = matcher.to(device)
    matcher.eval()
    _zju_loftr_variant_models[variant] = matcher
    _zju_loftr_variant_meta[variant] = {
        "loader": "zju_repo_strict",
        "repo_dir": repo_dir,
        "checkpoint_path": ckpt_path,
        "match_type": "sinkhorn" if variant.endswith("_ot") else "dual_softmax",
        "model_class": f"{matcher.__class__.__module__}.{matcher.__class__.__name__}",
    }
    logger.info("Loaded ZJU LoFTR %s via strict repo loader (%s) on %s", variant, ckpt_path, device)
    return matcher


def _load_zju_loftr_indoor():
    """Backward-compatible indoor_ds loader."""
    return _load_zju_loftr_variant("indoor_ds")


def _load_zju_loftr_legacy():
    """Legacy pre-4-variant behavior: single checkpoint loaded into Kornia LoFTR."""
    global _zju_legacy_matcher, _zju_legacy_checkpoint_path
    if _zju_legacy_matcher is not None:
        return _zju_legacy_matcher

    candidates: List[str] = []
    if settings.LOFTR_ZJU_INDOOR_CKPT:
        candidates.append(os.path.expanduser(str(settings.LOFTR_ZJU_INDOOR_CKPT).strip()))
    candidates.extend(
        [
            os.path.expanduser("~/.cache/torch/hub/checkpoints/loftr_indoor.ckpt"),
            os.path.expanduser("~/.cache/torch/hub/checkpoints/loftr_indoor_ds_new.ckpt"),
        ]
    )
    ckpt_path = next((p for p in candidates if p and os.path.isfile(p)), None)
    if not ckpt_path:
        raise RuntimeError(
            "ZJU legacy model id not found: expected LOFTR_ZJU_INDOOR_CKPT or "
            "~/.cache/torch/hub/checkpoints/loftr_indoor*.ckpt"
        )

    from kornia.feature import LoFTR

    matcher = LoFTR(pretrained=None)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unexpected checkpoint format for {ckpt_path}")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = str(key)
        for prefix in ("module.", "matcher."):
            if cleaned_key.startswith(prefix):
                cleaned_key = cleaned_key[len(prefix):]
        cleaned_state_dict[cleaned_key] = value

    matcher.load_state_dict(cleaned_state_dict, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = matcher.to(device)
    matcher.eval()
    _zju_legacy_matcher = matcher
    _zju_legacy_checkpoint_path = ckpt_path
    logger.info("Loaded ZJU legacy LoFTR from checkpoint (%s) on %s", ckpt_path, device)
    return matcher


def _load_efficient_loftr():
    """Load EfficientLoFTR from Hugging Face (lazy singleton)."""
    global _efficient_loftr_processor, _efficient_loftr_model
    if _efficient_loftr_processor is not None and _efficient_loftr_model is not None:
        return _efficient_loftr_processor, _efficient_loftr_model

    from transformers import AutoImageProcessor, AutoModelForKeypointMatching

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Keep model-card behavior stable: fast processor can shift keypoint outputs.
    processor = AutoImageProcessor.from_pretrained(EFFICIENT_LOFTR_MODEL_ID, use_fast=False)
    model = AutoModelForKeypointMatching.from_pretrained(EFFICIENT_LOFTR_MODEL_ID)
    model = model.to(device)
    model.eval()

    _efficient_loftr_processor = processor
    _efficient_loftr_model = model
    logger.info("Loaded EfficientLoFTR matcher (%s) on %s", EFFICIENT_LOFTR_MODEL_ID, device)
    return _efficient_loftr_processor, _efficient_loftr_model


def _extract_hw_from_processor_inputs(processor_inputs: Dict[str, Any]) -> Tuple[int | None, int | None]:
    for key in ("pixel_values", "pixel_values0", "pixel_values1"):
        value = processor_inputs.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 4:
            return int(value.shape[-1]), int(value.shape[-2])
    return None, None


def _as_xy_points(raw_points: Any) -> np.ndarray:
    if raw_points is None:
        return np.empty((0, 2), dtype=np.float32)
    if isinstance(raw_points, torch.Tensor):
        array = raw_points.detach().cpu().numpy()
    else:
        array = np.asarray(raw_points)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 2)
    elif array.ndim > 2:
        array = array.reshape(-1, array.shape[-1])
    if array.shape[-1] < 2:
        return np.empty((0, 2), dtype=np.float32)
    return np.ascontiguousarray(array[:, :2], dtype=np.float32)


def _as_score_vector(raw_scores: Any) -> np.ndarray:
    if raw_scores is None:
        return np.empty((0,), dtype=np.float32)
    if isinstance(raw_scores, torch.Tensor):
        array = raw_scores.detach().cpu().numpy()
    else:
        array = np.asarray(raw_scores)
    if array.size == 0:
        return np.empty((0,), dtype=np.float32)
    return np.ascontiguousarray(array.reshape(-1), dtype=np.float32)


def _matching_score_summary(scores: np.ndarray) -> Dict[str, float]:
    if scores.size == 0:
        return {
            "count": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": float(scores.size),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "p95": float(np.percentile(scores, 95)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


def _sample_scalar_map_at_points(map_image: np.ndarray, points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty((0,), dtype=np.float32)
    h, w = map_image.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((len(points),), dtype=np.float32)
    xs = np.clip(np.rint(points[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.rint(points[:, 1]).astype(np.int32), 0, h - 1)
    return np.ascontiguousarray(map_image[ys, xs], dtype=np.float32)


def _local_std_map(image_gray: np.ndarray, window: int = LOFTR_FILTER_LOCAL_WINDOW) -> np.ndarray:
    k = max(3, int(window) | 1)
    mean = cv2.boxFilter(image_gray, ddepth=-1, ksize=(k, k), normalize=True)
    mean_sq = cv2.boxFilter(image_gray * image_gray, ddepth=-1, ksize=(k, k), normalize=True)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var)
    vmax = float(std.max()) if std.size > 0 else 0.0
    if vmax > 1e-6:
        std = std / vmax
    return np.ascontiguousarray(std, dtype=np.float32)


def _filter_metric(
    scores: np.ndarray,
    keep_mask: np.ndarray,
    base_mask: np.ndarray | None = None,
) -> Dict[str, float]:
    total = int(scores.size)
    if total == 0:
        return {
            "kept_count": 0.0,
            "removed_count": 0.0,
            "excluded_count": 0.0,
            "active_total": 0.0,
            "kept_ratio": 0.0,
            "kept_score_mean": 0.0,
            "removed_score_mean": 0.0,
            "kept_count_term": 0.0,
            "final_score": 0.0,
        }

    keep = np.asarray(keep_mask, dtype=bool)
    if keep.shape[0] != total:
        keep = np.ones((total,), dtype=bool)

    if base_mask is None:
        active = np.ones((total,), dtype=bool)
    else:
        active = np.asarray(base_mask, dtype=bool)
        if active.shape[0] != total:
            active = np.ones((total,), dtype=bool)

    keep_active = keep & active
    removed_active = (~keep) & active
    active_total = int(active.sum())
    excluded_count = int(total - active_total)
    kept_count = int(keep_active.sum())
    removed_count = int(removed_active.sum())

    if active_total <= 0:
        return {
            "kept_count": 0.0,
            "removed_count": 0.0,
            "excluded_count": float(excluded_count),
            "active_total": 0.0,
            "kept_ratio": 0.0,
            "kept_score_mean": 0.0,
            "removed_score_mean": 0.0,
            "kept_count_term": 0.0,
            "final_score": 0.0,
        }

    kept_ratio = float(kept_count / max(1, active_total))
    kept_score_mean = float(np.mean(scores[keep_active])) if kept_count > 0 else 0.0
    removed_score_mean = float(np.mean(scores[removed_active])) if removed_count > 0 else 0.0

    # Scale count targets by active population so neutral exclusions (e.g. low texture)
    # do not penalize the final score.
    active_scale = float(active_total / max(1, total))
    count_zero = float(LOFTR_FILTER_KEEP_COUNT_ZERO) * active_scale
    count_target = float(max(LOFTR_FILTER_KEEP_COUNT_TARGET * active_scale, count_zero + 1.0))
    kept_count_term = float(np.clip((float(kept_count) - count_zero) / (count_target - count_zero), 0.0, 1.0))
    final_score = float(
        kept_score_mean
        * ((1.0 - LOFTR_FILTER_FINAL_KEEP_WEIGHT) + LOFTR_FILTER_FINAL_KEEP_WEIGHT * kept_ratio)
        * (0.30 + 0.70 * kept_count_term)
    )
    return {
        "kept_count": float(kept_count),
        "removed_count": float(removed_count),
        "excluded_count": float(excluded_count),
        "active_total": float(active_total),
        "kept_ratio": float(kept_ratio),
        "kept_score_mean": float(kept_score_mean),
        "removed_score_mean": float(removed_score_mean),
        "kept_count_term": float(kept_count_term),
        "final_score": float(final_score),
    }


def _analyze_loftr_match_filters(
    points0: np.ndarray,
    points1: np.ndarray,
    scores: np.ndarray,
    image0_gray: np.ndarray,
    image1_gray: np.ndarray,
    image0_rgb: np.ndarray | None = None,
    image1_rgb: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Compute per-filter mask rejection and score summaries for LoFTR debug."""
    n = int(min(len(points0), len(points1), len(scores)))
    if n <= 0:
        return {
            "config": {
                "local_window": float(LOFTR_FILTER_LOCAL_WINDOW),
                "min_grad_norm": float(LOFTR_FILTER_MIN_GRAD),
                "min_local_std_norm": float(LOFTR_FILTER_MIN_STD),
                "final_keep_weight": float(LOFTR_FILTER_FINAL_KEEP_WEIGHT),
            },
            "scores": {"combined": _filter_metric(np.empty((0,), dtype=np.float32), np.empty((0,), dtype=bool))},
            "filtered_keep_mask": np.empty((0,), dtype=bool),
            "reject_masks": {},
        }

    points0 = np.ascontiguousarray(points0[:n], dtype=np.float32)
    points1 = np.ascontiguousarray(points1[:n], dtype=np.float32)
    scores = np.ascontiguousarray(scores[:n], dtype=np.float32)
    h0, w0 = image0_gray.shape[:2]
    h1, w1 = image1_gray.shape[:2]

    grad0 = _compute_gradient_magnitude(image0_gray)
    grad1 = _compute_gradient_magnitude(image1_gray)
    g0_max = float(grad0.max()) if grad0.size > 0 else 0.0
    g1_max = float(grad1.max()) if grad1.size > 0 else 0.0
    if g0_max > 1e-6:
        grad0 = grad0 / g0_max
    if g1_max > 1e-6:
        grad1 = grad1 / g1_max
    std0 = _local_std_map(image0_gray)
    std1 = _local_std_map(image1_gray)

    p_grad0 = _sample_scalar_map_at_points(grad0, points0)
    p_grad1 = _sample_scalar_map_at_points(grad1, points1)
    p_std0 = _sample_scalar_map_at_points(std0, points0)
    p_std1 = _sample_scalar_map_at_points(std1, points1)

    y0n = np.clip(points0[:, 1] / max(1.0, float(h0)), 0.0, 1.0)
    y1n = np.clip(points1[:, 1] / max(1.0, float(h1)), 0.0, 1.0)

    sat0 = np.zeros((n,), dtype=np.float32)
    sat1 = np.zeros((n,), dtype=np.float32)
    val0 = np.zeros((n,), dtype=np.float32)
    val1 = np.zeros((n,), dtype=np.float32)
    blue_dom0 = np.zeros((n,), dtype=np.float32)
    blue_dom1 = np.zeros((n,), dtype=np.float32)
    has_rgb = image0_rgb is not None and image1_rgb is not None
    if has_rgb:
        hsv0 = cv2.cvtColor(np.asarray(image0_rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv1 = cv2.cvtColor(np.asarray(image1_rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        sat0_map = hsv0[:, :, 1] / 255.0
        sat1_map = hsv1[:, :, 1] / 255.0
        val0_map = hsv0[:, :, 2] / 255.0
        val1_map = hsv1[:, :, 2] / 255.0
        rgb0 = np.asarray(image0_rgb, dtype=np.float32) / 255.0
        rgb1 = np.asarray(image1_rgb, dtype=np.float32) / 255.0
        blue_dom0_map = rgb0[:, :, 2] - np.maximum(rgb0[:, :, 0], rgb0[:, :, 1])
        blue_dom1_map = rgb1[:, :, 2] - np.maximum(rgb1[:, :, 0], rgb1[:, :, 1])

        sat0 = _sample_scalar_map_at_points(sat0_map, points0)
        sat1 = _sample_scalar_map_at_points(sat1_map, points1)
        val0 = _sample_scalar_map_at_points(val0_map, points0)
        val1 = _sample_scalar_map_at_points(val1_map, points1)
        blue_dom0 = _sample_scalar_map_at_points(blue_dom0_map, points0)
        blue_dom1 = _sample_scalar_map_at_points(blue_dom1_map, points1)

    low_texture_pair = (p_grad0 < LOFTR_FILTER_MIN_GRAD) & (p_grad1 < LOFTR_FILTER_MIN_GRAD)
    low_variance_pair = (p_std0 < LOFTR_FILTER_MIN_STD) & (p_std1 < LOFTR_FILTER_MIN_STD)

    sky_pair = np.zeros((n,), dtype=bool)
    ceiling_pair = np.zeros((n,), dtype=bool)
    floor_pair = np.zeros((n,), dtype=bool)
    wall_pair = np.zeros((n,), dtype=bool)
    if has_rgb:
        sky0 = (
            (y0n <= LOFTR_FILTER_SKY_TOP_MAX_Y)
            & (val0 > 0.35)
            & ((blue_dom0 > 0.08) | ((sat0 < 0.12) & (val0 > 0.78)))
        )
        sky1 = (
            (y1n <= LOFTR_FILTER_SKY_TOP_MAX_Y)
            & (val1 > 0.35)
            & ((blue_dom1 > 0.08) | ((sat1 < 0.12) & (val1 > 0.78)))
        )
        sky_pair = sky0 & sky1

        ceiling0 = (
            (y0n <= LOFTR_FILTER_CEILING_TOP_MAX_Y)
            & (sat0 < LOFTR_FILTER_LOW_SAT)
            & (val0 > LOFTR_FILTER_BRIGHT)
            & (p_grad0 < 0.10)
            & (p_std0 < 0.10)
        )
        ceiling1 = (
            (y1n <= LOFTR_FILTER_CEILING_TOP_MAX_Y)
            & (sat1 < LOFTR_FILTER_LOW_SAT)
            & (val1 > LOFTR_FILTER_BRIGHT)
            & (p_grad1 < 0.10)
            & (p_std1 < 0.10)
        )
        ceiling_pair = ceiling0 & ceiling1

        floor0 = (
            (y0n >= LOFTR_FILTER_FLOOR_BOTTOM_MIN_Y)
            & (sat0 < LOFTR_FILTER_LOW_SAT)
            & (p_grad0 < 0.11)
            & (p_std0 < 0.11)
        )
        floor1 = (
            (y1n >= LOFTR_FILTER_FLOOR_BOTTOM_MIN_Y)
            & (sat1 < LOFTR_FILTER_LOW_SAT)
            & (p_grad1 < 0.11)
            & (p_std1 < 0.11)
        )
        floor_pair = floor0 & floor1

        wall0 = (
            (y0n >= LOFTR_FILTER_WALL_MIN_Y)
            & (y0n <= LOFTR_FILTER_WALL_MAX_Y)
            & (sat0 < LOFTR_FILTER_LOW_SAT)
            & (p_grad0 < 0.09)
            & (p_std0 < 0.09)
        )
        wall1 = (
            (y1n >= LOFTR_FILTER_WALL_MIN_Y)
            & (y1n <= LOFTR_FILTER_WALL_MAX_Y)
            & (sat1 < LOFTR_FILTER_LOW_SAT)
            & (p_grad1 < 0.09)
            & (p_std1 < 0.09)
        )
        wall_pair = wall0 & wall1

    weak_detail_pair = low_texture_pair & low_variance_pair

    reject_masks: Dict[str, np.ndarray] = {
        "low_texture_pair": low_texture_pair,
        "low_variance_pair": low_variance_pair,
        "weak_detail_pair": weak_detail_pair,
        "sky_pair": sky_pair,
        "ceiling_pair": ceiling_pair,
        "floor_pair": floor_pair,
        "wall_pair": wall_pair,
    }
    # Keep low_texture/low_variance as diagnostics only.
    # Treat only clearly risky regions as hard rejections in "combined".
    hard_reject_keys = {
        "weak_detail_pair",
        "sky_pair",
        "ceiling_pair",
        "floor_pair",
        "wall_pair",
    }
    combined_reject = np.zeros((n,), dtype=bool)
    combined_reject_strict = np.zeros((n,), dtype=bool)
    score_table: Dict[str, Dict[str, float]] = {}
    for key, reject_mask in reject_masks.items():
        single_keep = ~reject_mask
        score_table[key] = _filter_metric(scores, single_keep)
        if key in hard_reject_keys:
            combined_reject |= reject_mask
        combined_reject_strict |= reject_mask
    combined_base = ~low_texture_pair
    combined_keep = (~combined_reject) & combined_base
    # Combined metric is evaluated only on non-low-texture population.
    score_table["combined"] = _filter_metric(scores, combined_keep, base_mask=combined_base)
    score_table["combined_strict"] = _filter_metric(scores, ~combined_reject_strict)

    return {
        "config": {
            "local_window": float(LOFTR_FILTER_LOCAL_WINDOW),
            "min_grad_norm": float(LOFTR_FILTER_MIN_GRAD),
            "min_local_std_norm": float(LOFTR_FILTER_MIN_STD),
            "sky_top_max_y": float(LOFTR_FILTER_SKY_TOP_MAX_Y),
            "ceiling_top_max_y": float(LOFTR_FILTER_CEILING_TOP_MAX_Y),
            "floor_bottom_min_y": float(LOFTR_FILTER_FLOOR_BOTTOM_MIN_Y),
            "wall_min_y": float(LOFTR_FILTER_WALL_MIN_Y),
            "wall_max_y": float(LOFTR_FILTER_WALL_MAX_Y),
            "low_saturation_max": float(LOFTR_FILTER_LOW_SAT),
            "bright_min_value": float(LOFTR_FILTER_BRIGHT),
            "final_keep_weight": float(LOFTR_FILTER_FINAL_KEEP_WEIGHT),
            "uses_color_masks": bool(has_rgb),
            "combined_hard_reject_keys": sorted(hard_reject_keys),
        },
        "scores": score_table,
        "filtered_keep_mask": combined_keep,
        "reject_masks": reject_masks,
    }


def _build_filter_match_sets(
    points0: np.ndarray,
    points1: np.ndarray,
    width0: int,
    height0: int,
    width1: int,
    height1: int,
    filter_analysis: Dict[str, Any],
) -> Dict[str, List[Dict[str, float]]]:
    """Build normalized point overlays for each filter mode."""
    result: Dict[str, List[Dict[str, float]]] = {}
    result["loftr"] = _sample_normalized_matches(
        points0=points0,
        points1=points1,
        width0=width0,
        height0=height0,
        width1=width1,
        height1=height1,
        max_points=5000,
    )

    n = int(len(points0))
    keep_mask = np.asarray(filter_analysis.get("filtered_keep_mask", np.empty((0,), dtype=bool)), dtype=bool)
    if keep_mask.shape[0] != n:
        keep_mask = np.ones((n,), dtype=bool)
    result["combined"] = _sample_normalized_matches(
        points0=points0[keep_mask] if n > 0 else np.empty((0, 2), dtype=np.float32),
        points1=points1[keep_mask] if n > 0 else np.empty((0, 2), dtype=np.float32),
        width0=width0,
        height0=height0,
        width1=width1,
        height1=height1,
        max_points=5000,
    )

    reject_masks = filter_analysis.get("reject_masks")
    if isinstance(reject_masks, dict):
        for name, mask in reject_masks.items():
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape[0] != n:
                continue
            result[str(name)] = _sample_normalized_matches(
                points0=points0[mask_arr] if n > 0 else np.empty((0, 2), dtype=np.float32),
                points1=points1[mask_arr] if n > 0 else np.empty((0, 2), dtype=np.float32),
                width0=width0,
                height0=height0,
                width1=width1,
                height1=height1,
                max_points=5000,
            )
    return result


def _encode_debug_image_data_url(image: Image.Image, max_width: int = 1800) -> str | None:
    if image is None:
        return None
    try:
        preview = image.copy()
        if preview.width > max_width and preview.width > 0:
            scale = float(max_width / float(preview.width))
            new_h = max(1, int(preview.height * scale))
            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
            preview = preview.resize((max_width, new_h), resampling)
        buffer = io.BytesIO()
        preview.save(buffer, format="JPEG", quality=88, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None


def _build_efficient_hf_visualization(
    processor,
    img1: Image.Image,
    img2: Image.Image,
    post_processed_result: Dict[str, Any],
) -> str | None:
    """Generate HF-native keypoint matching visualization as a data URL."""
    if not isinstance(post_processed_result, dict):
        return None
    try:
        vis = processor.visualize_keypoint_matching(
            images=[img1, img2],
            keypoint_matching_output=[post_processed_result],
        )
    except TypeError:
        try:
            vis = processor.visualize_keypoint_matching(
                [img1, img2],
                [post_processed_result],
            )
        except Exception:
            return None
    except Exception:
        return None

    vis_image = None
    if isinstance(vis, Image.Image):
        vis_image = vis
    elif isinstance(vis, list) and vis:
        first = vis[0]
        if isinstance(first, Image.Image):
            vis_image = first

    if vis_image is None:
        return None
    return _encode_debug_image_data_url(vis_image)


def _post_process_efficient_matches(
    processor,
    outputs,
    image_sizes: List[Tuple[int, int]],
    threshold: float,
) -> Dict[str, Any]:
    # HF docs use target_sizes=[[(h0, w0), (h1, w1)]] for a single image pair.
    target_sizes = [image_sizes]
    try:
        processed = processor.post_process_keypoint_matching(
            outputs,
            target_sizes=target_sizes,
            threshold=float(threshold),
        )
    except TypeError:
        # Older/newer API variants may not accept keyword args consistently.
        processed = processor.post_process_keypoint_matching(
            outputs,
            target_sizes,
            threshold=float(threshold),
        )

    if isinstance(processed, list):
        if not processed:
            return {}
        first = processed[0]
        if isinstance(first, list):
            return first[0] if first else {}
        if isinstance(first, dict):
            return first
    if isinstance(processed, dict):
        return processed
    return {}


def _extract_loftr_points_and_scores(correspondences: Dict[str, Any] | None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(correspondences, dict):
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    points0 = _as_xy_points(correspondences.get("keypoints0"))
    if len(points0) == 0:
        points0 = _as_xy_points(correspondences.get("mkpts0_f"))

    points1 = _as_xy_points(correspondences.get("keypoints1"))
    if len(points1) == 0:
        points1 = _as_xy_points(correspondences.get("mkpts1_f"))

    scores = _as_score_vector(correspondences.get("confidence"))
    if scores.size == 0:
        scores = _as_score_vector(correspondences.get("mconf"))
    if scores.size == 0 and len(points0) > 0:
        scores = np.ones((len(points0),), dtype=np.float32)

    n = int(min(len(points0), len(points1), len(scores)))
    if len(points0) != n:
        points0 = points0[:n]
    if len(points1) != n:
        points1 = points1[:n]
    if len(scores) != n:
        scores = scores[:n]
    return points0, points1, scores


def _build_native_loftr_diagnostics(
    matcher_name: str,
    checkpoint: str,
    confidence_threshold: float,
    points0: np.ndarray,
    points1: np.ndarray,
    scores: np.ndarray,
    width0: int,
    height0: int,
    width1: int,
    height1: int,
    input_w: int | None,
    input_h: int | None,
    image0_gray: np.ndarray,
    image1_gray: np.ndarray,
    image0_rgb: np.ndarray | None = None,
    image1_rgb: np.ndarray | None = None,
) -> Tuple[int, int, float, Tuple[float, float], Dict[str, Any]]:
    raw_count = int(len(points0))
    conf_mask = scores >= float(confidence_threshold) if scores.size > 0 else np.zeros((0,), dtype=bool)
    conf_points0 = points0[conf_mask] if raw_count > 0 else np.empty((0, 2), dtype=np.float32)
    conf_points1 = points1[conf_mask] if raw_count > 0 else np.empty((0, 2), dtype=np.float32)
    conf_scores = scores[conf_mask] if raw_count > 0 else np.empty((0,), dtype=np.float32)
    kept_count = int(len(conf_points0))
    native_score_summary = _matching_score_summary(conf_scores)
    native_raw_summary = _matching_score_summary(scores)
    filter_analysis = _analyze_loftr_match_filters(
        points0=conf_points0,
        points1=conf_points1,
        scores=conf_scores,
        image0_gray=image0_gray,
        image1_gray=image1_gray,
        image0_rgb=image0_rgb,
        image1_rgb=image1_rgb,
    )
    filter_keep_mask = np.asarray(filter_analysis.get("filtered_keep_mask", np.empty((0,), dtype=bool)), dtype=bool)
    if filter_keep_mask.shape[0] != kept_count:
        filter_keep_mask = np.ones((kept_count,), dtype=bool)
    filter_combined = filter_analysis.get("scores", {}).get("combined", {})
    filtered_final_score = float(filter_combined.get("final_score", 0.0) or 0.0)
    filter_match_sets = _build_filter_match_sets(
        points0=conf_points0,
        points1=conf_points1,
        width0=width0,
        height0=height0,
        width1=width1,
        height1=height1,
        filter_analysis=filter_analysis,
    )
    filtered_points0 = conf_points0[filter_keep_mask] if kept_count > 0 else conf_points0
    filtered_points1 = conf_points1[filter_keep_mask] if kept_count > 0 else conf_points1

    inlier_points0 = np.empty((0, 2), dtype=np.float32)
    inlier_points1 = np.empty((0, 2), dtype=np.float32)
    geometry_model = "none"
    num_inliers = 0
    direction = (0.0, 0.0)
    segment_scores = _compute_segment_scores(
        inlier_points0,
        inlier_points1,
        width0=width0,
        height0=height0,
        width1=width1,
        height1=height1,
    )
    score_components = _compute_transition_geometry_components(
        num_matches=kept_count,
        num_inliers=0,
        inlier_points0=inlier_points0,
        inlier_points1=inlier_points1,
        width0=width0,
        height0=height0,
        width1=width1,
        height1=height1,
        segment_scores=segment_scores,
        inlier_ratio_zero_point=0.03,
        inlier_ratio_full_point=0.35,
    )
    geometric_score = 0.0

    if kept_count >= 8:
        inlier_mask, geometry_model = _estimate_geometric_inliers(
            conf_points0,
            conf_points1,
            reproj_threshold=RANSAC_REPROJ_THRESHOLD,
        )
        if inlier_mask is not None:
            num_inliers = int(inlier_mask.sum())
            if num_inliers > 0:
                inlier_points0 = conf_points0[inlier_mask]
                inlier_points1 = conf_points1[inlier_mask]
                direction = compute_direction_vector(conf_points0, conf_points1, inlier_mask)
                segment_scores = _compute_segment_scores(
                    inlier_points0,
                    inlier_points1,
                    width0=width0,
                    height0=height0,
                    width1=width1,
                    height1=height1,
                )
                score_components = _compute_transition_geometry_components(
                    num_matches=kept_count,
                    num_inliers=num_inliers,
                    inlier_points0=inlier_points0,
                    inlier_points1=inlier_points1,
                    width0=width0,
                    height0=height0,
                    width1=width1,
                    height1=height1,
                    segment_scores=segment_scores,
                    inlier_ratio_zero_point=0.03,
                    inlier_ratio_full_point=0.35,
                )
                geometric_score = float(score_components.get("final_score", 0.0) or 0.0)

    count_zero = float(NATIVE_SCORE_COUNT_ZERO)
    count_target = float(max(NATIVE_SCORE_COUNT_TARGET, NATIVE_SCORE_COUNT_ZERO + 1))
    match_count_term = float(np.clip((float(kept_count) - count_zero) / (count_target - count_zero), 0.0, 1.0))
    combined_score = float(geometric_score * (0.30 + 0.70 * match_count_term))
    score_components = dict(score_components or {})
    score_components["match_count_term"] = float(match_count_term)
    score_components["combined_score"] = float(combined_score)

    diagnostics = {
        "matcher": matcher_name,
        "checkpoint": checkpoint,
        "confidence_threshold": float(confidence_threshold),
        "geometry_model": geometry_model,
        "segment_scores": segment_scores,
        "score_components": score_components,
        "match_width": int(width0),
        "match_height": int(height0),
        "raw_correspondence_count": int(raw_count),
        "threshold_trials": [
            {
                "threshold": float(confidence_threshold),
                "raw_matches": int(raw_count),
                "num_matches": int(kept_count),
                "num_inliers": int(num_inliers),
                "score": float(combined_score),
                "geometric_score": float(geometric_score),
                "match_count_term": float(match_count_term),
                "geometry_model": geometry_model,
                "native_score_mean": float(native_score_summary["mean"]),
                "native_score_median": float(native_score_summary["median"]),
                "native_score_p95": float(native_score_summary["p95"]),
                "filtered_final_score": float(filtered_final_score),
            }
        ],
        "loftr_input_width": int(input_w if input_w is not None else width0),
        "loftr_input_height": int(input_h if input_h is not None else height0),
        "ransac_reproj_threshold": float(RANSAC_REPROJ_THRESHOLD),
        "raw_matches": _sample_normalized_matches(
            points0=points0,
            points1=points1,
            width0=width0,
            height0=height0,
            width1=width1,
            height1=height1,
            max_points=5000,
        ),
        "filtered_match_count": int(len(filtered_points0)),
        "filtered_matches": _sample_normalized_matches(
            points0=filtered_points0,
            points1=filtered_points1,
            width0=width0,
            height0=height0,
            width1=width1,
            height1=height1,
            max_points=5000,
        ),
        "filtered_final_score": float(filtered_final_score),
        "filter_config": filter_analysis.get("config", {}),
        "filter_scores": filter_analysis.get("scores", {}),
        "filter_match_sets": filter_match_sets,
        "inlier_match_count": int(num_inliers),
        "inlier_matches": _sample_normalized_matches(
            points0=inlier_points0,
            points1=inlier_points1,
            width0=width0,
            height0=height0,
            width1=width1,
            height1=height1,
            max_points=5000,
        ),
        "oracle": {
            "mode": "off",
            "evaluated": False,
            "passed": True,
            "decision": "native_no_oracle",
        },
        "native_matching_scores": native_score_summary,
        "native_matching_scores_raw": native_raw_summary,
        "geometric_score": float(geometric_score),
        "combined_score": float(combined_score),
        "match_count_term": float(match_count_term),
    }
    return kept_count, int(num_inliers), float(combined_score), direction, diagnostics


def _swap_normalized_match_points(points: Any) -> List[Dict[str, float]]:
    swapped: List[Dict[str, float]] = []
    if not isinstance(points, list):
        return swapped
    for p in points:
        if not isinstance(p, dict):
            continue
        x0 = float(p.get("x0", 0.0))
        y0 = float(p.get("y0", 0.0))
        x1 = float(p.get("x1", 0.0))
        y1 = float(p.get("y1", 0.0))
        swapped.append(
            {
                "x0": x1,
                "y0": y1,
                "x1": x0,
                "y1": y0,
                "dx": (x0 - x1),
                "dy": (y0 - y1),
            }
        )
    return swapped


def _swap_segment_scores(segment_scores: Any) -> Dict[str, float]:
    if not isinstance(segment_scores, dict):
        return {}
    return {
        "from_left_25_50": float(segment_scores.get("to_left_25_50", 0.0) or 0.0),
        "from_right_50_75": float(segment_scores.get("to_right_50_75", 0.0) or 0.0),
        "to_left_25_50": float(segment_scores.get("from_left_25_50", 0.0) or 0.0),
        "to_right_50_75": float(segment_scores.get("from_right_50_75", 0.0) or 0.0),
        "cross_left_to_right": float(segment_scores.get("cross_right_to_left", 0.0) or 0.0),
        "cross_right_to_left": float(segment_scores.get("cross_left_to_right", 0.0) or 0.0),
        "cross_center_to_center": float(segment_scores.get("cross_center_to_center", 0.0) or 0.0),
    }


def _reorient_reverse_native_diagnostics(diagnostics: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(diagnostics, dict):
        return {}
    out = dict(diagnostics)
    for key in ("raw_matches", "filtered_matches", "inlier_matches"):
        out[key] = _swap_normalized_match_points(out.get(key))

    filter_sets = out.get("filter_match_sets")
    if isinstance(filter_sets, dict):
        swapped_sets: Dict[str, List[Dict[str, float]]] = {}
        for name, points in filter_sets.items():
            swapped_sets[str(name)] = _swap_normalized_match_points(points)
        out["filter_match_sets"] = swapped_sets

    out["segment_scores"] = _swap_segment_scores(out.get("segment_scores"))
    out["reverse_orientation_selected"] = True
    return out


def _should_retry_reverse_native(num_matches: int, num_inliers: int) -> bool:
    if not NATIVE_REVERSE_RETRY_ENABLED:
        return False
    return (
        int(num_matches) < int(NATIVE_REVERSE_RETRY_MATCH_THRESHOLD)
        or int(num_inliers) < int(NATIVE_REVERSE_RETRY_INLIER_THRESHOLD)
    )


def _is_better_native_result(
    candidate: Tuple[int, int, float],
    current: Tuple[int, int, float],
) -> bool:
    c_matches, c_inliers, c_score = candidate
    p_matches, p_inliers, p_score = current
    # Prioritize geometric reliability first, then match count, then score.
    return (int(c_inliers), int(c_matches), float(c_score)) > (
        int(p_inliers),
        int(p_matches),
        float(p_score),
    )


def _maybe_retry_reverse_native(
    img1: Image.Image,
    img2: Image.Image,
    forward_result: Tuple[int, int, float, Tuple[float, float], Dict[str, Any]],
    run_fn,
    run_kwargs: Dict[str, Any],
) -> Tuple[int, int, float, Tuple[float, float], Dict[str, Any]]:
    num_matches, num_inliers, score, direction, diagnostics = forward_result
    if not _should_retry_reverse_native(num_matches, num_inliers):
        return forward_result

    rev_matches, rev_inliers, rev_score, rev_direction, rev_diag = run_fn(
        img1=img2,
        img2=img1,
        **run_kwargs,
    )
    if not _is_better_native_result(
        (int(rev_matches), int(rev_inliers), float(rev_score)),
        (int(num_matches), int(num_inliers), float(score)),
    ):
        if isinstance(diagnostics, dict):
            diagnostics = dict(diagnostics)
            diagnostics["reverse_retry_attempted"] = True
            diagnostics["reverse_retry_selected"] = False
        return num_matches, num_inliers, score, direction, diagnostics

    adjusted_direction = (-float(rev_direction[0]), -float(rev_direction[1]))
    adjusted_diag = _reorient_reverse_native_diagnostics(rev_diag)
    adjusted_diag["reverse_retry_attempted"] = True
    adjusted_diag["reverse_retry_selected"] = True
    adjusted_diag["reverse_forward_matches"] = int(num_matches)
    adjusted_diag["reverse_forward_inliers"] = int(num_inliers)
    adjusted_diag["reverse_forward_score"] = float(score)
    adjusted_diag["reverse_selected_matches"] = int(rev_matches)
    adjusted_diag["reverse_selected_inliers"] = int(rev_inliers)
    adjusted_diag["reverse_selected_score"] = float(rev_score)
    return int(rev_matches), int(rev_inliers), float(rev_score), adjusted_direction, adjusted_diag


def _match_loftr_kornia_indoor_native(
    img1: Image.Image,
    img2: Image.Image,
    confidence_threshold: float = LOFTR_NATIVE_CONFIDENCE_THRESHOLD,
) -> Tuple[int, int, float, Tuple[float, float], Dict[str, Any]]:
    """Native Kornia LoFTR indoor debug path (no custom geometric scoring)."""
    matcher = _load_loftr_checkpoint("indoor")
    device = next(matcher.parameters()).device

    img1_gray = np.array(img1.convert("L"), dtype=np.float32) / 255.0
    img2_gray = np.array(img2.convert("L"), dtype=np.float32) / 255.0
    img1_rgb = np.array(img1.convert("RGB"), dtype=np.uint8)
    img2_rgb = np.array(img2.convert("RGB"), dtype=np.uint8)

    target_long_side = max(64, int(max(DEFAULT_LOFTR_INPUT_SIZE)))
    img1_resized, meta0 = _resize_by_longest_side_and_pad(img1_gray, target_long_side=target_long_side, multiple=8)
    img2_resized, meta1 = _resize_by_longest_side_and_pad(img2_gray, target_long_side=target_long_side, multiple=8)
    img1_rgb_resized, _ = _resize_rgb_by_longest_side_and_pad(img1_rgb, target_long_side=target_long_side, multiple=8)
    img2_rgb_resized, _ = _resize_rgb_by_longest_side_and_pad(img2_rgb, target_long_side=target_long_side, multiple=8)

    tensor1 = torch.from_numpy(img1_resized).unsqueeze(0).unsqueeze(0).to(device)
    tensor2 = torch.from_numpy(img2_resized).unsqueeze(0).unsqueeze(0).to(device)
    batch = {"image0": tensor1, "image1": tensor2}
    with torch.no_grad():
        raw_output = matcher(batch)
        # ZJU LoFTR mutates batch in-place and returns None.
        correspondences = raw_output if isinstance(raw_output, dict) else batch

    points0, points1, scores = _extract_loftr_points_and_scores(correspondences)
    return _build_native_loftr_diagnostics(
        matcher_name="loftr_kornia_indoor_native",
        checkpoint="kornia:indoor",
        confidence_threshold=float(confidence_threshold),
        points0=points0,
        points1=points1,
        scores=scores,
        width0=int(meta0["content_w"]),
        height0=int(meta0["content_h"]),
        width1=int(meta1["content_w"]),
        height1=int(meta1["content_h"]),
        input_w=int(meta0["content_w"] + meta0["pad_w"]),
        input_h=int(meta0["content_h"] + meta0["pad_h"]),
        image0_gray=img1_resized,
        image1_gray=img2_resized,
        image0_rgb=img1_rgb_resized,
        image1_rgb=img2_rgb_resized,
    )


def _match_loftr_zju_variant_native(
    img1: Image.Image,
    img2: Image.Image,
    variant: str,
    confidence_threshold: float = ZJU_LOFTR_NATIVE_CONFIDENCE_THRESHOLD,
) -> Tuple[int, int, float, Tuple[float, float], Dict[str, Any]]:
    """Native ZJU LoFTR variant debug path (no custom geometric scoring)."""
    variant = str(variant).strip().lower()
    matcher = _load_zju_loftr_variant(variant)
    device = next(matcher.parameters()).device

    img1_gray = np.array(img1.convert("L"), dtype=np.float32) / 255.0
    img2_gray = np.array(img2.convert("L"), dtype=np.float32) / 255.0
    img1_rgb = np.array(img1.convert("RGB"), dtype=np.uint8)
    img2_rgb = np.array(img2.convert("RGB"), dtype=np.uint8)

    target_long_side = max(64, int(max(DEFAULT_LOFTR_INPUT_SIZE)))
    img1_resized, meta0 = _resize_by_longest_side_and_pad(img1_gray, target_long_side=target_long_side, multiple=8)
    img2_resized, meta1 = _resize_by_longest_side_and_pad(img2_gray, target_long_side=target_long_side, multiple=8)
    img1_rgb_resized, _ = _resize_rgb_by_longest_side_and_pad(img1_rgb, target_long_side=target_long_side, multiple=8)
    img2_rgb_resized, _ = _resize_rgb_by_longest_side_and_pad(img2_rgb, target_long_side=target_long_side, multiple=8)

    tensor1 = torch.from_numpy(img1_resized).unsqueeze(0).unsqueeze(0).to(device)
    tensor2 = torch.from_numpy(img2_resized).unsqueeze(0).unsqueeze(0).to(device)
    batch = {"image0": tensor1, "image1": tensor2}
    with torch.no_grad():
        raw_output = matcher(batch)
        # ZJU LoFTR mutates batch in-place and may return None.
        correspondences = raw_output if isinstance(raw_output, dict) else batch

    points0, points1, scores = _extract_loftr_points_and_scores(correspondences)
    num_matches, num_inliers, score, direction, diagnostics = _build_native_loftr_diagnostics(
        matcher_name=f"loftr_zju_{variant}_native",
        checkpoint=f"zju:{variant}",
        confidence_threshold=float(confidence_threshold),
        points0=points0,
        points1=points1,
        scores=scores,
        width0=int(meta0["content_w"]),
        height0=int(meta0["content_h"]),
        width1=int(meta1["content_w"]),
        height1=int(meta1["content_h"]),
        input_w=int(meta0["content_w"] + meta0["pad_w"]),
        input_h=int(meta0["content_h"] + meta0["pad_h"]),
        image0_gray=img1_resized,
        image1_gray=img2_resized,
        image0_rgb=img1_rgb_resized,
        image1_rgb=img2_rgb_resized,
    )
    loader_meta = _zju_loftr_variant_meta.get(variant, {})
    diagnostics["zju_variant"] = variant
    diagnostics["zju_loader"] = str(loader_meta.get("loader", "unknown"))
    diagnostics["zju_checkpoint_path"] = loader_meta.get("checkpoint_path")
    diagnostics["zju_repo_dir"] = loader_meta.get("repo_dir")
    diagnostics["zju_match_type"] = str(loader_meta.get("match_type", "unknown"))
    diagnostics["zju_model_class"] = str(
        loader_meta.get(
            "model_class",
            f"{matcher.__class__.__module__}.{matcher.__class__.__name__}",
        )
    )
    return num_matches, num_inliers, score, direction, diagnostics


def _match_loftr_zju_indoor_native(
    img1: Image.Image,
    img2: Image.Image,
    confidence_threshold: float = ZJU_LOFTR_NATIVE_CONFIDENCE_THRESHOLD,
) -> Tuple[int, int, float, Tuple[float, float], Dict[str, Any]]:
    """Backward-compatible indoor_ds native endpoint."""
    return _match_loftr_zju_variant_native(
        img1=img1,
        img2=img2,
        variant="indoor_ds",
        confidence_threshold=confidence_threshold,
    )


def _match_loftr_zju_legacy_native(
    img1: Image.Image,
    img2: Image.Image,
    confidence_threshold: float = ZJU_LOFTR_NATIVE_CONFIDENCE_THRESHOLD,
) -> Tuple[int, int, float, Tuple[float, float], Dict[str, Any]]:
    """Legacy ZJU-like matcher path from before DS/OT split."""
    matcher = _load_zju_loftr_legacy()
    device = next(matcher.parameters()).device

    img1_gray = np.array(img1.convert("L"), dtype=np.float32) / 255.0
    img2_gray = np.array(img2.convert("L"), dtype=np.float32) / 255.0
    img1_rgb = np.array(img1.convert("RGB"), dtype=np.uint8)
    img2_rgb = np.array(img2.convert("RGB"), dtype=np.uint8)

    target_long_side = max(64, int(max(DEFAULT_LOFTR_INPUT_SIZE)))
    img1_resized, meta0 = _resize_by_longest_side_and_pad(img1_gray, target_long_side=target_long_side, multiple=8)
    img2_resized, meta1 = _resize_by_longest_side_and_pad(img2_gray, target_long_side=target_long_side, multiple=8)
    img1_rgb_resized, _ = _resize_rgb_by_longest_side_and_pad(img1_rgb, target_long_side=target_long_side, multiple=8)
    img2_rgb_resized, _ = _resize_rgb_by_longest_side_and_pad(img2_rgb, target_long_side=target_long_side, multiple=8)

    tensor1 = torch.from_numpy(img1_resized).unsqueeze(0).unsqueeze(0).to(device)
    tensor2 = torch.from_numpy(img2_resized).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        correspondences = matcher({"image0": tensor1, "image1": tensor2})

    points0, points1, scores = _extract_loftr_points_and_scores(correspondences)
    num_matches, num_inliers, score, direction, diagnostics = _build_native_loftr_diagnostics(
        matcher_name="loftr_zju_legacy_native",
        checkpoint=f"zju:legacy:{_zju_legacy_checkpoint_path or 'unknown'}",
        confidence_threshold=float(confidence_threshold),
        points0=points0,
        points1=points1,
        scores=scores,
        width0=int(meta0["content_w"]),
        height0=int(meta0["content_h"]),
        width1=int(meta1["content_w"]),
        height1=int(meta1["content_h"]),
        input_w=int(meta0["content_w"] + meta0["pad_w"]),
        input_h=int(meta0["content_h"] + meta0["pad_h"]),
        image0_gray=img1_resized,
        image1_gray=img2_resized,
        image0_rgb=img1_rgb_resized,
        image1_rgb=img2_rgb_resized,
    )
    diagnostics["zju_variant"] = "legacy"
    diagnostics["zju_loader"] = "kornia_legacy"
    diagnostics["zju_checkpoint_path"] = _zju_legacy_checkpoint_path
    diagnostics["zju_repo_dir"] = None
    diagnostics["zju_match_type"] = "legacy_single"
    diagnostics["zju_model_class"] = f"{matcher.__class__.__module__}.{matcher.__class__.__name__}"
    return num_matches, num_inliers, score, direction, diagnostics


def _match_efficient_loftr_hf_native(
    img1: Image.Image,
    img2: Image.Image,
    threshold: float = 0.2,
) -> Tuple[int, int, float, Tuple[float, float], Dict[str, Any]]:
    """HF-parity EfficientLoFTR debug path.

    This mirrors the model card flow:
    processor -> model -> post_process_keypoint_matching(threshold=0.2)
    and reports native matching_scores statistics.
    """
    processor, model = _load_efficient_loftr()
    device = next(model.parameters()).device

    img1_rgb = img1.convert("RGB")
    img2_rgb = img2.convert("RGB")
    image_sizes = [(img1_rgb.height, img1_rgb.width), (img2_rgb.height, img2_rgb.width)]

    processor_inputs = processor(images=[img1_rgb, img2_rgb], return_tensors="pt")
    model_inputs: Dict[str, Any] = {}
    for key, value in processor_inputs.items():
        if isinstance(value, torch.Tensor):
            model_inputs[key] = value.to(device)
        else:
            model_inputs[key] = value

    with torch.no_grad():
        outputs = model(**model_inputs)

    raw_result = _post_process_efficient_matches(
        processor=processor,
        outputs=outputs,
        image_sizes=image_sizes,
        threshold=0.0,
    )
    raw_points0 = _as_xy_points(raw_result.get("keypoints0"))
    raw_points1 = _as_xy_points(raw_result.get("keypoints1"))
    raw_scores = _as_score_vector(raw_result.get("matching_scores"))
    raw_n = int(min(len(raw_points0), len(raw_points1), len(raw_scores)))
    if len(raw_points0) != raw_n:
        raw_points0 = raw_points0[:raw_n]
    if len(raw_points1) != raw_n:
        raw_points1 = raw_points1[:raw_n]
    if len(raw_scores) != raw_n:
        raw_scores = raw_scores[:raw_n]

    result = _post_process_efficient_matches(
        processor=processor,
        outputs=outputs,
        image_sizes=image_sizes,
        threshold=float(threshold),
    )

    points0 = _as_xy_points(result.get("keypoints0"))
    points1 = _as_xy_points(result.get("keypoints1"))
    scores = _as_score_vector(result.get("matching_scores"))

    n = int(min(len(points0), len(points1), len(scores)))
    if len(points0) != n:
        points0 = points0[:n]
    if len(points1) != n:
        points1 = points1[:n]
    if len(scores) != n:
        scores = scores[:n]

    score_summary = _matching_score_summary(scores)
    raw_score_summary = _matching_score_summary(raw_scores)
    hf_score = float(score_summary["mean"]) if n > 0 else 0.0
    img1_gray = np.array(img1_rgb.convert("L"), dtype=np.float32) / 255.0
    img2_gray = np.array(img2_rgb.convert("L"), dtype=np.float32) / 255.0
    filter_analysis = _analyze_loftr_match_filters(
        points0=points0,
        points1=points1,
        scores=scores,
        image0_gray=img1_gray,
        image1_gray=img2_gray,
        image0_rgb=np.array(img1_rgb, dtype=np.uint8),
        image1_rgb=np.array(img2_rgb, dtype=np.uint8),
    )
    filter_keep_mask = np.asarray(filter_analysis.get("filtered_keep_mask", np.empty((0,), dtype=bool)), dtype=bool)
    if filter_keep_mask.shape[0] != n:
        filter_keep_mask = np.ones((n,), dtype=bool)
    filtered_points0 = points0[filter_keep_mask] if n > 0 else np.empty((0, 2), dtype=np.float32)
    filtered_points1 = points1[filter_keep_mask] if n > 0 else np.empty((0, 2), dtype=np.float32)
    filtered_combined = filter_analysis.get("scores", {}).get("combined", {})
    filtered_final_score = float(filtered_combined.get("final_score", 0.0) or 0.0)
    filter_match_sets = _build_filter_match_sets(
        points0=points0,
        points1=points1,
        width0=int(img1_rgb.width),
        height0=int(img1_rgb.height),
        width1=int(img2_rgb.width),
        height1=int(img2_rgb.height),
        filter_analysis=filter_analysis,
    )
    hf_visualization_data_url = _build_efficient_hf_visualization(
        processor=processor,
        img1=img1_rgb,
        img2=img2_rgb,
        post_processed_result=result,
    )

    if n > 0:
        inlier_mask = np.ones((n,), dtype=bool)
        direction = compute_direction_vector(points0, points1, inlier_mask)
    else:
        direction = (0.0, 0.0)

    input_w, input_h = _extract_hw_from_processor_inputs(processor_inputs)
    width0 = int(img1_rgb.width)
    height0 = int(img1_rgb.height)
    width1 = int(img2_rgb.width)
    height1 = int(img2_rgb.height)

    diagnostics = {
        "matcher": "efficient_loftr_hf_native",
        "checkpoint": EFFICIENT_LOFTR_MODEL_ID,
        "confidence_threshold": float(threshold),
        "geometry_model": "none_hf_native",
        "segment_scores": {},
        "score_components": {},
        "match_width": width0,
        "match_height": height0,
        "raw_correspondence_count": int(raw_n),
        "threshold_trials": [
            {
                "threshold": float(threshold),
                "raw_matches": int(raw_n),
                "num_matches": int(n),
                "num_inliers": 0,
                "score": float(hf_score),
                "geometry_model": "none_hf_native",
                "native_score_mean": float(score_summary["mean"]),
                "native_score_median": float(score_summary["median"]),
                "native_score_p95": float(score_summary["p95"]),
            }
        ],
        "loftr_input_width": int(input_w if input_w is not None else width0),
        "loftr_input_height": int(input_h if input_h is not None else height0),
        "ransac_reproj_threshold": None,
        "raw_matches": _sample_normalized_matches(
            points0=raw_points0,
            points1=raw_points1,
            width0=width0,
            height0=height0,
            width1=width1,
            height1=height1,
            max_points=5000,
        ),
        "filtered_match_count": int(len(filtered_points0)),
        "filtered_matches": _sample_normalized_matches(
            points0=filtered_points0,
            points1=filtered_points1,
            width0=width0,
            height0=height0,
            width1=width1,
            height1=height1,
            max_points=5000,
        ),
        "filtered_final_score": float(filtered_final_score),
        "filter_config": filter_analysis.get("config", {}),
        "filter_scores": filter_analysis.get("scores", {}),
        "filter_match_sets": filter_match_sets,
        "inlier_match_count": 0,
        "inlier_matches": [],
        "oracle": {
            "mode": "off",
            "evaluated": False,
            "passed": True,
            "decision": "hf_native_no_oracle",
        },
        "native_matching_scores": score_summary,
        "native_matching_scores_raw": raw_score_summary,
        "hf_visualization_data_url": hf_visualization_data_url,
    }

    # In HF-native mode score is pure EfficientLoFTR confidence summary (mean), not geometric score.
    return int(n), 0, hf_score, direction, diagnostics


def _match_efficient_loftr(
    img1: Image.Image,
    img2: Image.Image,
    confidence_thresholds: Tuple[float, ...] = LOFTR_CONFIDENCE_LEVELS,
    reproj_threshold: float = RANSAC_REPROJ_THRESHOLD,
) -> Tuple[int, int, float, Tuple[float, float], Dict[str, Any]]:
    """Match using HF EfficientLoFTR with config aligned to model-card examples."""
    processor, model = _load_efficient_loftr()
    device = next(model.parameters()).device

    img1_rgb = img1.convert("RGB")
    img2_rgb = img2.convert("RGB")
    image_sizes = [(img1_rgb.height, img1_rgb.width), (img2_rgb.height, img2_rgb.width)]

    # Match HF example input format.
    processor_inputs = processor(images=[img1_rgb, img2_rgb], return_tensors="pt")
    model_inputs: Dict[str, Any] = {}
    for key, value in processor_inputs.items():
        if isinstance(value, torch.Tensor):
            model_inputs[key] = value.to(device)
        else:
            model_inputs[key] = value

    with torch.no_grad():
        outputs = model(**model_inputs)

    raw_result = _post_process_efficient_matches(
        processor=processor,
        outputs=outputs,
        image_sizes=image_sizes,
        threshold=0.0,
    )
    raw_points0 = _as_xy_points(raw_result.get("keypoints0"))
    raw_correspondence_count = int(len(raw_points0))

    input_w, input_h = _extract_hw_from_processor_inputs(processor_inputs)
    width0 = int(img1_rgb.width)
    height0 = int(img1_rgb.height)
    width1 = int(img2_rgb.width)
    height1 = int(img2_rgb.height)

    best_result: Optional[Dict[str, Any]] = None
    threshold_trials: List[Dict[str, Any]] = []
    for threshold in confidence_thresholds:
        result = _post_process_efficient_matches(
            processor=processor,
            outputs=outputs,
            image_sizes=image_sizes,
            threshold=float(threshold),
        )
        points0 = _as_xy_points(result.get("keypoints0"))
        points1 = _as_xy_points(result.get("keypoints1"))
        match_scores = _as_score_vector(result.get("matching_scores"))
        num_matches = int(min(len(points0), len(points1)))
        if len(points0) != num_matches:
            points0 = points0[:num_matches]
        if len(points1) != num_matches:
            points1 = points1[:num_matches]
        if len(match_scores) >= num_matches:
            match_scores = match_scores[:num_matches]

        trial: Dict[str, Any] = {
            "threshold": float(threshold),
            "raw_matches": int(raw_correspondence_count),
            "num_matches": int(num_matches),
            "num_inliers": 0,
            "score": 0.0,
            "geometry_model": "none",
        }
        if num_matches < 8:
            threshold_trials.append(trial)
            continue

        inlier_mask, geom_model = _estimate_geometric_inliers(
            points0,
            points1,
            reproj_threshold=reproj_threshold,
        )
        trial["geometry_model"] = geom_model
        if inlier_mask is None:
            threshold_trials.append(trial)
            continue
        num_inliers = int(inlier_mask.sum())
        trial["num_inliers"] = int(num_inliers)
        if num_inliers <= 0:
            threshold_trials.append(trial)
            continue

        inlier_points0 = points0[inlier_mask]
        inlier_points1 = points1[inlier_mask]
        direction = compute_direction_vector(points0, points1, inlier_mask)
        segment_scores = _compute_segment_scores(
            inlier_points0,
            inlier_points1,
            width0=width0,
            height0=height0,
            width1=width1,
            height1=height1,
        )
        score_components = _compute_transition_geometry_components(
            num_matches=num_matches,
            num_inliers=num_inliers,
            inlier_points0=inlier_points0,
            inlier_points1=inlier_points1,
            width0=width0,
            height0=height0,
            width1=width1,
            height1=height1,
            segment_scores=segment_scores,
            # EfficientLoFTR at low confidence returns many correspondences;
            # use a softer ratio calibration to avoid hard-zeroing valid geometry.
            inlier_ratio_zero_point=0.03,
            inlier_ratio_full_point=0.35,
        )
        score = float(score_components["final_score"])
        trial["score"] = float(score)
        trial["native_score_mean"] = float(np.mean(match_scores)) if match_scores.size > 0 else 0.0
        trial["native_score_median"] = float(np.median(match_scores)) if match_scores.size > 0 else 0.0
        trial["native_score_p95"] = float(np.percentile(match_scores, 95)) if match_scores.size > 0 else 0.0
        threshold_trials.append(trial)

        candidate = {
            "num_matches": num_matches,
            "num_inliers": num_inliers,
            "score": score,
            "direction": direction,
            "confidence_threshold": float(threshold),
            "checkpoint": EFFICIENT_LOFTR_MODEL_ID,
            "geometry_model": geom_model,
            "segment_scores": segment_scores,
            "score_components": score_components,
            "raw_points0": points0,
            "raw_points1": points1,
            "matching_scores": match_scores,
            "inlier_points0": inlier_points0,
            "inlier_points1": inlier_points1,
            "width0": int(width0),
            "height0": int(height0),
            "width1": int(width1),
            "height1": int(height1),
            "padded_width0": int(input_w if input_w is not None else width0),
            "padded_height0": int(input_h if input_h is not None else height0),
            "padded_width1": int(input_w if input_w is not None else width1),
            "padded_height1": int(input_h if input_h is not None else height1),
        }
        best_result = _pick_better_match_result(best_result, candidate)

    if best_result is None:
        return (
            0,
            0,
            0.0,
            (0.0, 0.0),
            {
                "matcher": "efficient_loftr",
                "checkpoint": EFFICIENT_LOFTR_MODEL_ID,
                "confidence_threshold": float(confidence_thresholds[-1]) if confidence_thresholds else 0.0,
                "geometry_model": "none",
                "segment_scores": _compute_segment_scores(
                    np.empty((0, 2), dtype=np.float32),
                    np.empty((0, 2), dtype=np.float32),
                    width0=width0,
                    height0=height0,
                    width1=width1,
                    height1=height1,
                ),
                "score_components": _compute_transition_geometry_components(
                    num_matches=0,
                    num_inliers=0,
                    inlier_points0=np.empty((0, 2), dtype=np.float32),
                    inlier_points1=np.empty((0, 2), dtype=np.float32),
                    width0=width0,
                    height0=height0,
                    width1=width1,
                    height1=height1,
                    segment_scores={},
                ),
                "match_width": int(width0),
                "match_height": int(height0),
                "raw_correspondence_count": int(raw_correspondence_count),
                "threshold_trials": threshold_trials,
                "loftr_input_width": int(input_w if input_w is not None else width0),
                "loftr_input_height": int(input_h if input_h is not None else height0),
                "ransac_reproj_threshold": float(reproj_threshold),
                "raw_matches": [],
                "filtered_match_count": 0,
                "filtered_matches": [],
                "filtered_final_score": 0.0,
                "filter_config": {
                    "local_window": float(LOFTR_FILTER_LOCAL_WINDOW),
                    "min_grad_norm": float(LOFTR_FILTER_MIN_GRAD),
                    "min_local_std_norm": float(LOFTR_FILTER_MIN_STD),
                    "final_keep_weight": float(LOFTR_FILTER_FINAL_KEEP_WEIGHT),
                },
                "filter_scores": {},
                "filter_match_sets": {},
                "inlier_match_count": 0,
                "inlier_matches": [],
                "oracle": {
                    "mode": _normalized_oracle_mode(),
                    "evaluated": False,
                    "passed": True,
                    "decision": "insufficient_inliers",
                },
            },
        )

    num_matches = int(best_result["num_matches"])
    num_inliers = int(best_result["num_inliers"])
    score = float(best_result["score"])
    direction = best_result["direction"]
    segment_scores = best_result.get("segment_scores", {})
    raw_points0 = np.ascontiguousarray(best_result.get("raw_points0", np.empty((0, 2), dtype=np.float32)))
    raw_points1 = np.ascontiguousarray(best_result.get("raw_points1", np.empty((0, 2), dtype=np.float32)))
    matching_scores = _as_score_vector(best_result.get("matching_scores"))
    img1_gray = np.array(img1_rgb.convert("L"), dtype=np.float32) / 255.0
    img2_gray = np.array(img2_rgb.convert("L"), dtype=np.float32) / 255.0
    filter_analysis = _analyze_loftr_match_filters(
        points0=raw_points0,
        points1=raw_points1,
        scores=matching_scores,
        image0_gray=img1_gray,
        image1_gray=img2_gray,
        image0_rgb=np.array(img1_rgb, dtype=np.uint8),
        image1_rgb=np.array(img2_rgb, dtype=np.uint8),
    )
    filter_keep_mask = np.asarray(filter_analysis.get("filtered_keep_mask", np.empty((0,), dtype=bool)), dtype=bool)
    if filter_keep_mask.shape[0] != len(raw_points0):
        filter_keep_mask = np.ones((len(raw_points0),), dtype=bool)
    filtered_points0 = raw_points0[filter_keep_mask] if len(raw_points0) > 0 else np.empty((0, 2), dtype=np.float32)
    filtered_points1 = raw_points1[filter_keep_mask] if len(raw_points0) > 0 else np.empty((0, 2), dtype=np.float32)
    filter_combined = filter_analysis.get("scores", {}).get("combined", {})
    filtered_final_score = float(filter_combined.get("final_score", 0.0) or 0.0)
    filter_match_sets = _build_filter_match_sets(
        points0=raw_points0,
        points1=raw_points1,
        width0=width0,
        height0=height0,
        width1=width1,
        height1=height1,
        filter_analysis=filter_analysis,
    )
    filter_match_sets = _build_filter_match_sets(
        points0=raw_points0,
        points1=raw_points1,
        width0=width0,
        height0=height0,
        width1=width1,
        height1=height1,
        filter_analysis=filter_analysis,
    )
    inlier_points0 = np.ascontiguousarray(best_result.get("inlier_points0", np.empty((0, 2), dtype=np.float32)))
    inlier_points1 = np.ascontiguousarray(best_result.get("inlier_points1", np.empty((0, 2), dtype=np.float32)))
    num_matches, num_inliers, score, direction, oracle_diag = _apply_kornia_oracle_to_match(
        num_matches=num_matches,
        num_inliers=num_inliers,
        score=score,
        direction=direction,
        inlier_points0=inlier_points0,
        inlier_points1=inlier_points1,
        width=width0,
        height=height0,
        segment_scores=segment_scores,
    )

    diagnostics = {
        "matcher": "efficient_loftr",
        "checkpoint": EFFICIENT_LOFTR_MODEL_ID,
        "confidence_threshold": best_result.get("confidence_threshold"),
        "geometry_model": best_result.get("geometry_model"),
        "segment_scores": segment_scores,
        "score_components": best_result.get("score_components"),
        "match_width": width0,
        "match_height": height0,
        "raw_correspondence_count": int(raw_correspondence_count),
        "threshold_trials": threshold_trials,
        "loftr_input_width": int(best_result.get("padded_width0", input_w if input_w is not None else width0)),
        "loftr_input_height": int(best_result.get("padded_height0", input_h if input_h is not None else height0)),
        "ransac_reproj_threshold": float(reproj_threshold),
        "raw_matches": _sample_normalized_matches(
            points0=raw_points0,
            points1=raw_points1,
            width0=width0,
            height0=height0,
            width1=width1,
            height1=height1,
            max_points=5000,
        ),
        "filtered_match_count": int(len(filtered_points0)),
        "filtered_matches": _sample_normalized_matches(
            points0=filtered_points0,
            points1=filtered_points1,
            width0=width0,
            height0=height0,
            width1=width1,
            height1=height1,
            max_points=5000,
        ),
        "filtered_final_score": float(filtered_final_score),
        "filter_config": filter_analysis.get("config", {}),
        "filter_scores": filter_analysis.get("scores", {}),
        "filter_match_sets": filter_match_sets,
        "inlier_match_count": int(num_inliers),
        "inlier_matches": _sample_normalized_matches(
            points0=inlier_points0,
            points1=inlier_points1,
            width0=width0,
            height0=height0,
            width1=width1,
            height1=height1,
            max_points=5000,
        ),
        "oracle": oracle_diag,
        "native_matching_scores": _matching_score_summary(matching_scores),
        "native_matching_scores_raw": _matching_score_summary(matching_scores),
    }
    return num_matches, num_inliers, score, direction, diagnostics


def _load_matcher():
    """Load learned feature matcher (lazy initialization)."""
    global _matcher_type

    # If matcher type was previously set to ORB fallback, keep it.
    if _matcher_type == "orb":
        return None, _matcher_type

    # Prefer indoor LoFTR as primary; outdoor can be used as fallback in matching.
    try:
        matcher = _load_loftr_checkpoint("indoor_new")
        _matcher_type = "loftr"
        return matcher, _matcher_type
    except Exception as indoor_err:
        logger.debug("LoFTR indoor_new unavailable: %s", indoor_err)
        try:
            matcher = _load_loftr_checkpoint("outdoor")
            _matcher_type = "loftr"
            return matcher, _matcher_type
        except Exception as outdoor_err:
            logger.debug("LoFTR outdoor unavailable: %s", outdoor_err)

    _matcher_type = "orb"
    logger.info("Using ORB matcher (kornia LoFTR unavailable)")
    return None, _matcher_type


def _pick_better_match_result(
    current_best: Optional[Dict[str, Any]],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    if current_best is None:
        return candidate
    # Prioritize geometric quality first; inlier count is secondary.
    score_delta = float(candidate["score"]) - float(current_best["score"])
    if score_delta > 0.02:
        return candidate
    if score_delta < -0.02:
        return current_best
    if candidate["num_inliers"] > current_best["num_inliers"]:
        return candidate
    if candidate["num_inliers"] < current_best["num_inliers"]:
        return current_best
    if candidate["score"] > current_best["score"]:
        return candidate
    if candidate["score"] < current_best["score"]:
        return current_best
    if candidate["num_matches"] > current_best["num_matches"]:
        return candidate
    return current_best


def _run_loftr_with_thresholds(
    matcher,
    img1_resized: np.ndarray,
    img2_resized: np.ndarray,
    confidence_thresholds: Tuple[float, ...],
    checkpoint: str,
    content_w0: int,
    content_h0: int,
    content_w1: int,
    content_h1: int,
    reproj_threshold: float = RANSAC_REPROJ_THRESHOLD,
) -> Dict[str, Any]:
    """Run LoFTR once and evaluate adaptive confidence thresholds."""
    device = next(matcher.parameters()).device
    tensor1 = torch.from_numpy(img1_resized).unsqueeze(0).unsqueeze(0).to(device)
    tensor2 = torch.from_numpy(img2_resized).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        correspondences = matcher({"image0": tensor1, "image1": tensor2})

    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    confidence = correspondences["confidence"].cpu().numpy()
    h0, w0 = img1_resized.shape[:2]
    h1, w1 = img2_resized.shape[:2]
    if ENABLE_PHOTOMETRIC_PREFILTER:
        grad0 = _compute_gradient_magnitude(img1_resized)
        grad1 = _compute_gradient_magnitude(img2_resized)
        photometric_mask = _photometric_consistency_mask(
            points0=mkpts0,
            points1=mkpts1,
            image0=img1_resized,
            image1=img2_resized,
            grad0=grad0,
            grad1=grad1,
        )
    else:
        photometric_mask = np.ones((len(confidence),), dtype=bool)
    best_result: Optional[Dict[str, Any]] = None
    threshold_trials: List[Dict[str, Any]] = []
    for threshold in confidence_thresholds:
        conf_mask = confidence > float(threshold)
        raw_matches = int(conf_mask.sum())
        conf_mask = conf_mask & photometric_mask
        num_matches = int(conf_mask.sum())
        trial: Dict[str, Any] = {
            "threshold": float(threshold),
            "raw_matches": int(raw_matches),
            "num_matches": int(num_matches),
            "num_inliers": 0,
            "score": 0.0,
            "geometry_model": "none",
        }
        if num_matches < 8:
            threshold_trials.append(trial)
            continue

        points0 = np.ascontiguousarray(mkpts0[conf_mask], dtype=np.float32)
        points1 = np.ascontiguousarray(mkpts1[conf_mask], dtype=np.float32)
        match_scores = np.ascontiguousarray(confidence[conf_mask], dtype=np.float32)
        if points0.shape[0] < 8 or points1.shape[0] < 8 or points0.shape[0] != points1.shape[0]:
            threshold_trials.append(trial)
            continue

        inlier_mask, geom_model = _estimate_geometric_inliers(
            points0,
            points1,
            reproj_threshold=reproj_threshold,
        )
        trial["geometry_model"] = geom_model
        if inlier_mask is None:
            threshold_trials.append(trial)
            continue
        num_inliers = int(inlier_mask.sum())
        trial["num_inliers"] = int(num_inliers)
        if num_inliers <= 0:
            threshold_trials.append(trial)
            continue

        inlier_points0 = points0[inlier_mask]
        inlier_points1 = points1[inlier_mask]
        direction = compute_direction_vector(points0, points1, inlier_mask)
        segment_scores = _compute_segment_scores(
            inlier_points0,
            inlier_points1,
            width0=content_w0,
            height0=content_h0,
            width1=content_w1,
            height1=content_h1,
        )
        score_components = _compute_transition_geometry_components(
            num_matches=num_matches,
            num_inliers=num_inliers,
            inlier_points0=inlier_points0,
            inlier_points1=inlier_points1,
            width0=content_w0,
            height0=content_h0,
            width1=content_w1,
            height1=content_h1,
            segment_scores=segment_scores,
        )
        score = float(score_components["final_score"])
        trial["score"] = float(score)
        trial["native_score_mean"] = float(np.mean(match_scores)) if match_scores.size > 0 else 0.0
        trial["native_score_median"] = float(np.median(match_scores)) if match_scores.size > 0 else 0.0
        trial["native_score_p95"] = float(np.percentile(match_scores, 95)) if match_scores.size > 0 else 0.0
        threshold_trials.append(trial)

        candidate = {
            "num_matches": num_matches,
            "num_inliers": num_inliers,
            "score": score,
            "direction": direction,
            "confidence_threshold": float(threshold),
            "checkpoint": checkpoint,
            "geometry_model": geom_model,
            "segment_scores": segment_scores,
            "score_components": score_components,
            "raw_points0": points0,
            "raw_points1": points1,
            "matching_scores": match_scores,
            "inlier_points0": inlier_points0,
            "inlier_points1": inlier_points1,
            "width0": int(content_w0),
            "height0": int(content_h0),
            "width1": int(content_w1),
            "height1": int(content_h1),
            "padded_width0": int(w0),
            "padded_height0": int(h0),
            "padded_width1": int(w1),
            "padded_height1": int(h1),
        }
        best_result = _pick_better_match_result(best_result, candidate)

    if best_result is None:
        return {
            "num_matches": 0,
            "num_inliers": 0,
            "score": 0.0,
            "direction": (0.0, 0.0),
            "confidence_threshold": None,
            "checkpoint": checkpoint,
            "geometry_model": "none",
            "segment_scores": _compute_segment_scores(
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                width0=content_w0,
                height0=content_h0,
                width1=content_w1,
                height1=content_h1,
            ),
            "score_components": _compute_transition_geometry_components(
                num_matches=0,
                num_inliers=0,
                inlier_points0=np.empty((0, 2), dtype=np.float32),
                inlier_points1=np.empty((0, 2), dtype=np.float32),
                width0=content_w0,
                height0=content_h0,
                width1=content_w1,
                height1=content_h1,
                segment_scores={},
            ),
            "raw_points0": np.empty((0, 2), dtype=np.float32),
            "raw_points1": np.empty((0, 2), dtype=np.float32),
            "matching_scores": np.empty((0,), dtype=np.float32),
            "inlier_points0": np.empty((0, 2), dtype=np.float32),
            "inlier_points1": np.empty((0, 2), dtype=np.float32),
            "width0": int(content_w0),
            "height0": int(content_h0),
            "width1": int(content_w1),
            "height1": int(content_h1),
            "padded_width0": int(w0),
            "padded_height0": int(h0),
            "padded_width1": int(w1),
            "padded_height1": int(h1),
            "raw_correspondence_count": int(len(confidence)),
            "threshold_trials": threshold_trials,
        }
    best_result["raw_correspondence_count"] = int(len(confidence))
    best_result["threshold_trials"] = threshold_trials
    return best_result


def match_image_pair(
    img1: Image.Image,
    img2: Image.Image,
    use_orb_prefilter: bool = False,
    return_diagnostics: bool = False,
    debug_options: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int, float, Tuple[float, float]] | Tuple[int, int, float, Tuple[float, float], Dict[str, Any]]:
    """Match two images using learned features with ORB pre-filtering.

    Performance optimization (optional): ORB is ~10x faster than LoFTR.
    - If ORB shows ≤2 inliers → definitely no overlap, skip LoFTR
    - If ORB shows ≥15 inliers → definitely overlap, use ORB result
    - Otherwise → run LoFTR for accurate matching

    Args:
        img1: First PIL Image
        img2: Second PIL Image
        use_orb_prefilter: Whether to use ORB as pre-filter (default True)

    Returns:
        Tuple of (num_matches, num_inliers, overlap_score, direction_vector).
        When return_diagnostics=True, appends a diagnostics dict.
        direction_vector is (dx, dy) showing how content shifted from img1 to img2.
    """
    # Normalize orientation so backend geometry uses same view as browser rendering.
    img1 = ImageOps.exif_transpose(img1)
    img2 = ImageOps.exif_transpose(img2)

    options = debug_options or {}
    matcher_preference = str(options.get("matcher", "current")).strip().lower()
    if matcher_preference in {"", "current", "default"}:
        matcher_preference = DEFAULT_PRODUCTION_MATCHER
    loftr_input_size = DEFAULT_LOFTR_INPUT_SIZE
    confidence_thresholds = LOFTR_CONFIDENCE_LEVELS
    has_custom_thresholds = False
    reproj_threshold = RANSAC_REPROJ_THRESHOLD
    enable_outdoor_fallback = True

    raw_input_size = options.get("loftr_input_size")
    if isinstance(raw_input_size, (list, tuple)) and len(raw_input_size) == 2:
        try:
            parsed_w = int(raw_input_size[0])
            parsed_h = int(raw_input_size[1])
            if parsed_w > 0 and parsed_h > 0:
                loftr_input_size = (parsed_w, parsed_h)
        except (TypeError, ValueError):
            pass

    raw_thresholds = options.get("confidence_thresholds")
    if isinstance(raw_thresholds, (list, tuple)) and raw_thresholds:
        parsed_thresholds = []
        for value in raw_thresholds:
            try:
                parsed_thresholds.append(float(value))
            except (TypeError, ValueError):
                continue
        parsed_thresholds = sorted({v for v in parsed_thresholds if 0.0 <= v <= 1.0}, reverse=True)
        if parsed_thresholds:
            confidence_thresholds = tuple(parsed_thresholds)
            has_custom_thresholds = True

    raw_reproj = options.get("reproj_threshold")
    try:
        if raw_reproj is not None:
            parsed_reproj = float(raw_reproj)
            if parsed_reproj > 0:
                reproj_threshold = parsed_reproj
    except (TypeError, ValueError):
        pass

    raw_outdoor = options.get("enable_outdoor_fallback")
    if isinstance(raw_outdoor, bool):
        enable_outdoor_fallback = raw_outdoor

    if matcher_preference == "efficient_hf":
        return _match_efficient_loftr_hf_native(
            img1=img1,
            img2=img2,
            threshold=0.2,
        )

    if matcher_preference in {"loftr_zju_indoor_native", "loftr_zju_indoor_ds_native"}:
        result = _match_loftr_zju_indoor_native(
            img1=img1,
            img2=img2,
            confidence_threshold=ZJU_LOFTR_NATIVE_CONFIDENCE_THRESHOLD,
        )
        result = _maybe_retry_reverse_native(
            img1=img1,
            img2=img2,
            forward_result=result,
            run_fn=_match_loftr_zju_indoor_native,
            run_kwargs={"confidence_threshold": ZJU_LOFTR_NATIVE_CONFIDENCE_THRESHOLD},
        )
        if return_diagnostics:
            return result
        return result[0], result[1], result[2], result[3]

    if matcher_preference == "loftr_zju_legacy_native":
        result = _match_loftr_zju_legacy_native(
            img1=img1,
            img2=img2,
            confidence_threshold=ZJU_LOFTR_NATIVE_CONFIDENCE_THRESHOLD,
        )
        result = _maybe_retry_reverse_native(
            img1=img1,
            img2=img2,
            forward_result=result,
            run_fn=_match_loftr_zju_legacy_native,
            run_kwargs={"confidence_threshold": ZJU_LOFTR_NATIVE_CONFIDENCE_THRESHOLD},
        )
        if return_diagnostics:
            return result
        return result[0], result[1], result[2], result[3]

    if matcher_preference == "loftr_zju_indoor_ot_native":
        result = _match_loftr_zju_variant_native(
            img1=img1,
            img2=img2,
            variant="indoor_ot",
            confidence_threshold=ZJU_LOFTR_NATIVE_CONFIDENCE_THRESHOLD,
        )
        result = _maybe_retry_reverse_native(
            img1=img1,
            img2=img2,
            forward_result=result,
            run_fn=_match_loftr_zju_variant_native,
            run_kwargs={
                "variant": "indoor_ot",
                "confidence_threshold": ZJU_LOFTR_NATIVE_CONFIDENCE_THRESHOLD,
            },
        )
        if return_diagnostics:
            return result
        return result[0], result[1], result[2], result[3]

    if matcher_preference == "loftr_zju_outdoor_ds_native":
        result = _match_loftr_zju_variant_native(
            img1=img1,
            img2=img2,
            variant="outdoor_ds",
            confidence_threshold=ZJU_LOFTR_NATIVE_CONFIDENCE_THRESHOLD,
        )
        result = _maybe_retry_reverse_native(
            img1=img1,
            img2=img2,
            forward_result=result,
            run_fn=_match_loftr_zju_variant_native,
            run_kwargs={
                "variant": "outdoor_ds",
                "confidence_threshold": ZJU_LOFTR_NATIVE_CONFIDENCE_THRESHOLD,
            },
        )
        if return_diagnostics:
            return result
        return result[0], result[1], result[2], result[3]

    if matcher_preference == "loftr_zju_outdoor_ot_native":
        result = _match_loftr_zju_variant_native(
            img1=img1,
            img2=img2,
            variant="outdoor_ot",
            confidence_threshold=ZJU_LOFTR_NATIVE_CONFIDENCE_THRESHOLD,
        )
        result = _maybe_retry_reverse_native(
            img1=img1,
            img2=img2,
            forward_result=result,
            run_fn=_match_loftr_zju_variant_native,
            run_kwargs={
                "variant": "outdoor_ot",
                "confidence_threshold": ZJU_LOFTR_NATIVE_CONFIDENCE_THRESHOLD,
            },
        )
        if return_diagnostics:
            return result
        return result[0], result[1], result[2], result[3]

    if matcher_preference == "loftr_kornia_indoor_native":
        result = _match_loftr_kornia_indoor_native(
            img1=img1,
            img2=img2,
            confidence_threshold=LOFTR_NATIVE_CONFIDENCE_THRESHOLD,
        )
        result = _maybe_retry_reverse_native(
            img1=img1,
            img2=img2,
            forward_result=result,
            run_fn=_match_loftr_kornia_indoor_native,
            run_kwargs={"confidence_threshold": LOFTR_NATIVE_CONFIDENCE_THRESHOLD},
        )
        if return_diagnostics:
            return result
        return result[0], result[1], result[2], result[3]

    if matcher_preference == "efficient":
        efficient_thresholds = confidence_thresholds if has_custom_thresholds else EFFICIENT_LOFTR_CONFIDENCE_LEVELS
        num_matches, num_inliers, score, direction, diagnostics = _match_efficient_loftr(
            img1=img1,
            img2=img2,
            confidence_thresholds=efficient_thresholds,
            reproj_threshold=reproj_threshold,
        )
        if return_diagnostics:
            return num_matches, num_inliers, score, direction, diagnostics
        return num_matches, num_inliers, score, direction

    matcher, matcher_type = _load_matcher()

    # If LoFTR is not available, just use ORB
    if matcher_type != "loftr" or matcher is None:
        num_matches, num_inliers, score, direction, diagnostics = _match_orb(
            img1,
            img2,
            reproj_threshold=reproj_threshold,
        )
        if return_diagnostics:
            return num_matches, num_inliers, score, direction, diagnostics
        return num_matches, num_inliers, score, direction

    # ORB pre-filter is disabled by default for consistency.
    # It can produce unstable errors and false positives on repetitive textures.
    if use_orb_prefilter:
        orb_matches, orb_inliers, orb_score, orb_direction, orb_diag = _match_orb(
            img1,
            img2,
            reproj_threshold=reproj_threshold,
        )

        # Definitely no overlap - skip LoFTR
        if orb_inliers <= ORB_QUICK_REJECT_INLIERS:
            logger.debug(f"ORB pre-filter: {orb_inliers} inliers ≤ {ORB_QUICK_REJECT_INLIERS} → skip LoFTR")
            if return_diagnostics:
                return orb_matches, orb_inliers, orb_score, orb_direction, orb_diag
            return orb_matches, orb_inliers, orb_score, orb_direction

        # Definitely overlap - use ORB result (skip expensive LoFTR)
        if orb_inliers >= ORB_QUICK_ACCEPT_INLIERS:
            logger.debug(f"ORB pre-filter: {orb_inliers} inliers ≥ {ORB_QUICK_ACCEPT_INLIERS} → use ORB")
            if return_diagnostics:
                return orb_matches, orb_inliers, orb_score, orb_direction, orb_diag
            return orb_matches, orb_inliers, orb_score, orb_direction

        # Ambiguous case - need LoFTR for accuracy
        logger.debug(f"ORB pre-filter: {orb_inliers} inliers → ambiguous, running LoFTR")

    num_matches, num_inliers, score, direction, diagnostics = _match_loftr(
        matcher,
        img1,
        img2,
        input_size=loftr_input_size,
        confidence_thresholds=confidence_thresholds,
        reproj_threshold=reproj_threshold,
        enable_outdoor_fallback=enable_outdoor_fallback,
    )
    if return_diagnostics:
        return num_matches, num_inliers, score, direction, diagnostics
    return num_matches, num_inliers, score, direction


def _match_loftr(
    matcher,
    img1: Image.Image,
    img2: Image.Image,
    input_size: Tuple[int, int] = DEFAULT_LOFTR_INPUT_SIZE,
    confidence_thresholds: Tuple[float, ...] = LOFTR_CONFIDENCE_LEVELS,
    reproj_threshold: float = RANSAC_REPROJ_THRESHOLD,
    enable_outdoor_fallback: bool = True,
) -> Tuple[int, int, float, Tuple[float, float], Dict[str, Any]]:
    """Match using LoFTR (learned dense matching)."""

    # Convert to grayscale tensors
    img1_gray = np.array(img1.convert("L"), dtype=np.float32) / 255.0
    img2_gray = np.array(img2.convert("L"), dtype=np.float32) / 255.0
    img1_rgb = np.array(img1.convert("RGB"), dtype=np.uint8)
    img2_rgb = np.array(img2.convert("RGB"), dtype=np.uint8)

    # Resize by longest side preserving aspect ratio, then pad to /8.
    target_long_side = max(64, int(max(input_size)))
    img1_resized, meta0 = _resize_by_longest_side_and_pad(
        img1_gray,
        target_long_side=target_long_side,
        multiple=8,
    )
    img2_resized, meta1 = _resize_by_longest_side_and_pad(
        img2_gray,
        target_long_side=target_long_side,
        multiple=8,
    )
    img1_rgb_resized, _ = _resize_rgb_by_longest_side_and_pad(
        img1_rgb,
        target_long_side=target_long_side,
        multiple=8,
    )
    img2_rgb_resized, _ = _resize_rgb_by_longest_side_and_pad(
        img2_rgb,
        target_long_side=target_long_side,
        multiple=8,
    )
    best_result = _run_loftr_with_thresholds(
        matcher=matcher,
        img1_resized=img1_resized,
        img2_resized=img2_resized,
        confidence_thresholds=confidence_thresholds,
        checkpoint="indoor_new",
        content_w0=int(meta0["content_w"]),
        content_h0=int(meta0["content_h"]),
        content_w1=int(meta1["content_w"]),
        content_h1=int(meta1["content_h"]),
        reproj_threshold=reproj_threshold,
    )

    # Fallback to outdoor checkpoint when indoor does not produce robust geometry.
    if enable_outdoor_fallback and best_result["num_inliers"] < LOFTR_OUTDOOR_FALLBACK_MIN_INLIERS:
        try:
            outdoor_matcher = _load_loftr_checkpoint("outdoor")
            outdoor_result = _run_loftr_with_thresholds(
                matcher=outdoor_matcher,
                img1_resized=img1_resized,
                img2_resized=img2_resized,
                confidence_thresholds=confidence_thresholds,
                checkpoint="outdoor",
                content_w0=int(meta0["content_w"]),
                content_h0=int(meta0["content_h"]),
                content_w1=int(meta1["content_w"]),
                content_h1=int(meta1["content_h"]),
                reproj_threshold=reproj_threshold,
            )
            best_result = _pick_better_match_result(best_result, outdoor_result)
        except Exception as outdoor_err:
            logger.debug("Outdoor LoFTR fallback unavailable: %s", outdoor_err)

    num_matches = int(best_result["num_matches"])
    num_inliers = int(best_result["num_inliers"])
    score = float(best_result["score"])
    direction = best_result["direction"]
    segment_scores = best_result.get("segment_scores", {})
    raw_points0 = np.ascontiguousarray(best_result.get("raw_points0", np.empty((0, 2), dtype=np.float32)))
    raw_points1 = np.ascontiguousarray(best_result.get("raw_points1", np.empty((0, 2), dtype=np.float32)))
    matching_scores = _as_score_vector(best_result.get("matching_scores"))
    width0 = int(best_result.get("width0", int(meta0["content_w"])))
    height0 = int(best_result.get("height0", int(meta0["content_h"])))
    width1 = int(best_result.get("width1", int(meta1["content_w"])))
    height1 = int(best_result.get("height1", int(meta1["content_h"])))
    inlier_points0 = np.ascontiguousarray(best_result.get("inlier_points0", np.empty((0, 2), dtype=np.float32)))
    inlier_points1 = np.ascontiguousarray(best_result.get("inlier_points1", np.empty((0, 2), dtype=np.float32)))
    filter_analysis = _analyze_loftr_match_filters(
        points0=raw_points0,
        points1=raw_points1,
        scores=matching_scores,
        image0_gray=img1_resized,
        image1_gray=img2_resized,
        image0_rgb=img1_rgb_resized,
        image1_rgb=img2_rgb_resized,
    )
    filter_keep_mask = np.asarray(filter_analysis.get("filtered_keep_mask", np.empty((0,), dtype=bool)), dtype=bool)
    if filter_keep_mask.shape[0] != len(raw_points0):
        filter_keep_mask = np.ones((len(raw_points0),), dtype=bool)
    filtered_points0 = raw_points0[filter_keep_mask] if len(raw_points0) > 0 else np.empty((0, 2), dtype=np.float32)
    filtered_points1 = raw_points1[filter_keep_mask] if len(raw_points0) > 0 else np.empty((0, 2), dtype=np.float32)
    filter_combined = filter_analysis.get("scores", {}).get("combined", {})
    filtered_final_score = float(filter_combined.get("final_score", 0.0) or 0.0)
    num_matches, num_inliers, score, direction, oracle_diag = _apply_kornia_oracle_to_match(
        num_matches=num_matches,
        num_inliers=num_inliers,
        score=score,
        direction=direction,
        inlier_points0=inlier_points0,
        inlier_points1=inlier_points1,
        width=width0,
        height=height0,
        segment_scores=segment_scores,
    )

    diagnostics = {
        "matcher": "loftr",
        "checkpoint": best_result.get("checkpoint"),
        "confidence_threshold": best_result.get("confidence_threshold"),
        "geometry_model": best_result.get("geometry_model"),
        "segment_scores": segment_scores,
        "score_components": best_result.get("score_components"),
        "match_width": width0,
        "match_height": height0,
        "raw_correspondence_count": int(best_result.get("raw_correspondence_count", 0)),
        "threshold_trials": best_result.get("threshold_trials", []),
        "loftr_input_width": int(best_result.get("padded_width0", img1_resized.shape[1])),
        "loftr_input_height": int(best_result.get("padded_height0", img1_resized.shape[0])),
        "ransac_reproj_threshold": float(reproj_threshold),
        "raw_matches": _sample_normalized_matches(
            points0=raw_points0,
            points1=raw_points1,
            width0=width0,
            height0=height0,
            width1=width1,
            height1=height1,
            max_points=5000,
        ),
        "filtered_match_count": int(len(filtered_points0)),
        "filtered_matches": _sample_normalized_matches(
            points0=filtered_points0,
            points1=filtered_points1,
            width0=width0,
            height0=height0,
            width1=width1,
            height1=height1,
            max_points=5000,
        ),
        "filtered_final_score": float(filtered_final_score),
        "filter_config": filter_analysis.get("config", {}),
        "filter_scores": filter_analysis.get("scores", {}),
        "filter_match_sets": filter_match_sets,
        "native_matching_scores": _matching_score_summary(matching_scores),
        "native_matching_scores_raw": _matching_score_summary(matching_scores),
        "inlier_match_count": int(num_inliers),
        "inlier_matches": _sample_normalized_matches(
            points0=inlier_points0,
            points1=inlier_points1,
            width0=width0,
            height0=height0,
            width1=width1,
            height1=height1,
            max_points=5000,
        ),
        "oracle": oracle_diag,
    }

    return (
        num_matches,
        num_inliers,
        score,
        direction,
        diagnostics,
    )


def _match_orb(
    img1: Image.Image,
    img2: Image.Image,
    reproj_threshold: float = RANSAC_REPROJ_THRESHOLD,
) -> Tuple[int, int, float, Tuple[float, float], Dict[str, Any]]:
    """Fallback ORB matching."""
    img1_gray = np.array(img1.convert("L"))
    img2_gray = np.array(img2.convert("L"))

    orb = cv2.ORB_create(nfeatures=2000)

    kp1, desc1 = orb.detectAndCompute(img1_gray, None)
    kp2, desc2 = orb.detectAndCompute(img2_gray, None)

    if desc1 is None or desc2 is None:
        return 0, 0, 0.0, (0.0, 0.0), {"matcher": "orb", "segment_scores": {}}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        matches = bf.knnMatch(desc1, desc2, k=2)
    except cv2.error:
        return 0, 0, 0.0, (0.0, 0.0), {"matcher": "orb", "segment_scores": {}}

    # Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    num_matches = len(good_matches)

    if num_matches < 8:
        return num_matches, 0, 0.0, (0.0, 0.0), {"matcher": "orb", "segment_scores": {}}

    src_pts = np.ascontiguousarray([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
    dst_pts = np.ascontiguousarray([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32)

    # Safety check
    if src_pts.shape[0] < 8 or dst_pts.shape[0] < 8 or src_pts.shape[0] != dst_pts.shape[0]:
        return num_matches, 0, 0.0, (0.0, 0.0), {"matcher": "orb", "segment_scores": {}}

    mask, geom_model = _estimate_geometric_inliers(
        src_pts,
        dst_pts,
        reproj_threshold=reproj_threshold,
    )
    if mask is None:
        return num_matches, 0, 0.0, (0.0, 0.0), {"matcher": "orb", "segment_scores": {}}

    num_inliers = int(mask.sum())
    inlier_points0 = src_pts[mask]
    inlier_points1 = dst_pts[mask]
    direction = compute_direction_vector(src_pts, dst_pts, mask)
    h0, w0 = img1_gray.shape[:2]
    h1, w1 = img2_gray.shape[:2]
    segment_scores = _compute_segment_scores(
        inlier_points0,
        inlier_points1,
        width0=int(w0),
        height0=int(h0),
        width1=int(w1),
        height1=int(h1),
    )
    score_components = _compute_transition_geometry_components(
        num_matches=num_matches,
        num_inliers=num_inliers,
        inlier_points0=inlier_points0,
        inlier_points1=inlier_points1,
        width0=int(w0),
        height0=int(h0),
        width1=int(w1),
        height1=int(h1),
        segment_scores=segment_scores,
    )
    score = float(score_components["final_score"])
    num_matches, num_inliers, score, direction, oracle_diag = _apply_kornia_oracle_to_match(
        num_matches=num_matches,
        num_inliers=num_inliers,
        score=score,
        direction=direction,
        inlier_points0=inlier_points0,
        inlier_points1=inlier_points1,
        width=int(w0),
        height=int(h0),
        segment_scores=segment_scores,
    )
    diagnostics = {
        "matcher": "orb",
        "checkpoint": None,
        "confidence_threshold": None,
        "geometry_model": geom_model,
        "segment_scores": segment_scores,
        "score_components": score_components,
        "match_width": int(w0),
        "match_height": int(h0),
        "raw_correspondence_count": int(num_matches),
        "raw_matches": _sample_normalized_matches(
            points0=src_pts,
            points1=dst_pts,
            width0=int(w0),
            height0=int(h0),
            width1=int(w1),
            height1=int(h1),
            max_points=5000,
        ),
        "inlier_match_count": int(num_inliers),
        "inlier_matches": _sample_normalized_matches(
            points0=inlier_points0,
            points1=inlier_points1,
            width0=int(w0),
            height0=int(h0),
            width1=int(w1),
            height1=int(h1),
            max_points=5000,
        ),
        "oracle": oracle_diag,
    }
    return num_matches, num_inliers, score, direction, diagnostics


def compute_overlap_within_cluster(
    images: List[Image.Image],
    photo_ids: List[int],
) -> np.ndarray:
    """Compute pairwise overlap within a single DINOv2 cluster.

    Args:
        images: List of PIL Images in the cluster
        photo_ids: List of photo IDs

    Returns:
        NxN overlap matrix
    """
    n = len(images)
    if n <= 1:
        return np.ones((n, n))

    overlap_matrix = np.zeros((n, n))
    np.fill_diagonal(overlap_matrix, 1.0)

    for i in range(n):
        for j in range(i + 1, n):
            num_matches, num_inliers, score, _ = match_image_pair(images[i], images[j])

            if num_inliers >= MIN_INLIERS_FOR_OVERLAP:
                overlap_matrix[i, j] = score
                overlap_matrix[j, i] = score
                logger.info(
                    f"Overlap {photo_ids[i]} <-> {photo_ids[j]}: "
                    f"{num_matches} matches -> {num_inliers} inliers, "
                    f"score={score:.3f} ✓"
                )
            else:
                logger.debug(
                    f"No overlap {photo_ids[i]} <-> {photo_ids[j]}: "
                    f"{num_matches} matches -> {num_inliers} inliers"
                )

    return overlap_matrix


def split_cluster_by_overlap(
    images: List[Image.Image],
    photo_ids: List[int],
    overlap_matrix: np.ndarray,
) -> List[List[int]]:
    """Split a DINOv2 cluster into sub-clusters by geometric overlap.

    Uses single-linkage to build chains of overlapping photos.

    Args:
        images: List of images in the cluster
        photo_ids: List of photo IDs
        overlap_matrix: NxN overlap matrix

    Returns:
        List of photo ID lists (sub-clusters with verified overlap)
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    n = len(photo_ids)
    if n <= 1:
        return [photo_ids]

    # Check if there's ANY significant overlap
    has_overlap = (overlap_matrix > OVERLAP_THRESHOLD).sum() > n  # More than diagonal

    if not has_overlap:
        # No overlaps found - each photo is its own cluster
        logger.debug(f"No geometric overlap found in cluster of {n} photos")
        return [[pid] for pid in photo_ids]

    # Convert to distance matrix
    distance_matrix = 1.0 - np.clip(overlap_matrix, 0, 1)
    np.fill_diagonal(distance_matrix, 0)

    # Single-linkage clustering (builds chains)
    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method='single')

    # Cut at threshold
    distance_threshold = 1.0 - OVERLAP_THRESHOLD
    labels = fcluster(Z, t=distance_threshold, criterion='distance')

    # Group by label
    clusters_dict = defaultdict(list)
    for i, label in enumerate(labels):
        clusters_dict[label].append(photo_ids[i])

    sub_clusters = list(clusters_dict.values())

    logger.info(f"Overlap clustering: {n} photos -> {len(sub_clusters)} sub-clusters")

    return sub_clusters


# ============================================================================
# GRAPH-BASED CLUSTERING: DINOv2 proposes edges, geometry verifies
# ============================================================================


def _connected_nodes(edge_mask: np.ndarray, start_idx: int) -> set:
    """Return connected node indices from start_idx using DFS over boolean edge mask."""
    stack = [start_idx]
    visited = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        neighbors = np.where(edge_mask[node])[0]
        for nbr in neighbors:
            if nbr not in visited:
                stack.append(int(nbr))
    return visited


def prune_weak_semantic_bridges(
    adjacency: np.ndarray,
    similarity: np.ndarray,
    edge_has_geometry: np.ndarray,
    photo_ids: List[int],
) -> None:
    """Prune semantic-only bridge edges that connect weakly related subgraphs.

    This reduces false merges caused by transitive chaining of weak temporal
    semantic edges (A-B and B-C) when A-C semantic support is weak.
    """
    n = adjacency.shape[0]
    edge_mask = adjacency > OVERLAP_THRESHOLD

    for i in range(n):
        for j in range(i + 1, n):
            if not edge_mask[i, j]:
                continue
            if edge_has_geometry[i, j]:
                continue

            component = _connected_nodes(edge_mask, i)
            if j not in component or len(component) < 3:
                continue

            # Temporarily remove candidate bridge and check whether it truly
            # splits the component.
            edge_mask[i, j] = False
            edge_mask[j, i] = False

            comp_i = _connected_nodes(edge_mask, i)
            if j in comp_i:
                edge_mask[i, j] = True
                edge_mask[j, i] = True
                continue
            comp_j = _connected_nodes(edge_mask, j)

            max_cross_sem = max(similarity[a, b] for a in comp_i for b in comp_j)
            if max_cross_sem < SEMANTIC_BRIDGE_SUPPORT_THRESHOLD:
                adjacency[i, j] = 0.0
                adjacency[j, i] = 0.0
                logger.info(
                    "Pruned semantic bridge %s <-> %s (max cross semantic=%.3f < %.2f)",
                    photo_ids[i],
                    photo_ids[j],
                    max_cross_sem,
                    SEMANTIC_BRIDGE_SUPPORT_THRESHOLD,
                )
                # keep edge removed in edge_mask
            else:
                edge_mask[i, j] = True
                edge_mask[j, i] = True


def build_component_edge_mask(
    adjacency: np.ndarray,
    edge_has_geometry: np.ndarray,
    similarity_records: List[Dict[str, object]],
    photo_ids: List[int],
    room_labels: Optional[List[str]] = None,
) -> np.ndarray:
    """Build component connectivity mask with stricter semantic-only gates."""
    if GEOMETRY_ONLY_CLUSTER_MEMBERSHIP:
        geometry_edges = edge_has_geometry.copy()
        logger.info(
            "Component connectivity uses geometry-only edges (%s undirected edges)",
            int(np.count_nonzero(np.triu(geometry_edges, k=1))),
        )
        return geometry_edges

    edge_mask = adjacency > OVERLAP_THRESHOLD
    if not STRICT_SEMANTIC_COMPONENT_CONNECTIVITY:
        return edge_mask

    pair_lookup = _build_similarity_lookup(similarity_records)
    removed = 0
    kept_semantic = 0

    for i in range(adjacency.shape[0]):
        for j in range(i + 1, adjacency.shape[0]):
            if not edge_mask[i, j]:
                continue
            if edge_has_geometry[i, j]:
                continue

            sem_score = float(adjacency[i, j])
            seq_gap = abs(j - i)
            pid_a = int(photo_ids[min(i, j)])
            pid_b = int(photo_ids[max(i, j)])
            record = pair_lookup.get((pid_a, pid_b), {})
            pair_source = str(record.get("pair_source") or "")

            room_i = room_labels[i] if room_labels and i < len(room_labels) else None
            room_j = room_labels[j] if room_labels and j < len(room_labels) else None
            room_i_norm = normalize_room_label(room_i)
            room_j_norm = normalize_room_label(room_j)
            known_i = room_i_norm not in {"", "unknown"}
            known_j = room_j_norm not in {"", "unknown"}
            same_known_label = known_i and known_j and room_i_norm == room_j_norm
            different_known_label = known_i and known_j and room_i_norm != room_j_norm
            ambiguous_same_label = same_known_label and room_i_norm in {"bathroom"}
            both_front_exterior = (
                room_family(room_i) == "front_exterior"
                and room_family(room_j) == "front_exterior"
            )

            adj_min = COMPONENT_SEMANTIC_ADJ_MIN
            dist2_min = COMPONENT_SEMANTIC_DIST2_MIN
            if same_known_label:
                if ambiguous_same_label:
                    adj_min = max(adj_min, COMPONENT_AMBIGUOUS_SAME_LABEL_MIN)
                    dist2_min = max(dist2_min, COMPONENT_AMBIGUOUS_SAME_LABEL_MIN)
                else:
                    adj_min = min(adj_min, COMPONENT_SAME_LABEL_ADJ_MIN)
                    dist2_min = min(dist2_min, COMPONENT_SAME_LABEL_DIST2_MIN)
            elif different_known_label:
                adj_min = max(adj_min, COMPONENT_CROSS_LABEL_ADJ_MIN)
                dist2_min = max(dist2_min, COMPONENT_CROSS_LABEL_DIST2_MIN)

            keep_edge = False
            reason = ""
            if "semantic_recovery" in pair_source:
                if both_front_exterior and sem_score >= COMPONENT_FRONT_RECOVERY_MIN:
                    keep_edge = True
                    reason = "front_recovery"
                elif seq_gap <= COMPONENT_SEMANTIC_MAX_GAP and sem_score >= COMPONENT_SEMANTIC_RECOVERY_MIN:
                    keep_edge = True
                    reason = "semantic_recovery"
            elif seq_gap <= 1 and sem_score >= adj_min:
                keep_edge = True
                reason = "adjacent_semantic"
            elif seq_gap <= COMPONENT_SEMANTIC_MAX_GAP and sem_score >= dist2_min:
                keep_edge = True
                reason = "dist2_semantic"

            if keep_edge:
                kept_semantic += 1
                logger.debug(
                    "Component semantic edge kept %s <-> %s (score=%.3f, gap=%s, source=%s, reason=%s)",
                    photo_ids[i],
                    photo_ids[j],
                    sem_score,
                    seq_gap,
                    pair_source,
                    reason,
                )
                continue

            edge_mask[i, j] = False
            edge_mask[j, i] = False
            removed += 1
            logger.info(
                "Component semantic edge pruned %s <-> %s (score=%.3f, gap=%s, source=%s)",
                photo_ids[i],
                photo_ids[j],
                sem_score,
                seq_gap,
                pair_source,
            )

    logger.info(
        "Component semantic safety: kept=%s, pruned=%s (adj>=%.2f, dist2>=%.2f, same_adj>=%.2f, same_dist2>=%.2f, cross_adj>=%.2f, cross_dist2>=%.2f, amb_same>=%.2f, recovery>=%.2f, front>=%.2f, max_gap=%s)",
        kept_semantic,
        removed,
        COMPONENT_SEMANTIC_ADJ_MIN,
        COMPONENT_SEMANTIC_DIST2_MIN,
        COMPONENT_SAME_LABEL_ADJ_MIN,
        COMPONENT_SAME_LABEL_DIST2_MIN,
        COMPONENT_CROSS_LABEL_ADJ_MIN,
        COMPONENT_CROSS_LABEL_DIST2_MIN,
        COMPONENT_AMBIGUOUS_SAME_LABEL_MIN,
        COMPONENT_SEMANTIC_RECOVERY_MIN,
        COMPONENT_FRONT_RECOVERY_MIN,
        COMPONENT_SEMANTIC_MAX_GAP,
    )
    return edge_mask

def cluster_photos_graph_based(
    images: List[Image.Image],
    photo_ids: List[int],
    k: int = 8,
    max_cluster_size: int = 6,
    db_session=None,
    job_id: int = None,
    room_labels: List[str] = None,
    return_metadata: bool = False,
) -> List[List[int]] | Tuple[List[List[int]], Dict[str, object]]:
    """Graph-based clustering: semantic proposals + geometric verification.

    This is a cleaner architecture than semantic-first clustering:
    1. DINOv2 embeddings propose candidate edges (top-K similar pairs)
    2. Geometric verification on proposed edges only
    3. Connected components = final clusters

    Key insight: Semantic similarity is used as a FILTER (edge proposal),
    not as ground truth. This eliminates the need for cross-cluster merging.

    Complexity: O(N × K) geometric checks instead of O(N²)

    Args:
        images: List of PIL Images
        photo_ids: List of photo IDs
        k: Number of candidate neighbors per image (default 8)
        max_cluster_size: Maximum photos per cluster
        room_labels: Optional list of room labels for each photo (used to penalize cross-room connections)

    Returns:
        Final clusters, or (clusters, metadata) when return_metadata=True.
    """
    from sklearn.preprocessing import normalize
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(images)
    logger.info(f"Graph-based clustering for {n} photos (k={k})")

    # Track similarity data for database storage
    similarity_records = []  # Will store dicts for batch insert

    if n <= 1:
        clusters = [photo_ids] if photo_ids else []
        if return_metadata:
            return clusters, {"duplicate_of_map": {}}
        return clusters

    # -------------------------------------------------------------------------
    # Stage 1: Compute DINOv2 embeddings and build candidate graph
    # -------------------------------------------------------------------------
    logger.info("Stage 1: Computing DINOv2 embeddings...")
    embeddings = compute_dinov2_embeddings(images)
    embeddings = normalize(embeddings)  # For cosine similarity

    # Compute similarity matrix
    similarity = embeddings @ embeddings.T

    # -------------------------------------------------------------------------
    # Stage 1b: Build candidate pairs from DINOv2 top-K
    # -------------------------------------------------------------------------
    logger.info(f"Stage 1b: Building candidate graph (top-{k} per image)...")
    semantic_pairs = set()
    for i in range(n):
        sorted_indices = np.argsort(-similarity[i])
        neighbors = [j for j in sorted_indices if j != i][:k]
        for j in neighbors:
            pair = (min(i, j), max(i, j))
            semantic_pairs.add(pair)
    logger.info(f"  DINOv2 top-{k} proposed {len(semantic_pairs)} semantic pairs")

    # -------------------------------------------------------------------------
    # Stage 1c: Add temporal window pairs (adjacent photos likely same room)
    # -------------------------------------------------------------------------
    logger.info(f"Stage 1c: Adding temporal window pairs (window=±{TEMPORAL_WINDOW})...")
    temporal_pairs = set()
    for i in range(n):
        for offset in range(1, TEMPORAL_WINDOW + 1):
            if i + offset < n:
                temporal_pairs.add((i, i + offset))
    logger.info(f"  Added {len(temporal_pairs)} temporal pairs")

    # -------------------------------------------------------------------------
    # Stage 2a: Temporal pairs - require geometric verification unless very high semantic
    # -------------------------------------------------------------------------
    logger.info("Stage 2a: Checking temporal pairs...")
    adjacency = np.zeros((n, n))
    edge_has_geometry = np.zeros((n, n), dtype=bool)
    directions = {}  # (i, j) -> (dx, dy) direction vectors for ordering
    temporal_matched = 0
    temporal_semantic_only = 0
    temporal_geometric = 0

    for i, j in sorted(temporal_pairs):
        sem_sim = similarity[i, j]
        temporal_dist = abs(j - i)

        # Check if rooms are different
        room_i = room_labels[i] if room_labels else None
        room_j = room_labels[j] if room_labels else None
        is_cross_room = rooms_are_different(room_i, room_j)

        # For NON-ADJACENT cross-room pairs (temporal_dist > 1), skip entirely
        # But for ADJACENT photos (temporal_dist = 1), DO geometric verification
        # because ML room labels are often wrong for adjacent photos
        if is_cross_room and temporal_dist > 1:
            logger.info(f"  Temporal {photo_ids[i]} <-> {photo_ids[j]}: "
                       f"SKIP - different rooms ({room_i} vs {room_j}), dist={temporal_dist}")
            similarity_records.append({
                "photo_a_id": photo_ids[min(i, j)],
                "photo_b_id": photo_ids[max(i, j)],
                "pair_source": "temporal_cross_room",
                "dinov2_similarity": float(sem_sim),
                "geometric_matches": None,
                "geometric_inliers": None,
                "geometric_score": None,
                "direction_dx": None,
                "direction_dy": None,
                "is_connected": 0,
                **_build_pair_metrics_payload(None),
            })
            continue

        room_info = f" [cross-room: {room_i}/{room_j}]" if is_cross_room else ""
        logger.info(f"  Temporal {photo_ids[i]} <-> {photo_ids[j]} (dist={temporal_dist}): "
                   f"semantic={sem_sim:.3f}{room_info}")

        is_matched = False
        num_matches = None
        num_inliers = None
        geo_score = None
        direction = (0.0, 0.0)
        diagnostics: Dict[str, Any] | None = None

        # Cross-room pairs need geometric verification with higher threshold
        # But for ADJACENT cross-room (temporal_dist=1), use lower threshold since ML often mislabels
        if is_cross_room:
            min_inliers_required = cross_room_min_inliers_required(
                temporal_dist=temporal_dist,
                sem_sim=float(sem_sim),
            )
        else:
            min_inliers_required = MIN_INLIERS_FOR_OVERLAP
        effective_min_inliers_required = max(int(min_inliers_required), int(NATIVE_EDGE_MIN_INLIERS))

        # Very high semantic similarity:
        # - geometry-only mode: still require strict geometric gate
        # - legacy mode: allow semantic-first trust with geometric refinement
        if sem_sim >= TEMPORAL_SEMANTIC_THRESHOLD and not is_cross_room:
            # Still run geometric to get direction for ordering
            num_matches, num_inliers, geo_score, direction, diagnostics = match_image_pair(
                images[i],
                images[j],
                return_diagnostics=True,
            )

            blended_score = None
            if strict_geometry_edge_gate(
                num_matches=num_matches,
                num_inliers=num_inliers,
                geometric_score=geo_score,
                diagnostics=diagnostics,
                min_inliers_required=effective_min_inliers_required,
            ):
                blended_score = blend_geometric_semantic_score(float(geo_score), float(sem_sim))
                edge_has_geometry[i, j] = True
                edge_has_geometry[j, i] = True
                adjacency[i, j] = blended_score
                adjacency[j, i] = blended_score
                temporal_matched += 1
                temporal_geometric += 1
                is_matched = True
            elif not GEOMETRY_ONLY_CLUSTER_MEMBERSHIP:
                blended_score = float(sem_sim)
                adjacency[i, j] = blended_score
                adjacency[j, i] = blended_score
                temporal_matched += 1
                temporal_semantic_only += 1
                is_matched = True

            if is_matched:
                if num_inliers >= MIN_INLIERS_FOR_DIRECTION and direction != (0.0, 0.0):
                    directions[(i, j)] = direction
                    dir_str = f"dir=({direction[0]:.2f},{direction[1]:.2f})"
                    logger.info(
                        "    ✓ MATCHED (semantic>=%.2f, sem=%.3f, geo=%s, blended=%.3f, %s from %s inliers)",
                        TEMPORAL_SEMANTIC_THRESHOLD,
                        sem_sim,
                        f"{geo_score:.3f}" if geo_score is not None else "n/a",
                        blended_score if blended_score is not None else float(sem_sim),
                        dir_str,
                        num_inliers,
                    )
                else:
                    logger.info(
                        "    ✓ MATCHED (semantic>=%.2f, sem=%.3f, geo=%s, blended=%.3f, no direction - %s inliers)",
                        TEMPORAL_SEMANTIC_THRESHOLD,
                        sem_sim,
                        f"{geo_score:.3f}" if geo_score is not None else "n/a",
                        blended_score if blended_score is not None else float(sem_sim),
                        num_inliers,
                    )
            else:
                logger.info(
                    "    ✗ Rejected by strict gate (semantic-high but no geometry edge: matches=%s, inliers=%s, geo=%s)",
                    num_matches,
                    num_inliers,
                    f"{geo_score:.3f}" if geo_score is not None else "n/a",
                )

        # Moderate semantic OR cross-room - require geometric verification
        elif sem_sim >= TEMPORAL_GEOMETRIC_THRESHOLD or is_cross_room:
            logger.info(
                "    Verifying geometrically (strict gate: inliers>=%s, matches>=%s, native_mean>=%.2f, native_median>=%.2f, geom_model in %s)...",
                effective_min_inliers_required,
                NATIVE_EDGE_MIN_MATCHES,
                NATIVE_EDGE_MIN_MEAN,
                NATIVE_EDGE_MIN_MEDIAN,
                sorted(NATIVE_EDGE_ALLOWED_GEOMETRY_MODELS),
            )
            num_matches, num_inliers, geo_score, direction, diagnostics = match_image_pair(
                images[i],
                images[j],
                return_diagnostics=True,
            )

            if strict_geometry_edge_gate(
                num_matches=num_matches,
                num_inliers=num_inliers,
                geometric_score=geo_score,
                diagnostics=diagnostics,
                min_inliers_required=effective_min_inliers_required,
            ):
                blended_score = blend_geometric_semantic_score(float(geo_score), float(sem_sim))
                adjacency[i, j] = blended_score
                adjacency[j, i] = blended_score
                edge_has_geometry[i, j] = True
                edge_has_geometry[j, i] = True
                directions[(i, j)] = direction
                temporal_matched += 1
                temporal_geometric += 1
                is_matched = True
                dir_str = f"dir=({direction[0]:.2f},{direction[1]:.2f})" if direction != (0.0, 0.0) else "dir=unknown"
                logger.info(
                    "    ✓ MATCHED (geometric: %s inliers >= %s, geo=%.3f, sem=%.3f, blended=%.3f, %s)",
                    num_inliers,
                    effective_min_inliers_required,
                    geo_score,
                    sem_sim,
                    blended_score,
                    dir_str,
                )
            # Fallback: semantic-only trust (disabled in geometry-only mode)
            elif (not GEOMETRY_ONLY_CLUSTER_MEMBERSHIP) and temporal_dist == 1 and sem_sim >= NEIGHBOR_TRUST_THRESHOLD and not is_cross_room:
                room_i_norm = normalize_room_label(room_i)
                room_j_norm = normalize_room_label(room_j)
                same_room_label = (
                    room_i_norm != ""
                    and room_i_norm != "unknown"
                    and room_i_norm == room_j_norm
                )
                ambiguous_rooms = {"bathroom"}
                same_label_threshold = (
                    AMBIGUOUS_SAME_LABEL_TRUST
                    if room_i_norm in ambiguous_rooms
                    else SAME_LABEL_NEIGHBOR_TRUST
                )
                same_label_trust = same_room_label and sem_sim >= same_label_threshold
                local_support = has_local_semantic_support(i, j, similarity)
                allow_semantic_neighbor = (
                    sem_sim >= HIGH_CONFIDENCE_NEIGHBOR_TRUST
                    or local_support
                    or same_label_trust
                )
                if not allow_semantic_neighbor:
                    logger.info(
                        "    ✗ No neighbor semantic trust (sem=%.3f, support=%s, same_label=%s)",
                        sem_sim,
                        "yes" if local_support else "no",
                        "yes" if same_label_trust else "no",
                    )
                else:
                    adjacency[i, j] = sem_sim
                    adjacency[j, i] = sem_sim
                    temporal_matched += 1
                    temporal_semantic_only += 1
                    is_matched = True

                    if num_inliers >= MIN_INLIERS_FOR_DIRECTION and direction != (0.0, 0.0):
                        directions[(i, j)] = direction
                        dir_str = f"dir=({direction[0]:.2f},{direction[1]:.2f})"
                        logger.info(
                            "    ✓ MATCHED (neighbor trust: semantic %.3f, support=%s, same_label=%s, %s from %s inliers)",
                            sem_sim,
                            "yes" if local_support else "no",
                            "yes" if same_label_trust else ("high-confidence" if sem_sim >= HIGH_CONFIDENCE_NEIGHBOR_TRUST else "no"),
                            dir_str,
                            num_inliers,
                        )
                    else:
                        logger.info(
                            "    ✓ MATCHED (neighbor trust: semantic %.3f, support=%s, same_label=%s, no direction)",
                            sem_sim,
                            "yes" if local_support else "no",
                            "yes" if same_label_trust else ("high-confidence" if sem_sim >= HIGH_CONFIDENCE_NEIGHBOR_TRUST else "no"),
                        )
            # Cross-room adjacent semantic recovery (disabled in geometry-only mode)
            elif (not GEOMETRY_ONLY_CLUSTER_MEMBERSHIP) and temporal_dist == 1 and is_cross_room:
                compatible_families = rooms_allow_adjacent_semantic_bridge(room_i, room_j)
                local_support = has_local_semantic_support(i, j, similarity)
                mutual_top = is_mutual_top_semantic_neighbor(i, j, similarity, top_k=CROSS_ROOM_RECOVERY_TOPK)
                recovery_threshold = (
                    SERVICE_ROOM_RECOVERY_THRESHOLD
                    if room_family(room_i) == "service" and room_family(room_j) == "service"
                    else CROSS_ROOM_RECOVERY_THRESHOLD
                )
                if (
                    compatible_families
                    and sem_sim >= recovery_threshold
                    and (mutual_top or local_support)
                ):
                    adjacency[i, j] = sem_sim
                    adjacency[j, i] = sem_sim
                    temporal_matched += 1
                    temporal_semantic_only += 1
                    is_matched = True
                    logger.info(
                        "    ✓ MATCHED (cross-room recovery: sem=%.3f >= %.2f, mutual_top=%s, support=%s, %s/%s)",
                        sem_sim,
                        recovery_threshold,
                        "yes" if mutual_top else "no",
                        "yes" if local_support else "no",
                        room_i,
                        room_j,
                    )
                else:
                    logger.info(
                        "    ✗ Cross-room recovery rejected (sem=%.3f, threshold=%.2f, mutual_top=%s, support=%s, compatible=%s)",
                        sem_sim,
                        recovery_threshold,
                        "yes" if mutual_top else "no",
                        "yes" if local_support else "no",
                        "yes" if compatible_families else "no",
                    )
            # Distance-2 semantic fallback (disabled in geometry-only mode).
            elif (not GEOMETRY_ONLY_CLUSTER_MEMBERSHIP) and temporal_dist == 2 and not is_cross_room:
                room_i_norm = normalize_room_label(room_i)
                room_j_norm = normalize_room_label(room_j)
                same_room_label = (
                    room_i_norm != ""
                    and room_i_norm != "unknown"
                    and room_i_norm == room_j_norm
                )

                dist2_threshold = DIST2_TRUST_THRESHOLD
                if same_room_label:
                    if room_i_norm in {"bathroom"}:
                        dist2_threshold = AMBIGUOUS_DIST2_TRUST
                    else:
                        dist2_threshold = SAME_LABEL_DIST2_TRUST

                if sem_sim >= dist2_threshold:
                    adjacency[i, j] = sem_sim
                    adjacency[j, i] = sem_sim
                    temporal_matched += 1
                    temporal_semantic_only += 1
                    is_matched = True

                    if num_inliers >= MIN_INLIERS_FOR_DIRECTION and direction != (0.0, 0.0):
                        directions[(i, j)] = direction
                        logger.info(
                            "    ✓ MATCHED (dist2 trust: semantic %.3f >= %.2f, "
                            "same_label=%s, dir=(%.2f,%.2f) from %s inliers)",
                            sem_sim,
                            dist2_threshold,
                            "yes" if same_room_label else "no",
                            direction[0],
                            direction[1],
                            num_inliers,
                        )
                    else:
                        logger.info(
                            "    ✓ MATCHED (dist2 trust: semantic %.3f >= %.2f, same_label=%s, no direction)",
                            sem_sim,
                            dist2_threshold,
                            "yes" if same_room_label else "no",
                        )
                else:
                    logger.info(
                        "    ✗ Dist2 semantic below trust threshold (%.3f < %.2f)",
                        sem_sim,
                        dist2_threshold,
                    )
            else:
                cross_note = " [cross-room]" if is_cross_room else ""
                if num_inliers >= effective_min_inliers_required:
                    logger.info(
                        "    ✗ Rejected by strict gate (inliers=%s, matches=%s, geo=%.3f)%s",
                        num_inliers,
                        num_matches,
                        float(geo_score) if geo_score is not None else 0.0,
                        cross_note,
                    )
                else:
                    logger.info(f"    ✗ No geometric match ({num_inliers} inliers < {effective_min_inliers_required}){cross_note}")
        else:
            logger.info(f"    ✗ No match (semantic {sem_sim:.3f} < {TEMPORAL_GEOMETRIC_THRESHOLD})")

        # Track for database storage
        pair_source = "both" if (i, j) in semantic_pairs else "temporal_window"
        pair_source = _annotate_pair_source_with_oracle(pair_source, diagnostics)
        similarity_records.append({
            "photo_a_id": photo_ids[min(i, j)],
            "photo_b_id": photo_ids[max(i, j)],
            "pair_source": pair_source,
            "dinov2_similarity": float(sem_sim),
            "geometric_matches": num_matches,
            "geometric_inliers": num_inliers,
            "geometric_score": float(geo_score) if geo_score else None,
            "direction_dx": float(direction[0]) if direction != (0.0, 0.0) else None,
            "direction_dy": float(direction[1]) if direction != (0.0, 0.0) else None,
            "is_connected": 1 if is_matched else 0,
            **_build_pair_metrics_payload(diagnostics),
        })

    logger.info(f"Stage 2a: {temporal_matched}/{len(temporal_pairs)} temporal pairs matched "
               f"(semantic-only={temporal_semantic_only}, geometric={temporal_geometric})")

    # -------------------------------------------------------------------------
    # Stage 2b: Non-temporal pairs - use GEOMETRIC verification (pixel overlap)
    # -------------------------------------------------------------------------
    # Only check semantic pairs that aren't already covered by temporal
    geometric_pairs = semantic_pairs - temporal_pairs
    logger.info(f"Stage 2b: Geometric verification on {len(geometric_pairs)} non-temporal pairs...")
    geometric_matched = 0

    skipped_low_semantic = 0
    skipped_cross_room = 0
    semantic_recovered = 0
    for idx, (i, j) in enumerate(sorted(geometric_pairs)):
        sem_sim = similarity[i, j]

        # Check if rooms are different - SKIP cross-room pairs entirely
        room_i = room_labels[i] if room_labels else None
        room_j = room_labels[j] if room_labels else None
        is_cross_room = rooms_are_different(room_i, room_j)

        if is_cross_room:
            logger.info(f"  [{idx+1}/{len(geometric_pairs)}] SKIP {photo_ids[i]} <-> {photo_ids[j]} "
                       f"(different rooms: {room_i} vs {room_j})")
            skipped_cross_room += 1
            similarity_records.append({
                "photo_a_id": photo_ids[min(i, j)],
                "photo_b_id": photo_ids[max(i, j)],
                "pair_source": "dinov2_topk_cross_room",
                "dinov2_similarity": float(sem_sim),
                "geometric_matches": None,
                "geometric_inliers": None,
                "geometric_score": None,
                "direction_dx": None,
                "direction_dy": None,
                "is_connected": 0,
                **_build_pair_metrics_payload(None),
            })
            continue

        # Skip pairs with very low semantic similarity (obviously unrelated)
        if sem_sim < MIN_SEMANTIC_FOR_GEOMETRIC:
            logger.info(f"  [{idx+1}/{len(geometric_pairs)}] SKIP {photo_ids[i]} <-> {photo_ids[j]} "
                       f"(DINOv2 similarity={sem_sim:.3f} < {MIN_SEMANTIC_FOR_GEOMETRIC})")
            skipped_low_semantic += 1
            # Still record for database but mark as not matched
            similarity_records.append({
                "photo_a_id": photo_ids[min(i, j)],
                "photo_b_id": photo_ids[max(i, j)],
                "pair_source": "dinov2_topk_skipped",
                "dinov2_similarity": float(sem_sim),
                "geometric_matches": None,
                "geometric_inliers": None,
                "geometric_score": None,
                "direction_dx": None,
                "direction_dy": None,
                "is_connected": 0,
                **_build_pair_metrics_payload(None),
            })
            continue

        # Photos far apart in sequence need stronger evidence to be connected
        # (prevents clustering photos from different physical locations with same room label)
        position_gap = abs(j - i)
        if position_gap >= VERY_FAR_POSITION_GAP_THRESHOLD:
            min_inliers = MIN_INLIERS_VERY_FAR
            gap_note = f", gap={position_gap} -> need {min_inliers} inliers + score>={MIN_SCORE_VERY_FAR:.2f}"
            if not accepts_very_far_pair(position_gap, float(sem_sim), min_inliers, MIN_SCORE_VERY_FAR):
                logger.info(
                    f"  [{idx+1}/{len(geometric_pairs)}] SKIP {photo_ids[i]} <-> {photo_ids[j]} "
                    f"(very-far gap={position_gap}, semantic={sem_sim:.3f} < {MIN_SEMANTIC_FOR_VERY_FAR})"
                )
                skipped_low_semantic += 1
                similarity_records.append({
                    "photo_a_id": photo_ids[min(i, j)],
                    "photo_b_id": photo_ids[max(i, j)],
                    "pair_source": "dinov2_topk_skipped",
                    "dinov2_similarity": float(sem_sim),
                    "geometric_matches": None,
                    "geometric_inliers": None,
                    "geometric_score": None,
                    "direction_dx": None,
                    "direction_dy": None,
                    "is_connected": 0,
                    **_build_pair_metrics_payload(None),
                })
                continue
        elif position_gap >= POSITION_GAP_THRESHOLD:
            min_inliers = MIN_INLIERS_FAR_APART
            gap_note = f", gap={position_gap} -> need {min_inliers} inliers"
        else:
            min_inliers = MIN_INLIERS_FOR_OVERLAP
            gap_note = ""
        effective_min_inliers = max(int(min_inliers), int(NATIVE_EDGE_MIN_INLIERS))

        logger.info(f"  [{idx+1}/{len(geometric_pairs)}] Checking {photo_ids[i]} <-> {photo_ids[j]} "
                   f"(DINOv2 similarity={sem_sim:.3f}{gap_note})...")
        logger.info(
            "    Strict gate: inliers>=%s, matches>=%s, native_mean>=%.2f, native_median>=%.2f, geom_model in %s",
            effective_min_inliers,
            NATIVE_EDGE_MIN_MATCHES,
            NATIVE_EDGE_MIN_MEAN,
            NATIVE_EDGE_MIN_MEDIAN,
            sorted(NATIVE_EDGE_ALLOWED_GEOMETRY_MODELS),
        )

        diagnostics: Dict[str, Any] | None = None
        num_matches, num_inliers, score, direction, diagnostics = match_image_pair(
            images[i],
            images[j],
            return_diagnostics=True,
        )

        is_matched = strict_geometry_edge_gate(
            num_matches=num_matches,
            num_inliers=num_inliers,
            geometric_score=score,
            diagnostics=diagnostics,
            min_inliers_required=effective_min_inliers,
        )
        if is_matched and not accepts_very_far_pair(position_gap, float(sem_sim), int(num_inliers), float(score)):
            is_matched = False
            logger.info(
                "    ✗ Rejected very-far geometric link (gap=%s, sem=%.3f, inliers=%s, score=%.3f)",
                position_gap,
                sem_sim,
                num_inliers,
                score,
            )
        pair_source = "dinov2_topk"
        if is_matched:
            blended_score = blend_geometric_semantic_score(float(score), float(sem_sim))
            adjacency[i, j] = max(adjacency[i, j], blended_score)  # Keep higher score
            adjacency[j, i] = max(adjacency[j, i], blended_score)
            edge_has_geometry[i, j] = True
            edge_has_geometry[j, i] = True
            directions[(i, j)] = direction  # Store direction for ordering
            geometric_matched += 1
            dir_str = f"dir=({direction[0]:.2f},{direction[1]:.2f})" if direction != (0.0, 0.0) else "dir=unknown"
            logger.info(
                "    ✓ MATCHED: %s matches, %s inliers, geo=%.3f, sem=%.3f, blended=%.3f, %s",
                num_matches,
                num_inliers,
                score,
                sem_sim,
                blended_score,
                dir_str,
            )
        else:
            same_room_label = (
                normalize_room_label(room_i) != ""
                and normalize_room_label(room_i) != "unknown"
                and normalize_room_label(room_i) == normalize_room_label(room_j)
            )
            exterior_pair = is_exterior_like(room_i) and is_exterior_like(room_j)
            if (
                (not GEOMETRY_ONLY_CLUSTER_MEMBERSHIP)
                and same_room_label
                and exterior_pair
                and position_gap >= EXTERIOR_LONG_GAP_MIN
                and sem_sim >= EXTERIOR_LONG_GAP_SEMANTIC_TRUST
            ):
                adjacency[i, j] = max(adjacency[i, j], sem_sim)
                adjacency[j, i] = max(adjacency[j, i], sem_sim)
                geometric_matched += 1
                semantic_recovered += 1
                is_matched = True
                pair_source = "dinov2_topk_semantic_recovery"
                logger.info(
                    "    ✓ MATCHED (exterior long-gap semantic recovery: sem=%.3f, gap=%s, rooms=%s/%s)",
                    sem_sim,
                    position_gap,
                    room_i,
                    room_j,
                )
            else:
                if num_inliers >= effective_min_inliers:
                    logger.info(
                        "    ✗ Rejected by strict gate (matches=%s, inliers=%s, geo=%.3f)",
                        num_matches,
                        num_inliers,
                        float(score),
                    )
                else:
                    logger.info(f"    ✗ No match: {num_matches} matches, {num_inliers} inliers < {effective_min_inliers}")

        # Track for database storage
        similarity_records.append({
            "photo_a_id": photo_ids[min(i, j)],
            "photo_b_id": photo_ids[max(i, j)],
            "pair_source": _annotate_pair_source_with_oracle(pair_source, diagnostics),
            "dinov2_similarity": float(sem_sim),
            "geometric_matches": num_matches,
            "geometric_inliers": num_inliers,
            "geometric_score": float(score) if score else None,
            "direction_dx": float(direction[0]) if direction != (0.0, 0.0) else None,
            "direction_dy": float(direction[1]) if direction != (0.0, 0.0) else None,
            "is_connected": 1 if is_matched else 0,
            **_build_pair_metrics_payload(diagnostics),
        })

    logger.info(
        "Stage 2b: %s/%s non-temporal pairs matched (%s cross-room, %s low-semantic, %s semantic recoveries)",
        geometric_matched,
        len(geometric_pairs),
        skipped_cross_room,
        skipped_low_semantic,
        semantic_recovered,
    )
    logger.info(f"Total edges: {temporal_matched + geometric_matched} "
               f"(temporal={temporal_matched}, geometric={geometric_matched})")

    # Prune weak semantic-only bridge edges that can cause transitive false merges.
    prune_weak_semantic_bridges(adjacency, similarity, edge_has_geometry, photo_ids)

    # -------------------------------------------------------------------------
    # Stage 3: Connected components = final clusters
    # -------------------------------------------------------------------------
    logger.info("Stage 3: Finding connected components...")

    component_edge_mask = build_component_edge_mask(
        adjacency=adjacency,
        edge_has_geometry=edge_has_geometry,
        similarity_records=similarity_records,
        photo_ids=photo_ids,
        room_labels=room_labels,
    )
    sparse_adj = csr_matrix(component_edge_mask)
    n_components, labels = connected_components(sparse_adj, directed=False)

    # Group photo INDICES by component (not photo_ids yet)
    clusters_by_label = defaultdict(list)
    for i, label in enumerate(labels):
        clusters_by_label[label].append(i)

    # -------------------------------------------------------------------------
    # Stage 4: Order photos within each cluster for smooth transitions
    # -------------------------------------------------------------------------
    logger.info("Stage 4: Ordering photos within clusters for transitions...")

    ordered_clusters = []
    for label, cluster_indices in clusters_by_label.items():
        if len(cluster_indices) == 1:
            # Single photo - no ordering needed
            ordered_clusters.append([photo_ids[cluster_indices[0]]])
        else:
            # Order using direction-aware algorithm
            # Returns (ordered_main_chain, isolated_photos)
            ordered_main, isolated = order_cluster_for_transitions(
                cluster_indices,
                photo_ids,
                adjacency,
                directions,
                min_score=MIN_TRANSITION_SCORE,
            )
            if ordered_main:
                ordered_clusters.append(ordered_main)
                logger.info(f"  Cluster {label}: ordered {len(cluster_indices)} -> main={len(ordered_main)}")

            # Add isolated photos as singleton clusters (they couldn't be ordered but are connected)
            for iso_pid in isolated:
                ordered_clusters.append([iso_pid])
                logger.info(f"  Cluster {label}: isolated photo {iso_pid} -> own cluster")

    # Stage 4b: Enforce hard transition quality constraints on ordered chains.
    ordered_clusters = enforce_transition_quality(
        ordered_clusters=ordered_clusters,
        photo_ids=photo_ids,
        adjacency=adjacency,
        similarity=similarity,
        edge_has_geometry=edge_has_geometry,
        similarity_records=similarity_records,
        room_labels=room_labels,
    )

    # -------------------------------------------------------------------------
    # Stage 5: Deduplicate and split large clusters
    # -------------------------------------------------------------------------
    logger.info(f"Stage 5: Deduplicating and splitting clusters (max {max_cluster_size} photos each)...")

    final_clusters = []
    duplicate_of_map: Dict[int, int] = {}
    for cluster in ordered_clusters:
        # Deduplicate obvious same-angle shots.
        split_clusters, cluster_duplicate_map = deduplicate_and_split_cluster(
            cluster,
            photo_ids,
            embeddings,
            adjacency,
            max_size=max_cluster_size,
            keep_duplicate_singletons=KEEP_DUPLICATE_SINGLETON_CLUSTERS,
        )
        final_clusters.extend(split_clusters)
        duplicate_of_map.update(cluster_duplicate_map)

    logger.info(
        f"Final result: {n} photos -> {len(final_clusters)} clusters "
        f"(sizes: {[len(c) for c in final_clusters]})"
    )
    if duplicate_of_map:
        if KEEP_DUPLICATE_SINGLETON_CLUSTERS:
            logger.info("Marked %s photos as duplicates (kept as singleton clusters)", len(duplicate_of_map))
        else:
            logger.info("Removed %s obvious duplicates from final clustering output", len(duplicate_of_map))

    # Save similarity records to database if session provided
    if db_session is not None and job_id is not None and similarity_records:
        from sqlalchemy import text

        # Backward-compatible schema detection:
        # some environments may not yet have latest overlap metric columns.
        available_cols: set[str] = set()
        try:
            col_rows = db_session.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'photo_similarities'
                      AND table_schema = current_schema()
                    """
                )
            ).fetchall()
            available_cols = {str(row[0]) for row in col_rows}
        except Exception as schema_err:
            logger.warning("Could not inspect photo_similarities schema: %s", schema_err)

        logger.info(f"Saving {len(similarity_records)} similarity records to database...")

        # Re-running clustering for a job should replace previous similarity rows.
        db_session.execute(
            text("DELETE FROM photo_similarities WHERE job_id = :job_id"),
            {"job_id": job_id},
        )

        base_columns = [
            "job_id",
            "photo_a_id",
            "photo_b_id",
            "pair_source",
            "dinov2_similarity",
            "geometric_matches",
            "geometric_inliers",
            "geometric_score",
            "is_connected",
        ]
        optional_columns = [
            "direction_dx",
            "direction_dy",
            "from_left_25_50",
            "from_right_50_75",
            "to_left_25_50",
            "to_right_50_75",
            "cross_left_to_right",
            "cross_right_to_left",
            "cross_center_to_center",
            "kornia_overlap_ratio",
            "kornia_side_overlap",
            "kornia_center_overlap",
            "kornia_inlier_ratio",
            "kornia_transition_overlap_ok",
        ]
        insert_columns = base_columns + [col for col in optional_columns if col in available_cols]
        column_sql = ", ".join(insert_columns)
        values_sql = ", ".join(f":{col}" for col in insert_columns)
        insert_sql = text(
            f"""
            INSERT INTO photo_similarities ({column_sql})
            VALUES ({values_sql})
            """
        )

        records_for_insert = []
        for record in similarity_records:
            payload = {
                "job_id": job_id,
                "photo_a_id": record["photo_a_id"],
                "photo_b_id": record["photo_b_id"],
                "pair_source": record["pair_source"],
                "dinov2_similarity": record["dinov2_similarity"],
                "geometric_matches": record["geometric_matches"],
                "geometric_inliers": record["geometric_inliers"],
                "geometric_score": record["geometric_score"],
                "is_connected": record["is_connected"],
            }
            for col in insert_columns:
                if col in payload:
                    continue
                payload[col] = record.get(col)
            records_for_insert.append(payload)

        db_session.execute(insert_sql, records_for_insert)
        db_session.commit()
        logger.info(
            "Saved %s similarity records (optional columns saved: %s)",
            len(similarity_records),
            ", ".join([c for c in insert_columns if c not in base_columns]) or "none",
        )

    if return_metadata:
        return final_clusters, {
            "duplicate_of_map": duplicate_of_map,
            "duplicates_dropped": not KEEP_DUPLICATE_SINGLETON_CLUSTERS,
        }
    return final_clusters


# ============================================================================
# MAIN PIPELINE ENTRY POINT
# ============================================================================

def cluster_photos_optimized(
    images: List[Image.Image],
    photo_ids: List[int],
    s3_client=None,
    db_session=None,
    job_id: int = None,
    room_labels: List[str] = None,
    return_metadata: bool = False,
) -> List[List[int]] | Tuple[List[List[int]], Dict[str, object]]:
    """Run optimized graph-based clustering pipeline.

    Uses the "propose + verify" pattern:
    1. DINOv2 embeddings propose candidate edges (top-K similar pairs)
    2. Geometric verification (LoFTR/ORB) confirms overlap
    3. Connected components = final clusters

    This is cleaner than semantic-first clustering because:
    - No artificial semantic boundaries to fix later
    - Semantic similarity is a filter, not ground truth
    - O(N × K) geometric checks instead of O(N²)

    Args:
        images: List of PIL Images
        photo_ids: List of photo IDs
        s3_client: S3 client (unused, for API compatibility)
        db_session: Optional SQLAlchemy session for saving similarity data
        job_id: Optional job ID for saving similarity data
        room_labels: Optional list of room labels for each photo (used to penalize cross-room connections)

    Returns:
        Final clusters, or (clusters, metadata) when return_metadata=True.
    """
    n = len(images)
    # Adaptive semantic neighborhood size:
    # - small jobs: keep checks tight
    # - medium jobs: balanced recall/compute
    # - large jobs: broader proposal set
    if n <= 80:
        candidate_k = 4
    elif n <= 140:
        candidate_k = 5
    else:
        candidate_k = 6

    logger.info("Adaptive candidate k=%s for %s photos", candidate_k, n)

    return cluster_photos_graph_based(
        images,
        photo_ids,
        k=candidate_k,
        max_cluster_size=3,  # Limit to 3 best photos per cluster
        db_session=db_session,
        job_id=job_id,
        room_labels=room_labels,
        return_metadata=return_metadata,
    )
