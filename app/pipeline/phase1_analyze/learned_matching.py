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
import os
import sys
import time
import warnings
from copy import deepcopy
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

# Performance controls for production matcher path.
DINO_BATCH_SIZE = 16


def _preferred_torch_device() -> torch.device:
    """Choose fastest available backend: CUDA -> MPS -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and bool(mps_backend.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def _load_dinov2():
    """Load DINOv2 model (lazy initialization)."""
    global _dinov2_model, _dinov2_transform

    if _dinov2_model is not None:
        return _dinov2_model, _dinov2_transform

    try:
        # Use transformers for DINOv2
        from transformers import AutoImageProcessor, AutoModel

        model_name = "facebook/dinov2-base"
        cache_dir = settings.MODEL_CACHE_DIR
        processor_kwargs: Dict[str, Any] = {
            "cache_dir": cache_dir,
            # Keep preprocessing deterministic across transformers versions.
            "use_fast": False,
        }
        model_kwargs: Dict[str, Any] = {"cache_dir": cache_dir}
        load_source = "local-cache"
        try:
            _dinov2_transform = AutoImageProcessor.from_pretrained(
                model_name,
                local_files_only=True,
                **processor_kwargs,
            )
            _dinov2_model = AutoModel.from_pretrained(
                model_name,
                local_files_only=True,
                **model_kwargs,
            )
        except Exception:
            load_source = "hf-hub"
            _dinov2_transform = AutoImageProcessor.from_pretrained(
                model_name,
                **processor_kwargs,
            )
            _dinov2_model = AutoModel.from_pretrained(
                model_name,
                **model_kwargs,
            )

        device = _preferred_torch_device()
        _dinov2_model = _dinov2_model.to(device)
        _dinov2_model.eval()

        logger.info("Loaded DINOv2 model on %s (source=%s, use_fast=False)", device, load_source)
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
    fell_back_from_mps = False
    embeddings: List[np.ndarray] = []
    batch_size = max(1, int(DINO_BATCH_SIZE))
    t_start = time.perf_counter()

    with torch.inference_mode():
        for batch_start in range(0, len(images), batch_size):
            batch_images = []
            for img in images[batch_start: batch_start + batch_size]:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                batch_images.append(img)
            inputs = transform(batch_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            try:
                outputs = model(**inputs)
            except (RuntimeError, NotImplementedError) as err:
                err_msg = str(err)
                # Some torch/transformers ops (e.g. upsample_bicubic2d) can fail on MPS.
                # Fall back to CPU for DINO only, while keeping LoFTR on preferred device.
                is_mps_op_gap = (
                    str(device) == "mps"
                    and (
                        "upsample_bicubic2d" in err_msg
                        or "MPS backend" in err_msg
                        or "not currently implemented for the MPS" in err_msg
                    )
                )
                if not is_mps_op_gap:
                    raise
                logger.warning(
                    "DINOv2 MPS op unsupported, falling back to CPU for embeddings: %s",
                    err_msg.splitlines()[0] if err_msg else "unknown",
                )
                model = model.to(torch.device("cpu"))
                device = torch.device("cpu")
                fell_back_from_mps = True
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            for row in batch_embeddings:
                embeddings.append(np.asarray(row, dtype=np.float32))

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    logger.info(
        "DINOv2 embedding timing: photos=%s batch_size=%s total_ms=%.1f avg_ms=%.2f device=%s",
        len(images),
        batch_size,
        elapsed_ms,
        elapsed_ms / max(1, len(images)),
        f"{device}{' (fallback from mps)' if fell_back_from_mps else ''}",
    )
    return np.asarray(embeddings, dtype=np.float32)


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
RANSAC_REPROJ_THRESHOLD = 3.0     # Wider threshold for indoor wide-angle real-estate pairs
OVERLAP_THRESHOLD = 0.15          # Minimum score to connect photos
MIN_GEOMETRIC_SCORE_FOR_EDGE = 0.30
LOW_SCORE_MOTION_COHERENCE_MIN = 0.08
LOW_SCORE_SEGMENT_STRENGTH_MIN = 0.10
FUNDAMENTAL_SAMPSON_THRESHOLD = 3.0  # Relaxed epipolar residual filter (pixels)
FUNDAMENTAL_RANSAC_CONFIDENCE = 0.995
ENABLE_FUNDAMENTAL_SAMPSON_REFINEMENT = False

# Room label mismatch controls.
# Strict mode blocks cross-room pairs entirely (including adjacent photos).
STRICT_ROOM_LABEL_BLOCK = True
STRICT_SOCIAL_CROSS_ROOM_RECHECK_MIN_SEMANTIC = 0.65  # Allow geometry check for likely open-plan relabel errors
STRICT_SERVICE_CROSS_ROOM_RECHECK_MIN_SEMANTIC = 0.60  # Allow adjacent bathroom/laundry relabel errors to reach geometry
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
MIN_SCORE_FAR_APART = 0.18         # Gap>=3 links also need minimum geometric confidence
VERY_FAR_POSITION_GAP_THRESHOLD = 5    # Gap>=5 is already very far in listing order; require stronger evidence
MIN_SEMANTIC_FOR_VERY_FAR = 0.55       # Very-far pairs also need moderate semantic affinity
MIN_INLIERS_VERY_FAR = 35              # Enforce stronger geometry for very-far pairs
MIN_SCORE_VERY_FAR = 0.45              # Reject weak-overlap scores for very-far pairs

# Adaptive geometric matching thresholds
DEFAULT_LOFTR_INPUT_SIZE = (768, 576)  # width, height
DEFAULT_PRODUCTION_MATCHER = "loftr_kornia_indoor_native"
LOFTR_NATIVE_CONFIDENCE_THRESHOLD = 0.20
NATIVE_SCORE_COUNT_ZERO = 80
NATIVE_SCORE_COUNT_TARGET = 260
NATIVE_EDGE_MIN_MATCHES = 40
NATIVE_EDGE_MIN_INLIERS = 20
NATIVE_EDGE_MIN_INLIER_RATIO = 0.25
NATIVE_EDGE_MIN_OVERLAP_RATIO = 0.12
NATIVE_EDGE_MIN_MEAN = 0.55
NATIVE_EDGE_MIN_MEDIAN = 0.54
ROBUST_SUPPORT_INLIER_ZERO = 12
ROBUST_SUPPORT_INLIER_FULL = 40
ROBUST_SUPPORT_OVERLAP_ZERO = 0.08
ROBUST_SUPPORT_OVERLAP_FULL = 0.25
ROBUST_SUPPORT_MIN_FACTOR = 0.20
ROBUST_RATIO_DENOMINATOR_MIN = 60
ROBUST_SCORE_MIN_INLIERS = 20
ROBUST_SCORE_MIN_ACTIVE_MATCHES = 40
FINAL_GATE_MIN_INLIERS = 20
FINAL_GATE_MIN_INLIER_RATIO = 0.20
FINAL_GATE_MIN_OVERLAP_RATIO = 0.10
NATIVE_EDGE_ALLOWED_GEOMETRY_MODELS = {"fundamental_magsac", "fundamental_ransac"}
# If native forward pass is weak, retry reverse orientation and keep the stronger result.
# This stabilizes pair-debug and clustering against directional LoFTR asymmetry.
NATIVE_REVERSE_RETRY_ENABLED = True
NATIVE_REVERSE_RETRY_MATCH_THRESHOLD = 120
NATIVE_REVERSE_RETRY_INLIER_THRESHOLD = 25
# Do not run reverse retry if forward geometry score is already acceptable.
NATIVE_REVERSE_RETRY_SCORE_THRESHOLD = MIN_GEOMETRIC_SCORE_FOR_EDGE

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
    sampson = _compute_fundamental_sampson_errors(p0, p1, f)
    if sampson.size == 0:
        return mask
    refined_local = sampson <= float(threshold * threshold)

    refined = np.zeros_like(mask, dtype=bool)
    refined_indices = np.where(mask)[0]
    refined[refined_indices] = refined_local
    return refined


def _compute_fundamental_sampson_errors(
    points0: np.ndarray,
    points1: np.ndarray,
    fundamental: np.ndarray | None,
) -> np.ndarray:
    f = np.asarray(fundamental, dtype=np.float64) if fundamental is not None else None
    if f is None or f.shape != (3, 3) or not np.isfinite(f).all():
        return np.empty((0,), dtype=np.float64)
    p0 = np.asarray(points0, dtype=np.float64)
    p1 = np.asarray(points1, dtype=np.float64)
    if p0.shape[0] == 0 or p0.shape[0] != p1.shape[0]:
        return np.empty((0,), dtype=np.float64)

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
    return numer / denom


def _estimate_geometric_inliers(
    points0: np.ndarray,
    points1: np.ndarray,
    reproj_threshold: float = RANSAC_REPROJ_THRESHOLD,
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
                float(FUNDAMENTAL_RANSAC_CONFIDENCE),
            )
            inlier_mask = _safe_inlier_mask(mask, len(points0))
            if inlier_mask is None:
                continue
            if ENABLE_FUNDAMENTAL_SAMPSON_REFINEMENT:
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

    if best_fundamental_mask is not None and best_fundamental_count > 0:
        # Fundamental inliers are the only geometry source in production flow.
        return best_fundamental_mask, best_fundamental_model

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


def _get_native_preprocessed_entry(
    image: Image.Image,
    target_long_side: int,
) -> Dict[str, Any]:
    image_gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
    gray_resized, meta = _resize_by_longest_side_and_pad(
        image_gray,
        target_long_side=target_long_side,
        multiple=8,
    )
    return {
        "gray_resized": gray_resized,
        "meta": meta,
    }


def _build_native_tensor(
    gray_resized: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.from_numpy(gray_resized).unsqueeze(0).unsqueeze(0)
    if tensor.device != device:
        tensor = tensor.to(device, non_blocking=True)
    return tensor


def _mps_memory_stats_mb() -> Dict[str, float]:
    """Best-effort MPS allocator stats in MB for runtime diagnostics."""
    stats = {
        "mps_current_allocated_mb": 0.0,
        "mps_driver_allocated_mb": 0.0,
    }
    mps_mod = getattr(torch, "mps", None)
    if mps_mod is None:
        return stats
    try:
        cur_fn = getattr(mps_mod, "current_allocated_memory", None)
        if callable(cur_fn):
            stats["mps_current_allocated_mb"] = float(cur_fn()) / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        drv_fn = getattr(mps_mod, "driver_allocated_memory", None)
        if callable(drv_fn):
            stats["mps_driver_allocated_mb"] = float(drv_fn()) / (1024.0 * 1024.0)
    except Exception:
        pass
    return stats


def _release_device_cache(device: torch.device) -> Dict[str, float]:
    """Release unused accelerator cache to keep per-pair memory bounded."""
    if device.type == "cuda":
        try:
            torch.cuda.synchronize(device=device)
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return {}

    if device.type == "mps":
        mps_mod = getattr(torch, "mps", None)
        if mps_mod is not None:
            try:
                sync_fn = getattr(mps_mod, "synchronize", None)
                if callable(sync_fn):
                    sync_fn()
            except Exception:
                pass
            try:
                empty_fn = getattr(mps_mod, "empty_cache", None)
                if callable(empty_fn):
                    empty_fn()
            except Exception:
                pass
        return _mps_memory_stats_mb()

    return {}


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
    inlier_ratio_denominator: int | None = None,
) -> Dict[str, float]:
    """Return detailed score components used by transition geometry scoring."""
    if num_matches <= 0 or num_inliers <= 0:
        return {
            "inlier_ratio": 0.0,
            "inlier_ratio_numerator": 0.0,
            "inlier_ratio_denominator": 0.0,
            "inlier_ratio_term": 0.0,
            "inlier_volume_bonus": 0.0,
            "inlier_volume_target": 0.0,
            "spread_area0": 0.0,
            "spread_area1": 0.0,
            "spread_bonus": 0.0,
            "segment_strength": 0.0,
            "segment_bonus": 0.0,
            "motion_coherence": 0.0,
            "motion_unit_mean_dx": 0.0,
            "motion_unit_mean_dy": 0.0,
            "motion_valid_count": 0.0,
            "motion_multiplier": 0.0,
            "final_score": 0.0,
        }

    ratio_den = int(inlier_ratio_denominator) if inlier_ratio_denominator is not None else int(num_matches)
    ratio_den = max(1, ratio_den)
    inlier_ratio = float(num_inliers / ratio_den)
    zero_point = float(np.clip(inlier_ratio_zero_point, 0.0, 0.99))
    full_point = float(np.clip(inlier_ratio_full_point, zero_point + 1e-6, 1.0))
    inlier_ratio_term = float(np.clip((inlier_ratio - zero_point) / (full_point - zero_point), 0.0, 1.0))
    # Scale volume target with the active (non-low-texture) pool so exclusions do not penalize score.
    inlier_volume_target = float(np.clip(0.60 * float(ratio_den), 40.0, 140.0))
    inlier_volume_bonus = min(1.0, float(num_inliers) / max(1.0, inlier_volume_target))

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
        motion_mean = unit.mean(axis=0)
        motion_mean_dx = float(motion_mean[0])
        motion_mean_dy = float(motion_mean[1])
        motion_coherence = float(np.linalg.norm(motion_mean))
        motion_valid_count = int(valid.sum())
    else:
        motion_mean_dx = 0.0
        motion_mean_dy = 0.0
        motion_coherence = 0.0
        motion_valid_count = 0

    motion_multiplier = (0.45 + 0.55 * motion_coherence)

    score = (
        inlier_ratio_term
        * (0.45 + 0.55 * inlier_volume_bonus)
        * (0.35 + 0.65 * spread_bonus)
        * (0.40 + 0.60 * segment_bonus)
        * motion_multiplier
    )
    final_score = float(np.clip(score, 0.0, 1.0))

    return {
        "inlier_ratio": float(inlier_ratio),
        "inlier_ratio_numerator": float(num_inliers),
        "inlier_ratio_denominator": float(ratio_den),
        "inlier_ratio_term": float(inlier_ratio_term),
        "inlier_ratio_zero_point": float(zero_point),
        "inlier_ratio_full_point": float(full_point),
        "inlier_volume_bonus": float(inlier_volume_bonus),
        "inlier_volume_target": float(inlier_volume_target),
        "spread_area0": float(area0),
        "spread_area1": float(area1),
        "spread_bonus": float(spread_bonus),
        "segment_strength": float(transition_segment_strength),
        "segment_bonus": float(segment_bonus),
        "motion_coherence": float(motion_coherence),
        "motion_unit_mean_dx": float(motion_mean_dx),
        "motion_unit_mean_dy": float(motion_mean_dy),
        "motion_valid_count": float(motion_valid_count),
        "motion_multiplier": float(motion_multiplier),
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


def _compute_robust_overlap_components(
    inlier_ratio: float,
    overlap_ratio: float,
    median_epipolar_error: float,
    f_inliers: int,
) -> Dict[str, float]:
    ratio_term = float(np.clip(inlier_ratio, 0.0, 1.0))
    overlap_term = float(np.clip(overlap_ratio, 0.0, 1.0))
    error_term = float(np.clip(1.0 - (float(median_epipolar_error) / 5.0), 0.0, 1.0))
    inlier_term = float(np.clip(float(f_inliers) / 100.0, 0.0, 1.0))
    base_score = float(0.45 * ratio_term + 0.40 * overlap_term + 0.10 * error_term + 0.05 * inlier_term)
    # Small-support pairs can show inflated ratio/error metrics; damp robust score
    # unless both inlier support and overlap support are meaningful.
    inlier_support = float(
        np.clip(
            (float(f_inliers) - float(ROBUST_SUPPORT_INLIER_ZERO))
            / max(1e-6, float(ROBUST_SUPPORT_INLIER_FULL - ROBUST_SUPPORT_INLIER_ZERO)),
            0.0,
            1.0,
        )
    )
    overlap_support = float(
        np.clip(
            (float(overlap_ratio) - float(ROBUST_SUPPORT_OVERLAP_ZERO))
            / max(1e-6, float(ROBUST_SUPPORT_OVERLAP_FULL - ROBUST_SUPPORT_OVERLAP_ZERO)),
            0.0,
            1.0,
        )
    )
    support_multiplier = float(
        float(ROBUST_SUPPORT_MIN_FACTOR)
        + (1.0 - float(ROBUST_SUPPORT_MIN_FACTOR)) * inlier_support
    )
    overlap_multiplier = float(
        float(ROBUST_SUPPORT_MIN_FACTOR)
        + (1.0 - float(ROBUST_SUPPORT_MIN_FACTOR)) * overlap_support
    )
    combined_support = float(np.sqrt(max(0.0, support_multiplier * overlap_multiplier)))
    final_score = float(np.clip(base_score * combined_support, 0.0, 1.0))
    return {
        "base_score": float(base_score),
        "inlier_support": float(inlier_support),
        "overlap_support": float(overlap_support),
        "inlier_support_zero": float(ROBUST_SUPPORT_INLIER_ZERO),
        "inlier_support_full": float(ROBUST_SUPPORT_INLIER_FULL),
        "overlap_support_zero": float(ROBUST_SUPPORT_OVERLAP_ZERO),
        "overlap_support_full": float(ROBUST_SUPPORT_OVERLAP_FULL),
        "support_multiplier": float(support_multiplier),
        "overlap_multiplier": float(overlap_multiplier),
        "combined_support_multiplier": float(combined_support),
        "final_score": float(final_score),
    }


def _compute_robust_overlap_score(
    inlier_ratio: float,
    overlap_ratio: float,
    median_epipolar_error: float,
    f_inliers: int,
) -> float:
    return float(
        _compute_robust_overlap_components(
            inlier_ratio=inlier_ratio,
            overlap_ratio=overlap_ratio,
            median_epipolar_error=median_epipolar_error,
            f_inliers=f_inliers,
        )["final_score"]
    )


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
    """Return final edge weight for accepted pairs.

    Production geometry-only mode uses pure geometric score.
    Semantic affinity remains candidate retrieval only.
    """
    geo = float(np.clip(geometric_score, 0.0, 1.0))
    if GEOMETRY_ONLY_CLUSTER_MEMBERSHIP:
        return geo
    sem = float(np.clip(semantic_score, 0.0, 1.0))
    w_geo = max(0.0, float(GEOMETRIC_SCORE_WEIGHT))
    w_sem = max(0.0, float(SEMANTIC_SCORE_WEIGHT))
    if (w_geo + w_sem) <= 1e-8:
        return geo
    return float(np.clip(((w_geo * geo) + (w_sem * sem)) / (w_geo + w_sem), 0.0, 1.0))


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
        logger.info(
            "    strict_gate decision=reject num_inliers=%s inlier_ratio=%.3f overlap_ratio=%.3f robust_valid=%s "
            "combined_score=%.3f reason=%s",
            num_inliers,
            0.0,
            0.0,
            "no",
            float(geometric_score) if geometric_score is not None else 0.0,
            "missing_counts",
        )
        return False

    inlier_ratio = float(num_inliers) / max(1.0, float(num_matches))
    overlap_ratio = 0.0
    robust_valid = False
    combined_score = float(geometric_score) if geometric_score is not None else 0.0
    score_components: Dict[str, Any] = {}
    if isinstance(diagnostics, dict):
        score_components = diagnostics.get("score_components")
        if isinstance(score_components, dict):
            inlier_ratio = float(score_components.get("inlier_ratio", inlier_ratio) or inlier_ratio)
            overlap_ratio_raw = float(score_components.get("overlap_ratio", 0.0) or 0.0)
            overlap_ratio_cov = float(score_components.get("robust_coverage", 0.0) or 0.0)
            overlap_ratio = max(overlap_ratio_raw, overlap_ratio_cov)
            robust_valid = bool(int(score_components.get("robust_score_valid", 0) or 0))
            combined_score = float(score_components.get("combined_score", combined_score) or combined_score)

    # Final geometric safety gate (hard reject).
    if int(num_inliers) < int(FINAL_GATE_MIN_INLIERS):
        logger.info(
            "    strict_gate decision=reject num_inliers=%s inlier_ratio=%.3f overlap_ratio=%.3f robust_valid=%s "
            "combined_score=%.3f reason=%s",
            num_inliers,
            inlier_ratio,
            overlap_ratio,
            "yes" if robust_valid else "no",
            combined_score,
            "final_gate_inliers",
        )
        return False
    if inlier_ratio < float(FINAL_GATE_MIN_INLIER_RATIO):
        logger.info(
            "    strict_gate decision=reject num_inliers=%s inlier_ratio=%.3f overlap_ratio=%.3f robust_valid=%s "
            "combined_score=%.3f reason=%s",
            num_inliers,
            inlier_ratio,
            overlap_ratio,
            "yes" if robust_valid else "no",
            combined_score,
            "final_gate_inlier_ratio",
        )
        return False
    if overlap_ratio < float(FINAL_GATE_MIN_OVERLAP_RATIO):
        logger.info(
            "    strict_gate decision=reject num_inliers=%s inlier_ratio=%.3f overlap_ratio=%.3f robust_valid=%s "
            "combined_score=%.3f reason=%s",
            num_inliers,
            inlier_ratio,
            overlap_ratio,
            "yes" if robust_valid else "no",
            combined_score,
            "final_gate_overlap_ratio",
        )
        return False

    required_inliers = max(int(min_inliers_required), int(NATIVE_EDGE_MIN_INLIERS))
    if num_inliers < required_inliers:
        logger.info(
            "    strict_gate decision=reject num_inliers=%s inlier_ratio=%.3f overlap_ratio=%.3f robust_valid=%s "
            "combined_score=%.3f reason=%s",
            num_inliers,
            inlier_ratio,
            overlap_ratio,
            "yes" if robust_valid else "no",
            combined_score,
            "min_inliers_required",
        )
        return False
    if inlier_ratio < float(NATIVE_EDGE_MIN_INLIER_RATIO):
        logger.info(
            "    strict_gate decision=reject num_inliers=%s inlier_ratio=%.3f overlap_ratio=%.3f robust_valid=%s "
            "combined_score=%.3f reason=%s",
            num_inliers,
            inlier_ratio,
            overlap_ratio,
            "yes" if robust_valid else "no",
            combined_score,
            "native_inlier_ratio",
        )
        return False
    if overlap_ratio < float(NATIVE_EDGE_MIN_OVERLAP_RATIO):
        logger.info(
            "    strict_gate decision=reject num_inliers=%s inlier_ratio=%.3f overlap_ratio=%.3f robust_valid=%s "
            "combined_score=%.3f reason=%s",
            num_inliers,
            inlier_ratio,
            overlap_ratio,
            "yes" if robust_valid else "no",
            combined_score,
            "native_overlap_ratio",
        )
        return False
    if not isinstance(diagnostics, dict):
        logger.info(
            "    strict_gate decision=reject num_inliers=%s inlier_ratio=%.3f overlap_ratio=%.3f robust_valid=%s "
            "combined_score=%.3f reason=%s",
            num_inliers,
            inlier_ratio,
            overlap_ratio,
            "yes" if robust_valid else "no",
            combined_score,
            "missing_diagnostics",
        )
        return False

    geom_model = str(diagnostics.get("geometry_model") or "").strip().lower()
    if geom_model not in NATIVE_EDGE_ALLOWED_GEOMETRY_MODELS:
        logger.info(
            "    strict_gate decision=reject num_inliers=%s inlier_ratio=%.3f overlap_ratio=%.3f robust_valid=%s "
            "combined_score=%.3f reason=%s",
            num_inliers,
            inlier_ratio,
            overlap_ratio,
            "yes" if robust_valid else "no",
            combined_score,
            f"geometry_model:{geom_model or 'none'}",
        )
        return False

    logger.info(
        "    strict_gate decision=accept num_inliers=%s inlier_ratio=%.3f overlap_ratio=%.3f robust_valid=%s "
        "combined_score=%.3f reason=%s",
        num_inliers,
        inlier_ratio,
        overlap_ratio,
        "yes" if robust_valid else "no",
        combined_score,
        "passed",
    )
    return True


def _extract_pair_runtime_metrics(
    diagnostics: Dict[str, Any] | None,
    fallback_pair_time_s: float,
) -> Dict[str, Any]:
    timing = diagnostics.get("timing") if isinstance(diagnostics, dict) else None
    if not isinstance(timing, dict):
        return {
            "time_pair_total_s": float(max(0.0, fallback_pair_time_s)),
            "time_loftr_s": 0.0,
            "time_loftr_forward_main_s": 0.0,
            "time_loftr_forward_reverse_s": 0.0,
            "time_resize_s": 0.0,
            "time_tensor_transfer_s": 0.0,
            "time_device_cleanup_s": 0.0,
            "time_postprocess_s": 0.0,
            "time_f_s": 0.0,
            "time_h_s": 0.0,
            "time_scoring_s": 0.0,
            "forward_pass_count": 1,
            "reverse_attempted": False,
            "reverse_selected": False,
            "model_device": "n/a",
            "tensor_device": "n/a",
            "mps_current_allocated_mb_before": 0.0,
            "mps_driver_allocated_mb_before": 0.0,
            "mps_current_allocated_mb_after": 0.0,
            "mps_driver_allocated_mb_after": 0.0,
        }
    return {
        "time_pair_total_s": float(timing.get("time_pair_total_s", fallback_pair_time_s) or fallback_pair_time_s),
        "time_loftr_s": float(timing.get("time_loftr_s", 0.0) or 0.0),
        "time_loftr_forward_main_s": float(timing.get("time_loftr_forward_main_s", 0.0) or 0.0),
        "time_loftr_forward_reverse_s": float(timing.get("time_loftr_forward_reverse_s", 0.0) or 0.0),
        "time_resize_s": float(timing.get("time_resize_s", 0.0) or 0.0),
        "time_tensor_transfer_s": float(timing.get("time_tensor_transfer_s", 0.0) or 0.0),
        "time_device_cleanup_s": float(timing.get("time_device_cleanup_s", 0.0) or 0.0),
        "time_postprocess_s": float(timing.get("time_postprocess_s", 0.0) or 0.0),
        "time_f_s": float(timing.get("time_f_s", 0.0) or 0.0),
        "time_h_s": float(timing.get("time_h_s", 0.0) or 0.0),
        "time_scoring_s": float(timing.get("time_scoring_s", 0.0) or 0.0),
        "forward_pass_count": int(timing.get("forward_pass_count", 1) or 1),
        "reverse_attempted": bool(timing.get("reverse_attempted", False)),
        "reverse_selected": bool(timing.get("reverse_selected", False)),
        "model_device": str(timing.get("model_device", "n/a")),
        "tensor_device": str(timing.get("tensor_device", "n/a")),
        "mps_current_allocated_mb_before": float(timing.get("mps_current_allocated_mb_before", 0.0) or 0.0),
        "mps_driver_allocated_mb_before": float(timing.get("mps_driver_allocated_mb_before", 0.0) or 0.0),
        "mps_current_allocated_mb_after": float(timing.get("mps_current_allocated_mb_after", 0.0) or 0.0),
        "mps_driver_allocated_mb_after": float(timing.get("mps_driver_allocated_mb_after", 0.0) or 0.0),
    }


def _accumulate_pair_runtime(stage_timers: Dict[str, float], pair_metrics: Dict[str, Any]) -> None:
    stage_timers["time_loftr_total"] += float(pair_metrics["time_loftr_s"])
    stage_timers["time_f_total"] += float(pair_metrics["time_f_s"])
    stage_timers["time_h_total"] += float(pair_metrics["time_h_s"])
    stage_timers["time_scoring_total"] += float(pair_metrics["time_scoring_s"])
    stage_timers["time_resize_total"] += float(pair_metrics["time_resize_s"])
    stage_timers["time_tensor_transfer_total"] += float(pair_metrics["time_tensor_transfer_s"])
    stage_timers["time_device_cleanup_total"] += float(pair_metrics["time_device_cleanup_s"])
    stage_timers["time_postprocess_total"] += float(pair_metrics["time_postprocess_s"])


def _extract_pair_quality_debug_metrics(
    num_matches: int,
    num_inliers: int,
    diagnostics: Dict[str, Any] | None,
) -> Tuple[float, float | None]:
    inlier_ratio_dbg = float(num_inliers) / max(1.0, float(num_matches))
    overlap_ratio_dbg: float | None = None
    if isinstance(diagnostics, dict):
        score_components = diagnostics.get("score_components")
        if isinstance(score_components, dict):
            inlier_ratio_dbg = float(score_components.get("inlier_ratio", inlier_ratio_dbg) or inlier_ratio_dbg)
            overlap_ratio_dbg = float(
                score_components.get("overlap_ratio", score_components.get("robust_coverage", 0.0)) or 0.0
            )
    return inlier_ratio_dbg, overlap_ratio_dbg


def _log_pair_timing(
    phase: str,
    left_photo_id: int,
    right_photo_id: int,
    pair_metrics: Dict[str, Any],
    num_matches: int,
    num_inliers: int,
    diagnostics: Dict[str, Any] | None,
) -> None:
    inlier_ratio_dbg, overlap_ratio_dbg = _extract_pair_quality_debug_metrics(
        num_matches=num_matches,
        num_inliers=num_inliers,
        diagnostics=diagnostics,
    )
    logger.info(
        "pair_timing phase=%s pair=%s<->%s resize_ms=%.1f xfer_ms=%.1f loFTR_ms=%.1f loFTR_main_ms=%.1f loFTR_reverse_ms=%.1f "
        "F_ms=%.1f H_ms=%.1f score_ms=%.1f cleanup_ms=%.1f total_ms=%.1f inlier_ratio=%.3f overlap_ratio=%s passes=%s reverse_attempted=%s reverse_selected=%s model_device=%s tensor_device=%s mps_drv_mb=%s->%s mps_cur_mb=%s->%s",
        phase,
        left_photo_id,
        right_photo_id,
        pair_metrics["time_resize_s"] * 1000.0,
        pair_metrics["time_tensor_transfer_s"] * 1000.0,
        pair_metrics["time_loftr_s"] * 1000.0,
        pair_metrics["time_loftr_forward_main_s"] * 1000.0,
        pair_metrics["time_loftr_forward_reverse_s"] * 1000.0,
        pair_metrics["time_f_s"] * 1000.0,
        pair_metrics["time_h_s"] * 1000.0,
        pair_metrics["time_scoring_s"] * 1000.0,
        pair_metrics["time_device_cleanup_s"] * 1000.0,
        pair_metrics["time_pair_total_s"] * 1000.0,
        inlier_ratio_dbg,
        f"{overlap_ratio_dbg:.3f}" if overlap_ratio_dbg is not None else "n/a",
        pair_metrics["forward_pass_count"],
        "yes" if pair_metrics["reverse_attempted"] else "no",
        "yes" if pair_metrics["reverse_selected"] else "no",
        pair_metrics["model_device"],
        pair_metrics["tensor_device"],
        f"{pair_metrics.get('mps_driver_allocated_mb_before', 0.0):.0f}",
        f"{pair_metrics.get('mps_driver_allocated_mb_after', 0.0):.0f}",
        f"{pair_metrics.get('mps_current_allocated_mb_before', 0.0):.0f}",
        f"{pair_metrics.get('mps_current_allocated_mb_after', 0.0):.0f}",
    )


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
PRE_PIPELINE_DUPLICATE_SIMILARITY_THRESHOLD = 0.95  # Pre-graph prune for obvious duplicates
PRE_PIPELINE_DUPLICATE_MAX_GAP = 4  # Keep dedupe local in sequence to avoid dropping valid alternate viewpoints
KEEP_DUPLICATE_SINGLETON_CLUSTERS = not settings.DELETE_OBVIOUS_DUPLICATES
SPLIT_LONG_GAP_DECAY = 0.25  # Prefer cutting oversized clusters at large capture-order jumps
PAIR_ONLY_SPLIT_GEOMETRY_WEIGHT = 0.45  # Keep geometry primary but let semantics break ties in pair-only split
PAIR_ONLY_SPLIT_SEMANTIC_WEIGHT = 0.55  # Helps avoid cross-room-ish pair picks when max cluster size is 2
PAIR_ONLY_SPLIT_CROSS_ROOM_PENALTY = 0.60  # Reduce cross-room edge preference in final pair-only split
PAIR_ONLY_MIN_OVERLAP_RATIO = 0.12  # Pair-only selection requires enough shared coverage
PAIR_ONLY_MIN_COMBINED_SCORE = 0.22  # Pair-only selection requires minimum overall pair quality
PAIR_ONLY_MIN_INLIERS = 20  # Pair-only selection requires minimum geometric support


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


def allow_strict_cross_room_temporal_recheck(
    room1: str | None,
    room2: str | None,
    temporal_dist: int,
    sem_sim: float,
) -> bool:
    """Allow adjacent same-family label mismatches to be resolved by geometry."""
    if temporal_dist != 1:
        return False
    family1 = room_family(room1)
    family2 = room_family(room2)
    if family1 != family2:
        return False
    sem = float(sem_sim)
    if family1 == "social":
        return sem >= float(STRICT_SOCIAL_CROSS_ROOM_RECHECK_MIN_SEMANTIC)
    if family1 == "service":
        return sem >= float(STRICT_SERVICE_CROSS_ROOM_RECHECK_MIN_SEMANTIC)
    return False


def is_mutual_top_semantic_neighbor(i: int, j: int, similarity: np.ndarray, top_k: int = 2) -> bool:
    """Require mutual top semantic affinity to avoid weak cross-room fallbacks."""
    sorted_i = [idx for idx in np.argsort(-similarity[i]) if idx != i][:top_k]
    sorted_j = [idx for idx in np.argsort(-similarity[j]) if idx != j][:top_k]
    return (j in sorted_i) and (i in sorted_j)


def _prefilter_obvious_room_duplicates(
    images: List[Image.Image],
    photo_ids: List[int],
    room_labels: List[str] | None,
    embeddings: np.ndarray,
    similarity: np.ndarray,
    similarity_records: List[Dict[str, object]],
    similarity_threshold: float = PRE_PIPELINE_DUPLICATE_SIMILARITY_THRESHOLD,
    max_gap: int = PRE_PIPELINE_DUPLICATE_MAX_GAP,
) -> Tuple[
    List[Image.Image],
    List[int],
    List[str] | None,
    np.ndarray,
    np.ndarray,
    Dict[int, int],
]:
    """Drop obvious near-identical photos before candidate pair generation.

    Only dedupes when both photos have the same known room label to avoid
    suppressing valid transitions across mislabeled or unknown rooms.
    """
    n = len(photo_ids)
    if n <= 1:
        return images, photo_ids, room_labels, embeddings, similarity, {}

    keep_mask = np.ones(n, dtype=bool)
    duplicate_of_map: Dict[int, int] = {}
    threshold = float(similarity_threshold)
    gap_limit = max(1, int(max_gap))

    for i in range(n):
        if not bool(keep_mask[i]):
            continue
        room_i = room_labels[i] if room_labels is not None and i < len(room_labels) else None
        room_i_norm = normalize_room_label(room_i)
        if not room_i_norm or room_i_norm == "unknown":
            continue

        max_j = min(n, i + gap_limit + 1)
        for j in range(i + 1, max_j):
            if not bool(keep_mask[j]):
                continue
            room_j = room_labels[j] if room_labels is not None and j < len(room_labels) else None
            room_j_norm = normalize_room_label(room_j)
            if not room_j_norm or room_j_norm == "unknown":
                continue
            if room_i_norm != room_j_norm:
                continue

            sem_sim = float(similarity[i, j])
            if sem_sim < threshold:
                continue

            # Keep earlier photo as canonical, remove later obvious duplicate.
            keep_mask[j] = False
            duplicate_of_map[int(photo_ids[j])] = int(photo_ids[i])
            _append_similarity_record(
                similarity_records=similarity_records,
                photo_ids=photo_ids,
                left_idx=i,
                right_idx=j,
                pair_source="pre_dedup_obvious",
                dinov2_similarity=sem_sim,
                is_connected=False,
            )
            logger.info(
                "Pre-dedup obvious duplicate %s -> %s (room=%s, sem=%.3f, gap=%s)",
                photo_ids[j],
                photo_ids[i],
                room_i_norm,
                sem_sim,
                (j - i),
            )

    if not duplicate_of_map:
        return images, photo_ids, room_labels, embeddings, similarity, {}

    kept_indices = [idx for idx in range(n) if bool(keep_mask[idx])]
    filtered_images = [images[idx] for idx in kept_indices]
    filtered_photo_ids = [int(photo_ids[idx]) for idx in kept_indices]
    filtered_room_labels = (
        [room_labels[idx] for idx in kept_indices]
        if room_labels is not None
        else None
    )
    filtered_embeddings = embeddings[kept_indices]
    filtered_similarity = similarity[np.ix_(kept_indices, kept_indices)]

    logger.info(
        "Pre-dedup summary: %s -> %s photos (removed=%s, threshold=%.2f, max_gap=%s)",
        n,
        len(filtered_photo_ids),
        len(duplicate_of_map),
        threshold,
        gap_limit,
    )
    return (
        filtered_images,
        filtered_photo_ids,
        filtered_room_labels,
        filtered_embeddings,
        filtered_similarity,
        duplicate_of_map,
    )


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
    room_labels: Optional[List[str]] = None,
    pair_metrics_lookup: Optional[Dict[Tuple[int, int], Dict[str, object]]] = None,
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

    if max_size == 2:
        # For pair-only output, build pairs left-to-right on the ordered chain.
        # This preserves temporal continuity while still using blended edge ranking.
        remaining_cluster_indices = [pid_to_idx[pid] for pid in remaining]
        edge_scores: Dict[Tuple[int, int], Dict[str, object]] = {}
        strict_rejected_edges: List[Tuple[int, int, str]] = []
        m = len(remaining)
        for i in range(m):
            for j in range(i + 1, m):
                idx_i = remaining_cluster_indices[i]
                idx_j = remaining_cluster_indices[j]
                geo_strength = float(adjacency[idx_i, idx_j])
                if not np.isfinite(geo_strength) or geo_strength <= 0.0:
                    continue
                src_i = remaining_indices[i]
                src_j = remaining_indices[j]
                sem_strength = float(sem_sim[src_i, src_j])
                if not np.isfinite(sem_strength):
                    sem_strength = 0.0
                blended_strength = (
                    PAIR_ONLY_SPLIT_GEOMETRY_WEIGHT * geo_strength
                    + PAIR_ONLY_SPLIT_SEMANTIC_WEIGHT * sem_strength
                )
                room_penalty = 1.0
                if room_labels and len(room_labels) == len(photo_ids):
                    room_i = room_labels[idx_i]
                    room_j = room_labels[idx_j]
                    if rooms_are_different(room_i, room_j):
                        room_penalty = PAIR_ONLY_SPLIT_CROSS_ROOM_PENALTY

                adjusted_strength = blended_strength * room_penalty
                pair_key = (min(remaining[i], remaining[j]), max(remaining[i], remaining[j]))
                pair_record = pair_metrics_lookup.get(pair_key) if pair_metrics_lookup else None
                if not isinstance(pair_record, dict):
                    strict_rejected_edges.append((remaining[i], remaining[j], "missing_pair_record"))
                    continue

                combined_score = _float_or_none(pair_record.get("combined_score"))
                if combined_score is None:
                    strict_rejected_edges.append((remaining[i], remaining[j], "missing_combined_score"))
                    continue

                overlap_ratio = _float_or_none(pair_record.get("overlap_ratio"))
                if overlap_ratio is None:
                    strict_rejected_edges.append((remaining[i], remaining[j], "missing_overlap_ratio"))
                    continue

                inliers = _int_or_none(pair_record.get("geometric_inliers"))
                if inliers is None:
                    strict_rejected_edges.append((remaining[i], remaining[j], "missing_geometric_inliers"))
                    continue

                if float(overlap_ratio) < float(PAIR_ONLY_MIN_OVERLAP_RATIO):
                    strict_rejected_edges.append((remaining[i], remaining[j], "low_overlap_ratio"))
                    continue
                if float(combined_score) < float(PAIR_ONLY_MIN_COMBINED_SCORE):
                    strict_rejected_edges.append((remaining[i], remaining[j], "low_combined_score"))
                    continue
                if int(inliers) < int(PAIR_ONLY_MIN_INLIERS):
                    strict_rejected_edges.append((remaining[i], remaining[j], "low_inliers"))
                    continue

                rank_key = (
                    float(overlap_ratio),
                    float(combined_score),
                    int(inliers),
                    float(adjusted_strength),
                    float(sem_strength),
                )
                edge_scores[(i, j)] = {
                    "rank_key": rank_key,
                    "adjusted_strength": adjusted_strength,
                    "geo_strength": geo_strength,
                    "sem_strength": sem_strength,
                    "room_penalty": room_penalty,
                    "combined_score": float(combined_score),
                    "overlap_ratio": float(overlap_ratio),
                    "inliers": int(inliers),
                }

        if strict_rejected_edges:
            reasons: Dict[str, int] = {}
            for _, _, reason in strict_rejected_edges:
                reasons[reason] = reasons.get(reason, 0) + 1
            examples = ", ".join(
                f"{left}<->{right}:{reason}" for left, right, reason in strict_rejected_edges[:8]
            )
            logger.warning(
                "Pair-only strict ranking rejected %s candidate edges due missing metrics (reasons=%s, examples=%s)",
                len(strict_rejected_edges),
                reasons,
                examples or "none",
            )

        used_local: set[int] = set()
        paired_clusters: List[List[int]] = []

        for i in range(m):
            if i in used_local:
                continue
            best_j: Optional[int] = None
            best_rank = (0.0, 0.0, 0, 0.0, 0.0)
            best_adjusted = 0.0
            best_geo = 0.0
            best_sem = 0.0
            best_penalty = 1.0
            best_combined = 0.0
            best_overlap = 0.0
            best_inliers = 0

            for j in range(i + 1, m):
                if j in used_local:
                    continue
                edge_info = edge_scores.get((i, j))
                if not edge_info:
                    continue
                rank_key = edge_info.get("rank_key", (0.0, 0.0, 0, 0.0, 0.0))
                if rank_key > best_rank:
                    best_rank = rank_key
                    best_j = j
                    best_adjusted = float(edge_info.get("adjusted_strength", 0.0) or 0.0)
                    best_geo = float(edge_info.get("geo_strength", 0.0) or 0.0)
                    best_sem = float(edge_info.get("sem_strength", 0.0) or 0.0)
                    best_penalty = float(edge_info.get("room_penalty", 1.0) or 1.0)
                    best_combined = float(edge_info.get("combined_score", 0.0) or 0.0)
                    best_overlap = float(edge_info.get("overlap_ratio", 0.0) or 0.0)
                    best_inliers = int(edge_info.get("inliers", 0) or 0)

            if best_j is not None and best_adjusted > 0.0:
                paired_clusters.append([remaining[i], remaining[best_j]])
                used_local.add(i)
                used_local.add(best_j)
                logger.debug(
                    "Pair-only split selected edge %s<->%s combined=%.3f overlap=%.3f inliers=%s adjusted=%.3f geo=%.3f sem=%.3f room_penalty=%.2f",
                    remaining[i],
                    remaining[best_j],
                    best_combined,
                    best_overlap,
                    best_inliers,
                    best_adjusted,
                    best_geo,
                    best_sem,
                    best_penalty,
                )
            else:
                paired_clusters.append([remaining[i]])
                used_local.add(i)

        # Keep deterministic order in output based on original cluster position.
        original_pos = {pid: idx for idx, pid in enumerate(cluster_photo_ids)}
        paired_clusters.sort(key=lambda cluster: min(original_pos.get(pid, 10**9) for pid in cluster))
        sizes = [len(c) for c in paired_clusters]
        logger.info(
            "Pair-only split: %s photos -> %s canonical clusters (sizes: %s)%s",
            len(cluster_photo_ids),
            len(paired_clusters),
            sizes,
            (
                f" + {len(duplicate_clusters)} duplicate singleton clusters"
                if keep_duplicate_singletons and duplicate_clusters
                else ""
            ),
        )
        return paired_clusters + emitted_duplicate_clusters, duplicate_of

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


def _score_metrics_from_diagnostics(diagnostics: Dict[str, Any] | None) -> Dict[str, float | int | None]:
    score_components = diagnostics.get("score_components") if diagnostics else {}
    raw = score_components if isinstance(score_components, dict) else {}
    overlap_ratio = _float_or_none(raw.get("overlap_ratio"))
    if overlap_ratio is None:
        overlap_ratio = _float_or_none(raw.get("robust_coverage"))
    return {
        "combined_score": _float_or_none(raw.get("combined_score")),
        "overlap_ratio": overlap_ratio,
        "inlier_ratio": _float_or_none(raw.get("inlier_ratio")),
    }


def _build_pair_metrics_payload(diagnostics: Dict[str, Any] | None) -> Dict[str, float | int | None]:
    payload: Dict[str, float | int | None] = {}
    payload.update(_segment_scores_from_diagnostics(diagnostics))
    payload.update(_oracle_metrics_from_diagnostics(diagnostics))
    payload.update(_score_metrics_from_diagnostics(diagnostics))
    return payload


def _append_similarity_record(
    similarity_records: List[Dict[str, object]],
    photo_ids: List[int],
    left_idx: int,
    right_idx: int,
    pair_source: str,
    dinov2_similarity: float,
    is_connected: bool,
    geometric_matches: int | None = None,
    geometric_inliers: int | None = None,
    geometric_score: float | None = None,
    direction: Tuple[float, float] = (0.0, 0.0),
    diagnostics: Dict[str, Any] | None = None,
) -> None:
    direction_dx: float | None = None
    direction_dy: float | None = None
    if direction != (0.0, 0.0):
        direction_dx = float(direction[0])
        direction_dy = float(direction[1])

    similarity_records.append({
        "photo_a_id": photo_ids[min(left_idx, right_idx)],
        "photo_b_id": photo_ids[max(left_idx, right_idx)],
        "pair_source": _annotate_pair_source_with_oracle(pair_source, diagnostics),
        "dinov2_similarity": float(dinov2_similarity),
        "geometric_matches": geometric_matches,
        "geometric_inliers": geometric_inliers,
        "geometric_score": float(geometric_score) if geometric_score is not None else None,
        "direction_dx": direction_dx,
        "direction_dy": direction_dy,
        "is_connected": 1 if is_connected else 0,
        **_build_pair_metrics_payload(diagnostics),
    })


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
_loftr_matchers: Dict[str, Any] = {}
_zju_loftr_matchers: Dict[str, Any] = {}
_zju_loftr_matcher_meta: Dict[str, Dict[str, str]] = {}
_matchformer_matchers: Dict[str, Any] = {}
_matchformer_matcher_meta: Dict[str, Dict[str, str]] = {}
_roma_v2_matchers: Dict[str, Any] = {}
_roma_v2_matcher_meta: Dict[str, Dict[str, str]] = {}
_ZJU_DEBUG_ENV_BY_VARIANT: Dict[str, str] = {
    "indoor_ds": "LOFTR_ZJU_INDOOR_DS_CKPT",
    "indoor_ot": "LOFTR_ZJU_INDOOR_OT_CKPT",
}
_MATCHFORMER_DEBUG_ENV_BY_VARIANT: Dict[str, str] = {
    "indoor": "MATCHFORMER_INDOOR_CKPT",
    "outdoor": "MATCHFORMER_OUTDOOR_CKPT",
}


def _same_torch_device(a: torch.device, b: torch.device) -> bool:
    if a.type != b.type:
        return False
    if a.type in {"cpu", "mps"}:
        return True
    if a.type == "cuda":
        if a.index is None or b.index is None:
            return True
        return a.index == b.index
    return a == b


def _env_or_settings(name: str, default: str = "") -> str:
    raw_env = str(os.getenv(name, "")).strip()
    if raw_env:
        return raw_env
    raw_settings = getattr(settings, name, None)
    if raw_settings is None:
        return str(default)
    return str(raw_settings).strip() or str(default)


def _zju_debug_repo_dir() -> str:
    repo_dir = _env_or_settings("LOFTR_ZJU_REPO_DIR", "")
    if not repo_dir:
        raise RuntimeError("ZJU debug matcher is missing env LOFTR_ZJU_REPO_DIR")
    if not os.path.isdir(repo_dir):
        raise RuntimeError(f"ZJU debug matcher repo path not found: {repo_dir}")
    return repo_dir


def _zju_debug_checkpoint_path(variant: str) -> Tuple[str, str]:
    env_key = _ZJU_DEBUG_ENV_BY_VARIANT.get(str(variant))
    if not env_key:
        raise RuntimeError(f"Unsupported ZJU debug variant '{variant}'")
    ckpt_path = _env_or_settings(env_key, "")
    if not ckpt_path:
        raise RuntimeError(f"ZJU debug matcher is missing env {env_key}")
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"ZJU debug checkpoint not found for {variant}: {ckpt_path}")
    return ckpt_path, env_key


def _load_zju_loftr_debug_variant(variant: str) -> Tuple[Any, Dict[str, str]]:
    variant = str(variant)
    preferred = _preferred_torch_device()
    if variant in _zju_loftr_matchers:
        matcher = _zju_loftr_matchers[variant]
        try:
            current = next(matcher.parameters()).device
        except Exception:
            current = torch.device("cpu")
        if not _same_torch_device(current, preferred):
            matcher = matcher.to(preferred)
            matcher.eval()
            _zju_loftr_matchers[variant] = matcher
            logger.info(
                "Moved cached ZJU LoFTR debug matcher (%s) from %s to %s",
                variant,
                current,
                preferred,
            )
        return matcher, dict(_zju_loftr_matcher_meta.get(variant, {}))

    repo_dir = _zju_debug_repo_dir()
    ckpt_path, env_key = _zju_debug_checkpoint_path(variant)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    try:
        from src.loftr import LoFTR as ZJULoFTR, default_cfg as zju_default_cfg
    except Exception as exc:
        raise RuntimeError(
            f"ZJU debug matcher failed importing repo modules from {repo_dir}: {exc}"
        ) from exc

    cfg = deepcopy(zju_default_cfg)
    if isinstance(cfg.get("match_coarse"), dict):
        cfg["match_coarse"].setdefault("sparse_spvs", False)

    matcher = ZJULoFTR(config=cfg)
    try:
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    except Exception as exc:
        raise RuntimeError(f"ZJU debug matcher failed loading checkpoint '{ckpt_path}': {exc}") from exc

    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise RuntimeError(
            f"ZJU debug matcher checkpoint has invalid state_dict format: {ckpt_path}"
        )

    try:
        incompatible = matcher.load_state_dict(state_dict, strict=False)
    except Exception as exc:
        raise RuntimeError(f"ZJU debug matcher failed loading state_dict for {variant}: {exc}") from exc

    matcher = matcher.to(preferred)
    matcher.eval()
    _zju_loftr_matchers[variant] = matcher
    _zju_loftr_matcher_meta[variant] = {
        "variant": variant,
        "repo_dir": repo_dir,
        "ckpt_path": ckpt_path,
        "env_key": env_key,
        "model_class": matcher.__class__.__name__,
        "missing_keys": str(len(getattr(incompatible, "missing_keys", []) or [])),
        "unexpected_keys": str(len(getattr(incompatible, "unexpected_keys", []) or [])),
    }
    logger.info(
        "Loaded ZJU LoFTR debug matcher (%s) from %s on %s [missing=%s unexpected=%s]",
        variant,
        ckpt_path,
        preferred,
        _zju_loftr_matcher_meta[variant]["missing_keys"],
        _zju_loftr_matcher_meta[variant]["unexpected_keys"],
    )
    return matcher, dict(_zju_loftr_matcher_meta[variant])


def _matchformer_debug_repo_dir() -> str:
    repo_dir = _env_or_settings("MATCHFORMER_REPO_DIR", "")
    if not repo_dir:
        raise RuntimeError("MatchFormer debug matcher is missing env MATCHFORMER_REPO_DIR")
    if not os.path.isdir(repo_dir):
        raise RuntimeError(f"MatchFormer debug matcher repo path not found: {repo_dir}")
    return repo_dir


def _matchformer_debug_checkpoint_path(variant: str) -> Tuple[str, str]:
    env_key = _MATCHFORMER_DEBUG_ENV_BY_VARIANT.get(str(variant))
    if not env_key:
        raise RuntimeError(f"Unsupported MatchFormer debug variant '{variant}'")
    ckpt_path = _env_or_settings(env_key, "")
    if not ckpt_path:
        raise RuntimeError(f"MatchFormer debug matcher is missing env {env_key}")
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"MatchFormer debug checkpoint not found for {variant}: {ckpt_path}")
    return ckpt_path, env_key


def _cfg_to_lower_dict(config: Any) -> Any:
    if isinstance(config, dict):
        return {str(k).lower(): _cfg_to_lower_dict(v) for k, v in config.items()}
    if isinstance(config, (list, tuple)):
        converted = [_cfg_to_lower_dict(v) for v in config]
        return type(config)(converted) if isinstance(config, tuple) else converted
    if hasattr(config, "items"):
        return {str(k).lower(): _cfg_to_lower_dict(v) for k, v in config.items()}
    return config


def _matchformer_config_for_variant(cfg: Any, variant: str) -> Any:
    if variant == "indoor":
        cfg.MATCHFORMER.BACKBONE_TYPE = "largesea"
        cfg.MATCHFORMER.SCENS = "indoor"
    elif variant == "outdoor":
        cfg.MATCHFORMER.BACKBONE_TYPE = "largela"
        cfg.MATCHFORMER.SCENS = "outdoor"
    else:
        raise RuntimeError(f"Unsupported MatchFormer debug variant '{variant}'")

    # Keep parity with the repo test/inference defaults.
    cfg.MATCHFORMER.RESOLUTION = (8, 2)
    cfg.MATCHFORMER.COARSE.D_MODEL = 256
    cfg.MATCHFORMER.COARSE.D_FFN = 256
    return cfg


def _load_matchformer_debug_variant(variant: str) -> Tuple[Any, Dict[str, str]]:
    variant = str(variant)
    requested_device = _env_or_settings("MATCHFORMER_DEBUG_DEVICE", "auto").lower()
    preferred = _preferred_torch_device()
    if requested_device in {"", "auto"}:
        # On macOS MPS, MatchFormer can trigger large CPU fallback buffers.
        # Prefer CPU locally unless explicitly overridden.
        preferred = torch.device("cpu") if preferred.type == "mps" else preferred
    elif requested_device == "cpu":
        preferred = torch.device("cpu")
    elif requested_device == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not bool(mps_backend.is_available()):
            raise RuntimeError("MatchFormer debug matcher requested MATCHFORMER_DEBUG_DEVICE=mps but MPS is unavailable")
        preferred = torch.device("mps")
    elif requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("MatchFormer debug matcher requested MATCHFORMER_DEBUG_DEVICE=cuda but CUDA is unavailable")
        preferred = torch.device("cuda")
    else:
        raise RuntimeError(
            f"MatchFormer debug matcher has invalid MATCHFORMER_DEBUG_DEVICE='{requested_device}'. "
            "Use one of: auto, cpu, mps, cuda."
        )
    if variant in _matchformer_matchers:
        matcher = _matchformer_matchers[variant]
        try:
            current = next(matcher.parameters()).device
        except Exception:
            current = torch.device("cpu")
        if not _same_torch_device(current, preferred):
            matcher = matcher.to(preferred)
            matcher.eval()
            _matchformer_matchers[variant] = matcher
            logger.info(
                "Moved cached MatchFormer debug matcher (%s) from %s to %s",
                variant,
                current,
                preferred,
            )
        return matcher, dict(_matchformer_matcher_meta.get(variant, {}))

    repo_dir = _matchformer_debug_repo_dir()
    ckpt_path, env_key = _matchformer_debug_checkpoint_path(variant)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Importing from timm\.models\.layers is deprecated, please import via timm\.layers",
                category=FutureWarning,
            )
            from config.defaultmf import get_cfg_defaults
            from model.matchformer import Matchformer
    except Exception as exc:
        raise RuntimeError(
            f"MatchFormer debug matcher failed importing repo modules from {repo_dir}: {exc}"
        ) from exc

    try:
        cfg = get_cfg_defaults()
    except Exception as exc:
        raise RuntimeError(f"MatchFormer debug matcher failed building default config: {exc}") from exc
    cfg = _matchformer_config_for_variant(cfg, variant)
    cfg_lower = _cfg_to_lower_dict(cfg)
    model_cfg = cfg_lower.get("matchformer")
    if not isinstance(model_cfg, dict):
        raise RuntimeError("MatchFormer debug matcher config is invalid: missing 'matchformer' block")

    matcher = Matchformer(config=model_cfg)
    try:
        checkpoint = torch.load(
            ckpt_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    except Exception as exc:
        raise RuntimeError(f"MatchFormer debug matcher failed loading checkpoint '{ckpt_path}': {exc}") from exc

    if isinstance(checkpoint, dict):
        nested_state = checkpoint.get("state_dict")
        state_dict = nested_state if isinstance(nested_state, dict) else checkpoint
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise RuntimeError(
            f"MatchFormer debug matcher checkpoint has invalid state_dict format: {ckpt_path}"
        )

    model_state_dict: Dict[str, Any] = {}
    for key, value in state_dict.items():
        normalized_key = str(key)
        if normalized_key.startswith("matcher."):
            normalized_key = normalized_key[len("matcher."):]
        elif normalized_key.startswith("model.matcher."):
            normalized_key = normalized_key[len("model.matcher."):]
        elif normalized_key.startswith("model."):
            normalized_key = normalized_key[len("model."):]
        model_state_dict[normalized_key] = value

    try:
        incompatible = matcher.load_state_dict(model_state_dict, strict=False)
    except Exception as exc:
        raise RuntimeError(
            f"MatchFormer debug matcher failed loading state_dict for {variant}: {exc}"
        ) from exc

    matcher = matcher.to(preferred)
    matcher.eval()
    _matchformer_matchers[variant] = matcher
    _matchformer_matcher_meta[variant] = {
        "variant": variant,
        "repo_dir": repo_dir,
        "ckpt_path": ckpt_path,
        "env_key": env_key,
        "model_class": matcher.__class__.__name__,
        "requested_device": requested_device or "auto",
        "missing_keys": str(len(getattr(incompatible, "missing_keys", []) or [])),
        "unexpected_keys": str(len(getattr(incompatible, "unexpected_keys", []) or [])),
    }
    logger.info(
        "Loaded MatchFormer debug matcher (%s) from %s on %s (requested=%s) [missing=%s unexpected=%s]",
        variant,
        ckpt_path,
        preferred,
        requested_device or "auto",
        _matchformer_matcher_meta[variant]["missing_keys"],
        _matchformer_matcher_meta[variant]["unexpected_keys"],
    )
    return matcher, dict(_matchformer_matcher_meta[variant])


def _load_roma_v2_debug_matcher() -> Tuple[Any, Dict[str, str]]:
    cache_key = "outdoor"
    if cache_key in _roma_v2_matchers:
        return _roma_v2_matchers[cache_key], dict(_roma_v2_matcher_meta.get(cache_key, {}))

    requested_device = _env_or_settings("ROMA_DEBUG_DEVICE", "auto").lower()
    preferred_device = _preferred_torch_device()
    if requested_device in {"", "auto"}:
        # On macOS MPS RoMa uses unsupported ops that bounce to CPU anyway.
        # Prefer CPU locally for stable/consistent timing unless explicitly overridden.
        model_device = torch.device("cpu") if preferred_device.type == "mps" else preferred_device
    elif requested_device == "cpu":
        model_device = torch.device("cpu")
    elif requested_device == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not bool(mps_backend.is_available()):
            raise RuntimeError("RoMa-v2 debug matcher requested ROMA_DEBUG_DEVICE=mps but MPS is unavailable")
        model_device = torch.device("mps")
    elif requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("RoMa-v2 debug matcher requested ROMA_DEBUG_DEVICE=cuda but CUDA is unavailable")
        model_device = torch.device("cuda")
    else:
        raise RuntimeError(
            f"RoMa-v2 debug matcher has invalid ROMA_DEBUG_DEVICE='{requested_device}'. "
            "Use one of: auto, cpu, mps, cuda."
        )
    try:
        from romatch import roma_outdoor
    except Exception as exc:
        raise RuntimeError(
            "RoMa-v2 debug matcher requires package 'romatch'. Install it in media-service venv first."
        ) from exc

    try:
        matcher = roma_outdoor(device=str(model_device))
    except TypeError:
        matcher = roma_outdoor(device=model_device)
    except Exception as exc:
        raise RuntimeError(f"RoMa-v2 debug matcher failed to initialize: {exc}") from exc

    _roma_v2_matchers[cache_key] = matcher
    _roma_v2_matcher_meta[cache_key] = {
        "variant": "outdoor",
        "model_class": matcher.__class__.__name__,
        "device": str(model_device),
        "requested_device": requested_device or "auto",
    }
    logger.info(
        "Loaded RoMa-v2 debug matcher (outdoor) on %s (requested=%s, preferred=%s)",
        model_device,
        requested_device or "auto",
        preferred_device,
    )
    return matcher, dict(_roma_v2_matcher_meta[cache_key])


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
    def _same_device(a: torch.device, b: torch.device) -> bool:
        # On MPS, PyTorch may surface cached params as "mps:0" while preferred is "mps".
        # Treat those as equivalent so we do not keep remapping the same model.
        if a.type != b.type:
            return False
        if a.type in {"cpu", "mps"}:
            return True
        if a.type == "cuda":
            if a.index is None or b.index is None:
                return True
            return a.index == b.index
        return a == b

    if checkpoint in _loftr_matchers:
        matcher = _loftr_matchers[checkpoint]
        preferred = _preferred_torch_device()
        try:
            current = next(matcher.parameters()).device
        except Exception:
            current = torch.device("cpu")
        if not _same_device(current, preferred):
            matcher = matcher.to(preferred)
            matcher.eval()
            _loftr_matchers[checkpoint] = matcher
            logger.info(
                "Moved cached LoFTR matcher (%s) from %s to %s",
                checkpoint,
                current,
                preferred,
            )
        return matcher

    device = _preferred_torch_device()
    from kornia.feature import LoFTR

    matcher = LoFTR(pretrained=checkpoint)
    matcher = matcher.to(device)
    matcher.eval()
    _loftr_matchers[checkpoint] = matcher
    logger.info("Loaded LoFTR matcher (%s) on %s", checkpoint, device)
    return matcher


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
    full_diagnostics: bool = False,
) -> Tuple[int, int, float, Tuple[float, float], Dict[str, Any]]:
    raw_count = int(len(points0))
    conf_mask = scores >= float(confidence_threshold) if scores.size > 0 else np.zeros((0,), dtype=bool)
    conf_points0 = points0[conf_mask] if raw_count > 0 else np.empty((0, 2), dtype=np.float32)
    conf_points1 = points1[conf_mask] if raw_count > 0 else np.empty((0, 2), dtype=np.float32)
    conf_scores = scores[conf_mask] if raw_count > 0 else np.empty((0,), dtype=np.float32)
    kept_count = int(len(conf_points0))
    native_threshold_summary = _matching_score_summary(conf_scores)
    native_raw_summary = _matching_score_summary(scores)
    active_points0 = conf_points0
    active_points1 = conf_points1
    active_scores = conf_scores
    active_count = int(len(active_points0))
    native_score_summary = _matching_score_summary(active_scores)
    inlier_ratio_denominator = int(max(ROBUST_RATIO_DENOMINATOR_MIN, active_count))
    if inlier_ratio_denominator <= 0:
        inlier_ratio_denominator = int(max(ROBUST_RATIO_DENOMINATOR_MIN, kept_count))
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
        num_matches=active_count,
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
        inlier_ratio_denominator=inlier_ratio_denominator,
    )
    geometric_score = 0.0
    robust_overlap_score = 0.0
    overlap_ratio = 0.0
    median_epipolar_error = 5.0
    overlap_centroid_x0 = 0.0
    overlap_centroid_x1 = 0.0
    fundamental_seconds = 0.0

    if active_count >= 8:
        f_started_at = time.perf_counter()
        inlier_mask, geometry_model = _estimate_geometric_inliers(
            active_points0,
            active_points1,
            reproj_threshold=RANSAC_REPROJ_THRESHOLD,
        )
        fundamental_seconds += (time.perf_counter() - f_started_at)
        if inlier_mask is not None:
            num_inliers = int(inlier_mask.sum())
            if num_inliers > 0:
                inlier_points0 = active_points0[inlier_mask]
                inlier_points1 = active_points1[inlier_mask]
                direction = compute_direction_vector(active_points0, active_points1, inlier_mask)
                segment_scores = _compute_segment_scores(
                    inlier_points0,
                    inlier_points1,
                    width0=width0,
                    height0=height0,
                    width1=width1,
                    height1=height1,
                )
                score_components = _compute_transition_geometry_components(
                    num_matches=active_count,
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
                    inlier_ratio_denominator=inlier_ratio_denominator,
                )
                geometric_score = float(score_components.get("final_score", 0.0) or 0.0)
                overlap_centroid_x0 = float(np.mean(inlier_points0[:, 0]) / max(1.0, float(width0)))
                overlap_centroid_x1 = float(np.mean(inlier_points1[:, 0]) / max(1.0, float(width1)))

    scoring_started_at = time.perf_counter()
    inlier_ratio_for_overlap = float(score_components.get("inlier_ratio", 0.0) or 0.0)
    convex_overlap_ratio = float(
        min(
            float(score_components.get("spread_area0", 0.0) or 0.0),
            float(score_components.get("spread_area1", 0.0) or 0.0),
        )
    )
    overlap_ratio = float(convex_overlap_ratio)
    robust_coverage = float(overlap_ratio)
    if (
        geometry_model in {"fundamental_magsac", "fundamental_ransac"}
        and int(num_inliers) >= 8
        and inlier_points0.shape[0] >= 8
        and inlier_points1.shape[0] >= 8
    ):
        try:
            fundamental_refit, _ = cv2.findFundamentalMat(
                inlier_points0,
                inlier_points1,
                cv2.FM_8POINT,
            )
            sampson_errors = _compute_fundamental_sampson_errors(
                inlier_points0,
                inlier_points1,
                fundamental_refit,
            )
            if sampson_errors.size > 0:
                median_epipolar_error = float(np.median(np.sqrt(np.maximum(0.0, sampson_errors))))
        except cv2.error:
            median_epipolar_error = 5.0
    robust_components = _compute_robust_overlap_components(
        inlier_ratio=inlier_ratio_for_overlap,
        overlap_ratio=robust_coverage,
        median_epipolar_error=median_epipolar_error,
        f_inliers=int(num_inliers),
    )
    robust_score_valid = bool(
        int(num_inliers) >= int(ROBUST_SCORE_MIN_INLIERS)
        and int(active_count) >= int(ROBUST_SCORE_MIN_ACTIVE_MATCHES)
    )
    robust_overlap_score_raw = float(robust_components.get("final_score", 0.0) or 0.0)
    # Smooth small-support penalty: medium-low inlier counts should not remain high-scoring.
    if int(num_inliers) < 40:
        robust_overlap_score_raw *= max(0.0, float(num_inliers) / 40.0)
    robust_overlap_score = float(robust_overlap_score_raw) if robust_score_valid else 0.0
    transition_overlap_score = float(geometric_score)
    geometric_score = float(robust_overlap_score)
    overlap_centroid_shift_x = float(overlap_centroid_x1 - overlap_centroid_x0)
    overlap_side_code0 = -1.0 if overlap_centroid_x0 < 0.4 else (1.0 if overlap_centroid_x0 > 0.6 else 0.0)
    overlap_side_code1 = -1.0 if overlap_centroid_x1 < 0.4 else (1.0 if overlap_centroid_x1 > 0.6 else 0.0)

    # Count calibration operates on the confidence-thresholded active set.
    active_count_for_count_term = int(max(1, inlier_ratio_denominator))
    count_zero = float(np.clip(float(NATIVE_SCORE_COUNT_ZERO), 0.0, max(0.0, float(active_count_for_count_term - 1))))
    count_target = float(
        np.clip(float(NATIVE_SCORE_COUNT_TARGET), count_zero + 1.0, float(max(1, active_count_for_count_term)))
    )
    match_count_term = float(
        np.clip((float(active_count_for_count_term) - count_zero) / (count_target - count_zero), 0.0, 1.0)
    )
    if robust_score_valid:
        combined_score = float(geometric_score * (0.30 + 0.70 * match_count_term))
    else:
        combined_score = float(transition_overlap_score * 0.5)
    scoring_seconds = time.perf_counter() - scoring_started_at
    score_components = dict(score_components or {})
    score_components["match_count_source"] = float(active_count_for_count_term)
    score_components["match_count_zero"] = float(count_zero)
    score_components["match_count_target"] = float(count_target)
    score_components["match_count_term"] = float(match_count_term)
    score_components["transition_overlap_score"] = float(transition_overlap_score)
    score_components["robust_overlap_score"] = float(robust_overlap_score)
    score_components["robust_overlap_score_raw"] = float(robust_overlap_score_raw)
    score_components["robust_small_support_factor"] = float(min(1.0, max(0.0, float(num_inliers) / 40.0)))
    score_components["robust_score_valid"] = 1.0 if robust_score_valid else 0.0
    score_components["robust_score_min_inliers"] = float(ROBUST_SCORE_MIN_INLIERS)
    score_components["robust_score_min_active_matches"] = float(ROBUST_SCORE_MIN_ACTIVE_MATCHES)
    score_components["robust_base_score"] = float(robust_components.get("base_score", 0.0) or 0.0)
    score_components["robust_inlier_support"] = float(robust_components.get("inlier_support", 0.0) or 0.0)
    score_components["robust_overlap_support"] = float(robust_components.get("overlap_support", 0.0) or 0.0)
    score_components["robust_support_multiplier"] = float(
        robust_components.get("support_multiplier", 0.0) or 0.0
    )
    score_components["robust_overlap_multiplier"] = float(
        robust_components.get("overlap_multiplier", 0.0) or 0.0
    )
    score_components["robust_combined_support_multiplier"] = float(
        robust_components.get("combined_support_multiplier", 0.0) or 0.0
    )
    score_components["robust_inlier_support_zero"] = float(
        robust_components.get("inlier_support_zero", 0.0) or 0.0
    )
    score_components["robust_inlier_support_full"] = float(
        robust_components.get("inlier_support_full", 0.0) or 0.0
    )
    score_components["robust_overlap_support_zero"] = float(
        robust_components.get("overlap_support_zero", 0.0) or 0.0
    )
    score_components["robust_overlap_support_full"] = float(
        robust_components.get("overlap_support_full", 0.0) or 0.0
    )
    score_components["overlap_ratio"] = float(overlap_ratio)
    score_components["overlap_ratio_convex"] = float(convex_overlap_ratio)
    score_components["robust_coverage"] = float(robust_coverage)
    score_components["overlap_centroid_x0"] = float(overlap_centroid_x0)
    score_components["overlap_centroid_x1"] = float(overlap_centroid_x1)
    score_components["overlap_centroid_shift_x"] = float(overlap_centroid_shift_x)
    score_components["overlap_side_code0"] = float(overlap_side_code0)
    score_components["overlap_side_code1"] = float(overlap_side_code1)
    score_components["median_epipolar_error"] = float(median_epipolar_error)
    score_components["combined_score"] = float(combined_score)
    if full_diagnostics:
        raw_matches_payload = _sample_normalized_matches(
            points0=points0,
            points1=points1,
            width0=width0,
            height0=height0,
            width1=width1,
            height1=height1,
            max_points=5000,
        )
        inlier_matches_payload = _sample_normalized_matches(
            points0=inlier_points0,
            points1=inlier_points1,
            width0=width0,
            height0=height0,
            width1=width1,
            height1=height1,
            max_points=5000,
        )
    else:
        raw_matches_payload = []
        inlier_matches_payload = []

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
        "threshold_match_count": int(kept_count),
        "active_match_count": int(active_count),
        "threshold_trials": [
            {
                "threshold": float(confidence_threshold),
                "raw_matches": int(raw_count),
                "num_matches": int(active_count),
                "num_threshold_matches": int(kept_count),
                "num_active_matches": int(inlier_ratio_denominator),
                "num_inliers": int(num_inliers),
                "score": float(combined_score),
                "geometric_score": float(geometric_score),
                "match_count_source": int(active_count_for_count_term),
                "match_count_zero": float(count_zero),
                "match_count_target": float(count_target),
                "match_count_term": float(match_count_term),
                "geometry_model": geometry_model,
                "inlier_ratio_denominator": int(inlier_ratio_denominator),
                "native_score_mean": float(native_score_summary["mean"]),
                "native_score_median": float(native_score_summary["median"]),
                "native_score_p95": float(native_score_summary["p95"]),
            }
        ],
        "loftr_input_width": int(input_w if input_w is not None else width0),
        "loftr_input_height": int(input_h if input_h is not None else height0),
        "ransac_reproj_threshold": float(RANSAC_REPROJ_THRESHOLD),
        "raw_matches": raw_matches_payload,
        "inlier_match_count": int(num_inliers),
        "inlier_matches": inlier_matches_payload,
        "oracle": {
            "mode": "off",
            "evaluated": False,
            "passed": True,
            "decision": "native_no_oracle",
        },
        "native_matching_scores": native_score_summary,
        "native_matching_scores_threshold": native_threshold_summary,
        "native_matching_scores_raw": native_raw_summary,
        "geometric_score": float(geometric_score),
        "combined_score": float(combined_score),
        "match_count_term": float(match_count_term),
        "timing": {
            "time_f_s": float(fundamental_seconds),
            "time_h_s": 0.0,
            "time_scoring_s": float(scoring_seconds),
        },
    }
    return active_count, int(num_inliers), float(combined_score), direction, diagnostics


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
    for key in ("raw_matches", "inlier_matches"):
        out[key] = _swap_normalized_match_points(out.get(key))

    out["segment_scores"] = _swap_segment_scores(out.get("segment_scores"))
    out["reverse_orientation_selected"] = True
    return out


def _should_retry_reverse_native(
    num_matches: int,
    num_inliers: int,
    score: float,
    diagnostics: Dict[str, Any] | None = None,
) -> bool:
    if not NATIVE_REVERSE_RETRY_ENABLED:
        return False
    # If forward pass already satisfies strict production gate, skip reverse retry.
    # Reverse should be a rescue path for weak/fragile forward pairs only.
    try:
        if strict_geometry_edge_gate(
            num_matches=int(num_matches),
            num_inliers=int(num_inliers),
            geometric_score=float(score),
            diagnostics=diagnostics if isinstance(diagnostics, dict) else None,
            min_inliers_required=int(NATIVE_EDGE_MIN_INLIERS),
        ):
            return False
    except Exception:
        # Never let retry-guard diagnostics issues break matching flow.
        pass
    if float(score) >= float(NATIVE_REVERSE_RETRY_SCORE_THRESHOLD):
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
    if not _should_retry_reverse_native(num_matches, num_inliers, score, diagnostics=diagnostics):
        return forward_result

    forward_timing = diagnostics.get("timing") if isinstance(diagnostics, dict) and isinstance(diagnostics.get("timing"), dict) else {}
    reverse_started_at = time.perf_counter()
    rev_matches, rev_inliers, rev_score, rev_direction, rev_diag = run_fn(
        img1=img2,
        img2=img1,
        **run_kwargs,
    )
    reverse_elapsed = time.perf_counter() - reverse_started_at
    reverse_timing = rev_diag.get("timing") if isinstance(rev_diag, dict) and isinstance(rev_diag.get("timing"), dict) else {}
    forward_pair_total = float(forward_timing.get("time_pair_total_s", 0.0) or 0.0)
    reverse_pair_total = float(reverse_timing.get("time_pair_total_s", reverse_elapsed) or reverse_elapsed)
    forward_loftr = float(forward_timing.get("time_loftr_s", 0.0) or 0.0)
    reverse_loftr = float(reverse_timing.get("time_loftr_s", 0.0) or 0.0)

    def _merged_timing(base_timing: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base_timing or {})
        merged["reverse_attempted"] = True
        merged["time_reverse_retry_s"] = float(reverse_elapsed)
        merged["time_reverse_pair_total_s"] = float(reverse_pair_total)
        merged["time_loftr_forward_main_s"] = float(forward_loftr)
        merged["time_loftr_forward_reverse_s"] = float(reverse_loftr)
        merged["forward_pass_count"] = 2
        merged["time_model_load_s"] = float(
            (float(forward_timing.get("time_model_load_s", 0.0) or 0.0))
            + (float(reverse_timing.get("time_model_load_s", 0.0) or 0.0))
        )
        merged["time_resize_s"] = float(
            (float(forward_timing.get("time_resize_s", 0.0) or 0.0))
            + (float(reverse_timing.get("time_resize_s", 0.0) or 0.0))
        )
        merged["time_tensor_transfer_s"] = float(
            (float(forward_timing.get("time_tensor_transfer_s", 0.0) or 0.0))
            + (float(reverse_timing.get("time_tensor_transfer_s", 0.0) or 0.0))
        )
        merged["time_loftr_s"] = float(forward_loftr + reverse_loftr)
        merged["time_postprocess_s"] = float(
            (float(forward_timing.get("time_postprocess_s", 0.0) or 0.0))
            + (float(reverse_timing.get("time_postprocess_s", 0.0) or 0.0))
        )
        merged["time_f_s"] = float(
            (float(forward_timing.get("time_f_s", 0.0) or 0.0))
            + (float(reverse_timing.get("time_f_s", 0.0) or 0.0))
        )
        merged["time_h_s"] = float(
            (float(forward_timing.get("time_h_s", 0.0) or 0.0))
            + (float(reverse_timing.get("time_h_s", 0.0) or 0.0))
        )
        merged["time_scoring_s"] = float(
            (float(forward_timing.get("time_scoring_s", 0.0) or 0.0))
            + (float(reverse_timing.get("time_scoring_s", 0.0) or 0.0))
        )
        merged["time_pair_total_s"] = float(forward_pair_total + reverse_pair_total)
        return merged

    if not _is_better_native_result(
        (int(rev_matches), int(rev_inliers), float(rev_score)),
        (int(num_matches), int(num_inliers), float(score)),
    ):
        if isinstance(diagnostics, dict):
            diagnostics = dict(diagnostics)
            diagnostics["reverse_retry_attempted"] = True
            diagnostics["reverse_retry_selected"] = False
            diagnostics["timing"] = _merged_timing(forward_timing)
            diagnostics["timing"]["reverse_selected"] = False
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
    adjusted_diag["timing"] = _merged_timing(reverse_timing)
    adjusted_diag["timing"]["reverse_selected"] = True
    return int(rev_matches), int(rev_inliers), float(rev_score), adjusted_direction, adjusted_diag


def _match_loftr_kornia_indoor_native(
    img1: Image.Image,
    img2: Image.Image,
    confidence_threshold: float = LOFTR_NATIVE_CONFIDENCE_THRESHOLD,
    full_diagnostics: bool = False,
) -> Tuple[int, int, float, Tuple[float, float], Dict[str, Any]]:
    """Native Kornia LoFTR indoor debug path (no custom geometric scoring)."""
    total_started_at = time.perf_counter()
    checkpoint_name = "indoor"
    cache_hit = checkpoint_name in _loftr_matchers
    model_load_started_at = time.perf_counter()
    matcher = _load_loftr_checkpoint(checkpoint_name)
    model_load_seconds = time.perf_counter() - model_load_started_at
    device = next(matcher.parameters()).device

    target_long_side = max(64, int(max(DEFAULT_LOFTR_INPUT_SIZE)))
    prep_started_at = time.perf_counter()
    prep1 = _get_native_preprocessed_entry(img1, target_long_side=target_long_side)
    prep2 = _get_native_preprocessed_entry(img2, target_long_side=target_long_side)
    img1_resized = prep1["gray_resized"]
    img2_resized = prep2["gray_resized"]
    meta0 = prep1["meta"]
    meta1 = prep2["meta"]
    resize_seconds = time.perf_counter() - prep_started_at

    tensor_transfer_started_at = time.perf_counter()
    tensor1 = _build_native_tensor(img1_resized, device=device)
    tensor2 = _build_native_tensor(img2_resized, device=device)
    tensor_transfer_seconds = time.perf_counter() - tensor_transfer_started_at
    tensor_device = str(tensor1.device)
    mps_mem_before = _mps_memory_stats_mb() if device.type == "mps" else {}

    loftr_started_at = time.perf_counter()
    batch = {"image0": tensor1, "image1": tensor2}
    with torch.inference_mode():
        raw_output = matcher(batch)
        # ZJU LoFTR mutates batch in-place and returns None.
        correspondences = raw_output if isinstance(raw_output, dict) else batch
    loftr_seconds = time.perf_counter() - loftr_started_at

    points0, points1, scores = _extract_loftr_points_and_scores(correspondences)
    # Explicitly drop device tensors before CPU-heavy geometry/scoring.
    del correspondences
    del batch
    del tensor1
    del tensor2
    try:
        del raw_output
    except Exception:
        pass
    cleanup_started_at = time.perf_counter()
    mps_mem_after = _release_device_cache(device=device)
    cleanup_seconds = time.perf_counter() - cleanup_started_at

    diagnostics_started_at = time.perf_counter()
    result = _build_native_loftr_diagnostics(
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
        full_diagnostics=full_diagnostics,
    )
    diagnostics_seconds = time.perf_counter() - diagnostics_started_at
    total_seconds = time.perf_counter() - total_started_at
    num_matches, num_inliers, score, direction, diagnostics = result
    inner_timing = diagnostics.get("timing") if isinstance(diagnostics, dict) and isinstance(diagnostics.get("timing"), dict) else {}
    timing_payload = {
        "time_model_load_s": float(model_load_seconds),
        "time_resize_s": float(resize_seconds),
        "time_tensor_transfer_s": float(tensor_transfer_seconds),
        "time_loftr_s": float(loftr_seconds),
        "time_loftr_forward_main_s": float(loftr_seconds),
        "time_loftr_forward_reverse_s": 0.0,
        "time_device_cleanup_s": float(cleanup_seconds),
        "time_postprocess_s": float(diagnostics_seconds),
        "time_pair_total_s": float(total_seconds),
        "time_reverse_pair_total_s": 0.0,
        "time_f_s": float(inner_timing.get("time_f_s", 0.0) or 0.0),
        "time_h_s": float(inner_timing.get("time_h_s", 0.0) or 0.0),
        "time_scoring_s": float(inner_timing.get("time_scoring_s", 0.0) or 0.0),
        "forward_pass_count": 1,
        "reverse_attempted": False,
        "reverse_selected": False,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "preferred_device": str(_preferred_torch_device()),
        "model_device": str(device),
        "tensor_device": tensor_device,
        "model_cache_hit": bool(cache_hit),
    }
    if mps_mem_before:
        timing_payload.update({
            "mps_current_allocated_mb_before": float(mps_mem_before.get("mps_current_allocated_mb", 0.0)),
            "mps_driver_allocated_mb_before": float(mps_mem_before.get("mps_driver_allocated_mb", 0.0)),
        })
    if mps_mem_after:
        timing_payload.update({
            "mps_current_allocated_mb_after": float(mps_mem_after.get("mps_current_allocated_mb", 0.0)),
            "mps_driver_allocated_mb_after": float(mps_mem_after.get("mps_driver_allocated_mb", 0.0)),
        })
    if isinstance(diagnostics, dict):
        diagnostics["timing"] = timing_payload
    return int(num_matches), int(num_inliers), float(score), direction, diagnostics


def _match_loftr_zju_variant_debug(
    img1: Image.Image,
    img2: Image.Image,
    variant: str,
    confidence_threshold: float = LOFTR_NATIVE_CONFIDENCE_THRESHOLD,
    full_diagnostics: bool = False,
) -> Tuple[int, int, float, Tuple[float, float], Dict[str, Any]]:
    total_started_at = time.perf_counter()
    matcher, matcher_meta = _load_zju_loftr_debug_variant(variant)
    model_load_seconds = 0.0
    device = next(matcher.parameters()).device

    target_long_side = max(64, int(max(DEFAULT_LOFTR_INPUT_SIZE)))
    prep_started_at = time.perf_counter()
    prep1 = _get_native_preprocessed_entry(img1, target_long_side=target_long_side)
    prep2 = _get_native_preprocessed_entry(img2, target_long_side=target_long_side)
    img1_resized = prep1["gray_resized"]
    img2_resized = prep2["gray_resized"]
    meta0 = prep1["meta"]
    meta1 = prep2["meta"]
    resize_seconds = time.perf_counter() - prep_started_at

    tensor_transfer_started_at = time.perf_counter()
    tensor1 = _build_native_tensor(img1_resized, device=device)
    tensor2 = _build_native_tensor(img2_resized, device=device)
    tensor_transfer_seconds = time.perf_counter() - tensor_transfer_started_at
    tensor_device = str(tensor1.device)
    mps_mem_before = _mps_memory_stats_mb() if device.type == "mps" else {}

    loftr_started_at = time.perf_counter()
    batch: Dict[str, Any] = {"image0": tensor1, "image1": tensor2}
    with torch.inference_mode():
        raw_output = matcher(batch)
        correspondences = raw_output if isinstance(raw_output, dict) else batch
    loftr_seconds = time.perf_counter() - loftr_started_at

    points0, points1, scores = _extract_loftr_points_and_scores(correspondences)
    del correspondences
    del batch
    del tensor1
    del tensor2
    try:
        del raw_output
    except Exception:
        pass
    cleanup_started_at = time.perf_counter()
    mps_mem_after = _release_device_cache(device=device)
    cleanup_seconds = time.perf_counter() - cleanup_started_at

    diagnostics_started_at = time.perf_counter()
    matcher_name = f"loftr_zju_{variant}_debug"
    result = _build_native_loftr_diagnostics(
        matcher_name=matcher_name,
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
        full_diagnostics=full_diagnostics,
    )
    diagnostics_seconds = time.perf_counter() - diagnostics_started_at
    total_seconds = time.perf_counter() - total_started_at
    num_matches, num_inliers, score, direction, diagnostics = result
    inner_timing = diagnostics.get("timing") if isinstance(diagnostics, dict) and isinstance(diagnostics.get("timing"), dict) else {}

    timing_payload = {
        "time_model_load_s": float(model_load_seconds),
        "time_resize_s": float(resize_seconds),
        "time_tensor_transfer_s": float(tensor_transfer_seconds),
        "time_loftr_s": float(loftr_seconds),
        "time_loftr_forward_main_s": float(loftr_seconds),
        "time_loftr_forward_reverse_s": 0.0,
        "time_device_cleanup_s": float(cleanup_seconds),
        "time_postprocess_s": float(diagnostics_seconds),
        "time_pair_total_s": float(total_seconds),
        "time_reverse_pair_total_s": 0.0,
        "time_f_s": float(inner_timing.get("time_f_s", 0.0) or 0.0),
        "time_h_s": float(inner_timing.get("time_h_s", 0.0) or 0.0),
        "time_scoring_s": float(inner_timing.get("time_scoring_s", 0.0) or 0.0),
        "forward_pass_count": 1,
        "reverse_attempted": False,
        "reverse_selected": False,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "preferred_device": str(_preferred_torch_device()),
        "model_device": str(device),
        "tensor_device": tensor_device,
        "model_cache_hit": True,
    }
    if mps_mem_before:
        timing_payload.update({
            "mps_current_allocated_mb_before": float(mps_mem_before.get("mps_current_allocated_mb", 0.0)),
            "mps_driver_allocated_mb_before": float(mps_mem_before.get("mps_driver_allocated_mb", 0.0)),
        })
    if mps_mem_after:
        timing_payload.update({
            "mps_current_allocated_mb_after": float(mps_mem_after.get("mps_current_allocated_mb", 0.0)),
            "mps_driver_allocated_mb_after": float(mps_mem_after.get("mps_driver_allocated_mb", 0.0)),
        })

    diagnostics["zju_variant"] = str(variant)
    diagnostics["zju_loader"] = "strict_repo_debug"
    diagnostics["zju_checkpoint_path"] = matcher_meta.get("ckpt_path")
    diagnostics["zju_repo_dir"] = matcher_meta.get("repo_dir")
    diagnostics["zju_match_type"] = "native_repo"
    diagnostics["zju_model_class"] = matcher_meta.get("model_class")
    diagnostics["timing"] = timing_payload
    return int(num_matches), int(num_inliers), float(score), direction, diagnostics


def _match_matchformer_variant_debug(
    img1: Image.Image,
    img2: Image.Image,
    variant: str,
    confidence_threshold: float = LOFTR_NATIVE_CONFIDENCE_THRESHOLD,
    full_diagnostics: bool = False,
) -> Tuple[int, int, float, Tuple[float, float], Dict[str, Any]]:
    total_started_at = time.perf_counter()
    cache_hit = str(variant) in _matchformer_matchers
    model_load_started_at = time.perf_counter()
    matcher, matcher_meta = _load_matchformer_debug_variant(variant)
    model_load_seconds = time.perf_counter() - model_load_started_at
    device = next(matcher.parameters()).device

    target_long_side = max(64, int(max(DEFAULT_LOFTR_INPUT_SIZE)))
    prep_started_at = time.perf_counter()
    prep1 = _get_native_preprocessed_entry(img1, target_long_side=target_long_side)
    prep2 = _get_native_preprocessed_entry(img2, target_long_side=target_long_side)
    img1_resized = prep1["gray_resized"]
    img2_resized = prep2["gray_resized"]
    meta0 = prep1["meta"]
    meta1 = prep2["meta"]
    resize_seconds = time.perf_counter() - prep_started_at

    tensor_transfer_started_at = time.perf_counter()
    tensor1 = _build_native_tensor(img1_resized, device=device)
    tensor2 = _build_native_tensor(img2_resized, device=device)
    tensor_transfer_seconds = time.perf_counter() - tensor_transfer_started_at
    tensor_device = str(tensor1.device)
    mps_mem_before = _mps_memory_stats_mb() if device.type == "mps" else {}

    loftr_started_at = time.perf_counter()
    batch: Dict[str, Any] = {"image0": tensor1, "image1": tensor2}
    with torch.inference_mode():
        raw_output = matcher(batch)
        correspondences = raw_output if isinstance(raw_output, dict) else batch
    loftr_seconds = time.perf_counter() - loftr_started_at

    points0, points1, scores = _extract_loftr_points_and_scores(correspondences)
    del correspondences
    del batch
    del tensor1
    del tensor2
    try:
        del raw_output
    except Exception:
        pass
    cleanup_started_at = time.perf_counter()
    mps_mem_after = _release_device_cache(device=device)
    cleanup_seconds = time.perf_counter() - cleanup_started_at

    diagnostics_started_at = time.perf_counter()
    matcher_name = f"matchformer_{variant}_debug"
    result = _build_native_loftr_diagnostics(
        matcher_name=matcher_name,
        checkpoint=f"matchformer:{variant}",
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
        full_diagnostics=full_diagnostics,
    )
    diagnostics_seconds = time.perf_counter() - diagnostics_started_at
    total_seconds = time.perf_counter() - total_started_at
    num_matches, num_inliers, score, direction, diagnostics = result
    inner_timing = diagnostics.get("timing") if isinstance(diagnostics, dict) and isinstance(diagnostics.get("timing"), dict) else {}
    timing_payload = {
        "time_model_load_s": float(model_load_seconds),
        "time_resize_s": float(resize_seconds),
        "time_tensor_transfer_s": float(tensor_transfer_seconds),
        "time_loftr_s": float(loftr_seconds),
        "time_loftr_forward_main_s": float(loftr_seconds),
        "time_loftr_forward_reverse_s": 0.0,
        "time_device_cleanup_s": float(cleanup_seconds),
        "time_postprocess_s": float(diagnostics_seconds),
        "time_pair_total_s": float(total_seconds),
        "time_reverse_pair_total_s": 0.0,
        "time_f_s": float(inner_timing.get("time_f_s", 0.0) or 0.0),
        "time_h_s": float(inner_timing.get("time_h_s", 0.0) or 0.0),
        "time_scoring_s": float(inner_timing.get("time_scoring_s", 0.0) or 0.0),
        "forward_pass_count": 1,
        "reverse_attempted": False,
        "reverse_selected": False,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "preferred_device": str(_preferred_torch_device()),
        "model_device": str(device),
        "tensor_device": tensor_device,
        "model_cache_hit": bool(cache_hit),
    }
    if mps_mem_before:
        timing_payload.update({
            "mps_current_allocated_mb_before": float(mps_mem_before.get("mps_current_allocated_mb", 0.0)),
            "mps_driver_allocated_mb_before": float(mps_mem_before.get("mps_driver_allocated_mb", 0.0)),
        })
    if mps_mem_after:
        timing_payload.update({
            "mps_current_allocated_mb_after": float(mps_mem_after.get("mps_current_allocated_mb", 0.0)),
            "mps_driver_allocated_mb_after": float(mps_mem_after.get("mps_driver_allocated_mb", 0.0)),
        })
    diagnostics["matchformer_variant"] = str(variant)
    diagnostics["matchformer_loader"] = "strict_repo_debug"
    diagnostics["matchformer_checkpoint_path"] = matcher_meta.get("ckpt_path")
    diagnostics["matchformer_repo_dir"] = matcher_meta.get("repo_dir")
    diagnostics["matchformer_model_class"] = matcher_meta.get("model_class")
    diagnostics["timing"] = timing_payload
    return int(num_matches), int(num_inliers), float(score), direction, diagnostics


def _match_roma_v2_debug(
    img1: Image.Image,
    img2: Image.Image,
    confidence_threshold: float = LOFTR_NATIVE_CONFIDENCE_THRESHOLD,
    full_diagnostics: bool = False,
) -> Tuple[int, int, float, Tuple[float, float], Dict[str, Any]]:
    total_started_at = time.perf_counter()
    cache_hit = "outdoor" in _roma_v2_matchers
    model_load_started_at = time.perf_counter()
    matcher, matcher_meta = _load_roma_v2_debug_matcher()
    model_load_seconds = time.perf_counter() - model_load_started_at
    model_device = torch.device(str(matcher_meta.get("device", "cpu")))

    width0, height0 = img1.size
    width1, height1 = img2.size

    resize_seconds = 0.0
    tensor_transfer_seconds = 0.0
    tensor_device = str(model_device)
    loftr_started_at = time.perf_counter()
    try:
        warp, certainty = matcher.match(
            img1.convert("RGB"),
            img2.convert("RGB"),
            device=str(model_device),
        )
    except TypeError:
        warp, certainty = matcher.match(
            img1.convert("RGB"),
            img2.convert("RGB"),
        )
    except Exception as exc:
        raise RuntimeError(f"RoMa-v2 debug matcher forward failed: {exc}") from exc
    try:
        sampled_matches, sampled_certainty = matcher.sample(warp, certainty)
    except Exception as exc:
        raise RuntimeError(f"RoMa-v2 debug matcher sampling failed: {exc}") from exc
    try:
        points0_raw, points1_raw = matcher.to_pixel_coordinates(
            sampled_matches,
            int(height0),
            int(width0),
            int(height1),
            int(width1),
        )
    except Exception as exc:
        raise RuntimeError(f"RoMa-v2 debug matcher coordinate conversion failed: {exc}") from exc
    loftr_seconds = time.perf_counter() - loftr_started_at

    points0 = _as_xy_points(points0_raw)
    points1 = _as_xy_points(points1_raw)
    scores = _as_score_vector(sampled_certainty)

    cleanup_started_at = time.perf_counter()
    mps_mem_after = _release_device_cache(device=model_device)
    cleanup_seconds = time.perf_counter() - cleanup_started_at

    diagnostics_started_at = time.perf_counter()
    result = _build_native_loftr_diagnostics(
        matcher_name="roma_v2_debug",
        checkpoint="roma_v2:outdoor",
        confidence_threshold=float(confidence_threshold),
        points0=points0,
        points1=points1,
        scores=scores,
        width0=int(width0),
        height0=int(height0),
        width1=int(width1),
        height1=int(height1),
        input_w=int(width0),
        input_h=int(height0),
        full_diagnostics=full_diagnostics,
    )
    diagnostics_seconds = time.perf_counter() - diagnostics_started_at
    total_seconds = time.perf_counter() - total_started_at
    num_matches, num_inliers, score, direction, diagnostics = result
    inner_timing = diagnostics.get("timing") if isinstance(diagnostics, dict) and isinstance(diagnostics.get("timing"), dict) else {}

    timing_payload = {
        "time_model_load_s": float(model_load_seconds),
        "time_resize_s": float(resize_seconds),
        "time_tensor_transfer_s": float(tensor_transfer_seconds),
        "time_loftr_s": float(loftr_seconds),
        "time_loftr_forward_main_s": float(loftr_seconds),
        "time_loftr_forward_reverse_s": 0.0,
        "time_device_cleanup_s": float(cleanup_seconds),
        "time_postprocess_s": float(diagnostics_seconds),
        "time_pair_total_s": float(total_seconds),
        "time_reverse_pair_total_s": 0.0,
        "time_f_s": float(inner_timing.get("time_f_s", 0.0) or 0.0),
        "time_h_s": float(inner_timing.get("time_h_s", 0.0) or 0.0),
        "time_scoring_s": float(inner_timing.get("time_scoring_s", 0.0) or 0.0),
        "forward_pass_count": 1,
        "reverse_attempted": False,
        "reverse_selected": False,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "preferred_device": str(_preferred_torch_device()),
        "model_device": str(model_device),
        "tensor_device": tensor_device,
        "model_cache_hit": bool(cache_hit),
    }
    if mps_mem_after:
        timing_payload.update({
            "mps_current_allocated_mb_after": float(mps_mem_after.get("mps_current_allocated_mb", 0.0)),
            "mps_driver_allocated_mb_after": float(mps_mem_after.get("mps_driver_allocated_mb", 0.0)),
        })

    diagnostics["roma_variant"] = "outdoor"
    diagnostics["roma_loader"] = "romatch_debug"
    diagnostics["roma_model_class"] = matcher_meta.get("model_class")
    diagnostics["timing"] = timing_payload
    return int(num_matches), int(num_inliers), float(score), direction, diagnostics


def match_image_pair(
    img1: Image.Image,
    img2: Image.Image,
    use_orb_prefilter: bool = False,
    return_diagnostics: bool = False,
    debug_options: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int, float, Tuple[float, float]] | Tuple[int, int, float, Tuple[float, float], Dict[str, Any]]:
    """Match two images using the production native LoFTR pipeline.

    Args:
        img1: First PIL Image
        img2: Second PIL Image
        use_orb_prefilter: Deprecated compatibility flag (ignored).

    Returns:
        Tuple of (num_matches, num_inliers, overlap_score, direction_vector).
        When return_diagnostics=True, appends a diagnostics dict.
        direction_vector is (dx, dy) showing how content shifted from img1 to img2.
    """
    _ = use_orb_prefilter  # compatibility-only; ORB path removed from production flow
    # Normalize orientation only when EXIF indicates rotation.
    # Avoid unconditional copies; they destroy image-object cache hit rate.
    try:
        orientation1 = int((img1.getexif() or {}).get(274, 1))
    except Exception:
        orientation1 = 1
    try:
        orientation2 = int((img2.getexif() or {}).get(274, 1))
    except Exception:
        orientation2 = 1
    if orientation1 not in (0, 1):
        img1 = ImageOps.exif_transpose(img1)
    if orientation2 not in (0, 1):
        img2 = ImageOps.exif_transpose(img2)

    options = debug_options or {}
    matcher_preference = str(options.get("matcher", "current")).strip().lower()
    if matcher_preference in {"", "current", "default"}:
        matcher_preference = DEFAULT_PRODUCTION_MATCHER
    full_diagnostics = bool(options.get("full_diagnostics", False))
    requested_threshold = options.get("confidence_threshold")
    default_threshold = LOFTR_NATIVE_CONFIDENCE_THRESHOLD
    confidence_threshold = default_threshold
    if requested_threshold is not None:
        try:
            confidence_threshold = float(requested_threshold)
        except (TypeError, ValueError):
            confidence_threshold = default_threshold
    confidence_threshold = float(np.clip(confidence_threshold, 0.1, 1.0))

    allowed_matchers = {
        "loftr_kornia_indoor_native",
        "loftr_zju_indoor_ds_debug",
        "loftr_zju_indoor_ot_debug",
        "roma_v2_debug",
        "matchformer_indoor_debug",
        "matchformer_outdoor_debug",
    }
    if matcher_preference not in allowed_matchers:
        raise ValueError(
            f"Unsupported matcher '{matcher_preference}'. "
            "Supported matchers: 'loftr_kornia_indoor_native', "
            "'loftr_zju_indoor_ds_debug', 'loftr_zju_indoor_ot_debug', "
            "'roma_v2_debug', 'matchformer_indoor_debug', 'matchformer_outdoor_debug'."
        )
    if matcher_preference == "loftr_zju_indoor_ds_debug":
        result = _match_loftr_zju_variant_debug(
            img1=img1,
            img2=img2,
            variant="indoor_ds",
            confidence_threshold=confidence_threshold,
            full_diagnostics=full_diagnostics,
        )
    elif matcher_preference == "loftr_zju_indoor_ot_debug":
        result = _match_loftr_zju_variant_debug(
            img1=img1,
            img2=img2,
            variant="indoor_ot",
            confidence_threshold=confidence_threshold,
            full_diagnostics=full_diagnostics,
        )
    elif matcher_preference == "roma_v2_debug":
        result = _match_roma_v2_debug(
            img1=img1,
            img2=img2,
            confidence_threshold=confidence_threshold,
            full_diagnostics=full_diagnostics,
        )
    elif matcher_preference == "matchformer_indoor_debug":
        result = _match_matchformer_variant_debug(
            img1=img1,
            img2=img2,
            variant="indoor",
            confidence_threshold=confidence_threshold,
            full_diagnostics=full_diagnostics,
        )
    elif matcher_preference == "matchformer_outdoor_debug":
        result = _match_matchformer_variant_debug(
            img1=img1,
            img2=img2,
            variant="outdoor",
            confidence_threshold=confidence_threshold,
            full_diagnostics=full_diagnostics,
        )
    else:
        result = _match_loftr_kornia_indoor_native(
            img1=img1,
            img2=img2,
            confidence_threshold=confidence_threshold,
            full_diagnostics=full_diagnostics,
        )
        result = _maybe_retry_reverse_native(
            img1=img1,
            img2=img2,
            forward_result=result,
            run_fn=_match_loftr_kornia_indoor_native,
            run_kwargs={
                "confidence_threshold": confidence_threshold,
                "full_diagnostics": full_diagnostics,
            },
        )
    if return_diagnostics:
        return result
    return result[0], result[1], result[2], result[3]

# ============================================================================
# GRAPH-BASED CLUSTERING: DINOv2 proposes edges, geometry verifies
# ============================================================================


def _connected_component_nodes(edge_mask: np.ndarray, start_idx: int) -> np.ndarray:
    """Return sorted node indices in start_idx component using iterative DFS."""
    n = edge_mask.shape[0]
    if start_idx < 0 or start_idx >= n:
        return np.empty((0,), dtype=np.int32)

    visited = np.zeros((n,), dtype=bool)
    stack = [int(start_idx)]
    visited[start_idx] = True
    while stack:
        node = stack.pop()
        next_nodes = np.flatnonzero(edge_mask[node] & (~visited))
        if next_nodes.size == 0:
            continue
        visited[next_nodes] = True
        stack.extend(int(idx) for idx in next_nodes)
    return np.flatnonzero(visited).astype(np.int32, copy=False)


def _find_undirected_bridge_edges(edge_mask: np.ndarray) -> List[Tuple[int, int]]:
    """Find bridge edges in an undirected graph represented by a boolean adjacency matrix."""
    n = edge_mask.shape[0]
    neighbors: List[List[int]] = [np.flatnonzero(edge_mask[i]).astype(np.int32).tolist() for i in range(n)]
    disc = np.full((n,), -1, dtype=np.int32)
    low = np.full((n,), -1, dtype=np.int32)
    parent = np.full((n,), -1, dtype=np.int32)
    bridges: List[Tuple[int, int]] = []
    time_idx = 0

    def dfs(node: int) -> None:
        nonlocal time_idx
        disc[node] = time_idx
        low[node] = time_idx
        time_idx += 1
        for nbr in neighbors[node]:
            if disc[nbr] == -1:
                parent[nbr] = node
                dfs(nbr)
                low[node] = min(low[node], low[nbr])
                if low[nbr] > disc[node]:
                    bridges.append((min(node, nbr), max(node, nbr)))
            elif nbr != parent[node]:
                low[node] = min(low[node], disc[nbr])

    for node in range(n):
        if disc[node] == -1:
            dfs(node)
    return bridges


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
    edge_mask = adjacency > OVERLAP_THRESHOLD
    pass_idx = 0
    total_pruned = 0

    while True:
        bridge_edges = _find_undirected_bridge_edges(edge_mask)
        if not bridge_edges:
            break

        pruned_this_pass = 0
        pass_idx += 1
        for i, j in bridge_edges:
            if edge_has_geometry[i, j]:
                continue
            if not edge_mask[i, j]:
                continue

            # Temporarily remove candidate bridge and evaluate cross-support.
            edge_mask[i, j] = False
            edge_mask[j, i] = False

            comp_i = _connected_component_nodes(edge_mask, i)
            if comp_i.size == 0 or np.any(comp_i == j):
                edge_mask[i, j] = True
                edge_mask[j, i] = True
                continue

            comp_j = _connected_component_nodes(edge_mask, j)
            if comp_j.size == 0:
                edge_mask[i, j] = True
                edge_mask[j, i] = True
                continue

            if (comp_i.size + comp_j.size) < 3:
                edge_mask[i, j] = True
                edge_mask[j, i] = True
                continue

            cross_block = similarity[np.ix_(comp_i, comp_j)]
            max_cross_sem = float(np.max(cross_block)) if cross_block.size else 0.0
            if max_cross_sem < SEMANTIC_BRIDGE_SUPPORT_THRESHOLD:
                adjacency[i, j] = 0.0
                adjacency[j, i] = 0.0
                pruned_this_pass += 1
                total_pruned += 1
                logger.info(
                    "Pruned semantic bridge %s <-> %s (max cross semantic=%.3f < %.2f)",
                    photo_ids[i],
                    photo_ids[j],
                    max_cross_sem,
                    SEMANTIC_BRIDGE_SUPPORT_THRESHOLD,
                )
                # Keep edge removed in edge_mask.
            else:
                edge_mask[i, j] = True
                edge_mask[j, i] = True

        if pruned_this_pass <= 0:
            break
        logger.info(
            "Semantic-bridge prune pass %s removed %s edges (total=%s)",
            pass_idx,
            pruned_this_pass,
            total_pruned,
        )


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

    listing_started_at = time.perf_counter()
    n = len(images)
    logger.info(f"Graph-based clustering for {n} photos (k={k})")

    # Track similarity data for database storage
    similarity_records = []  # Will store dicts for batch insert

    if n <= 1:
        clusters = [photo_ids] if photo_ids else []
        if return_metadata:
            return clusters, {"duplicate_of_map": {}}
        return clusters

    # Runtime environment snapshot (critical for performance diagnosis).
    matcher = _load_loftr_checkpoint("indoor")
    matcher_device = str(next(matcher.parameters()).device)
    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    dino_device = str(next(_dinov2_model.parameters()).device) if _dinov2_model is not None else "not_loaded"
    logger.info(
        "Runtime device: cuda_available=%s mps_available=%s preferred_device=%s matcher_device=%s dino_device=%s",
        cuda_available,
        mps_available,
        str(_preferred_torch_device()),
        matcher_device,
        dino_device,
    )

    # Aggregate timing metrics for structured performance reporting.
    stage_timers: Dict[str, float] = {
        "time_dino_total": 0.0,
        "time_candidate_generation": 0.0,
        "time_loftr_total": 0.0,
        "time_f_total": 0.0,
        "time_h_total": 0.0,
        "time_scoring_total": 0.0,
        "time_resize_total": 0.0,
        "time_tensor_transfer_total": 0.0,
        "time_device_cleanup_total": 0.0,
        "time_postprocess_total": 0.0,
    }
    pair_timing_count = 0

    # -------------------------------------------------------------------------
    # Stage 1: Compute DINOv2 embeddings and build candidate graph
    # -------------------------------------------------------------------------
    logger.info("Stage 1: Computing DINOv2 embeddings...")
    stage1_started_at = time.perf_counter()
    embeddings = compute_dinov2_embeddings(images)
    embeddings = normalize(embeddings)  # For cosine similarity
    stage_timers["time_dino_total"] = time.perf_counter() - stage1_started_at
    if _dinov2_model is not None:
        logger.info("DINO runtime device after load: %s", str(next(_dinov2_model.parameters()).device))

    # Compute similarity matrix
    similarity = embeddings @ embeddings.T
    pre_pipeline_duplicate_of_map: Dict[int, int] = {}

    # -------------------------------------------------------------------------
    # Stage 1a: Pre-pipeline duplicate pruning (same-room, obvious 95%+ matches)
    # -------------------------------------------------------------------------
    if room_labels is not None and len(room_labels) == n:
        (
            images,
            photo_ids,
            room_labels,
            embeddings,
            similarity,
            pre_pipeline_duplicate_of_map,
        ) = _prefilter_obvious_room_duplicates(
            images=images,
            photo_ids=photo_ids,
            room_labels=room_labels,
            embeddings=embeddings,
            similarity=similarity,
            similarity_records=similarity_records,
        )
        n = len(photo_ids)
        if n <= 1:
            clusters = [photo_ids] if photo_ids else []
            if return_metadata:
                return clusters, {
                    "duplicate_of_map": pre_pipeline_duplicate_of_map,
                    "duplicates_dropped": not KEEP_DUPLICATE_SINGLETON_CLUSTERS,
                }
            return clusters
    else:
        logger.info(
            "Pre-dedup skipped: room_labels unavailable or mismatched length (labels=%s, photos=%s)",
            len(room_labels) if room_labels is not None else None,
            n,
        )

    # -------------------------------------------------------------------------
    # Stage 1b: Build candidate pairs from DINOv2 top-K
    # -------------------------------------------------------------------------
    stage1b1c_started_at = time.perf_counter()
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
    stage_timers["time_candidate_generation"] = time.perf_counter() - stage1b1c_started_at

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
    temporal_geo_checks = 0
    temporal_geo_seconds = 0.0
    stage2a_started_at = time.perf_counter()

    for i, j in sorted(temporal_pairs):
        sem_sim = similarity[i, j]
        temporal_dist = abs(j - i)

        # Check if rooms are different
        room_i = room_labels[i] if room_labels else None
        room_j = room_labels[j] if room_labels else None
        is_cross_room = rooms_are_different(room_i, room_j)

        # In strict room mode, block cross-room temporal pairs except adjacent
        # social-family pairs with strong semantic affinity (likely label error).
        strict_recheck_allowed = (
            is_cross_room
            and STRICT_ROOM_LABEL_BLOCK
            and allow_strict_cross_room_temporal_recheck(
                room1=room_i,
                room2=room_j,
                temporal_dist=temporal_dist,
                sem_sim=float(sem_sim),
            )
        )
        # Non-strict mode keeps the legacy behavior (skip only non-adjacent cross-room pairs).
        if is_cross_room and ((STRICT_ROOM_LABEL_BLOCK and not strict_recheck_allowed) or temporal_dist > 1):
            if STRICT_ROOM_LABEL_BLOCK:
                logger.info(
                    f"  Temporal {photo_ids[i]} <-> {photo_ids[j]}: "
                    f"SKIP - strict room block ({room_i} vs {room_j}), dist={temporal_dist}"
                )
            else:
                logger.info(
                    f"  Temporal {photo_ids[i]} <-> {photo_ids[j]}: "
                    f"SKIP - different rooms ({room_i} vs {room_j}), dist={temporal_dist}"
                )
            _append_similarity_record(
                similarity_records=similarity_records,
                photo_ids=photo_ids,
                left_idx=i,
                right_idx=j,
                pair_source="temporal_cross_room",
                dinov2_similarity=float(sem_sim),
                is_connected=False,
            )
            continue
        if strict_recheck_allowed:
            logger.info(
                "  Temporal %s <-> %s: strict cross-room recheck enabled (%s vs %s, sem=%.3f)",
                photo_ids[i],
                photo_ids[j],
                room_i,
                room_j,
                sem_sim,
            )

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
            pair_started_at = time.perf_counter()
            num_matches, num_inliers, geo_score, direction, diagnostics = match_image_pair(
                images[i],
                images[j],
                return_diagnostics=True,
            )
            temporal_geo_checks += 1
            pair_elapsed = time.perf_counter() - pair_started_at
            temporal_geo_seconds += pair_elapsed
            pair_metrics = _extract_pair_runtime_metrics(diagnostics, fallback_pair_time_s=pair_elapsed)
            _accumulate_pair_runtime(stage_timers, pair_metrics)
            pair_timing_count += 1
            _log_pair_timing(
                phase="2a",
                left_photo_id=photo_ids[i],
                right_photo_id=photo_ids[j],
                pair_metrics=pair_metrics,
                num_matches=num_matches,
                num_inliers=num_inliers,
                diagnostics=diagnostics,
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
            pair_started_at = time.perf_counter()
            num_matches, num_inliers, geo_score, direction, diagnostics = match_image_pair(
                images[i],
                images[j],
                return_diagnostics=True,
            )
            temporal_geo_checks += 1
            pair_elapsed = time.perf_counter() - pair_started_at
            temporal_geo_seconds += pair_elapsed
            pair_metrics = _extract_pair_runtime_metrics(diagnostics, fallback_pair_time_s=pair_elapsed)
            _accumulate_pair_runtime(stage_timers, pair_metrics)
            pair_timing_count += 1
            _log_pair_timing(
                phase="2a",
                left_photo_id=photo_ids[i],
                right_photo_id=photo_ids[j],
                pair_metrics=pair_metrics,
                num_matches=num_matches,
                num_inliers=num_inliers,
                diagnostics=diagnostics,
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
            elif (not GEOMETRY_ONLY_CLUSTER_MEMBERSHIP) and (not STRICT_ROOM_LABEL_BLOCK) and temporal_dist == 1 and is_cross_room:
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
        _append_similarity_record(
            similarity_records=similarity_records,
            photo_ids=photo_ids,
            left_idx=i,
            right_idx=j,
            pair_source=pair_source,
            dinov2_similarity=float(sem_sim),
            geometric_matches=num_matches,
            geometric_inliers=num_inliers,
            geometric_score=float(geo_score) if geo_score is not None else None,
            direction=direction,
            is_connected=is_matched,
            diagnostics=diagnostics,
        )

    logger.info(f"Stage 2a: {temporal_matched}/{len(temporal_pairs)} temporal pairs matched "
               f"(semantic-only={temporal_semantic_only}, geometric={temporal_geometric})")
    total_stage2a_seconds = time.perf_counter() - stage2a_started_at
    avg_geo_ms = (temporal_geo_seconds / temporal_geo_checks * 1000.0) if temporal_geo_checks > 0 else 0.0
    logger.info(
        "Stage 2a timing: total=%.2fs, geo_checks=%s, geo_time=%.2fs, avg_geo=%.1fms",
        total_stage2a_seconds,
        temporal_geo_checks,
        temporal_geo_seconds,
        avg_geo_ms,
    )

    # -------------------------------------------------------------------------
    # Stage 2b: Non-temporal pairs - use GEOMETRIC verification (pixel overlap)
    # -------------------------------------------------------------------------
    # Only check semantic pairs that aren't already covered by temporal
    geometric_pairs = semantic_pairs - temporal_pairs
    logger.info(f"Stage 2b: Geometric verification on {len(geometric_pairs)} non-temporal pairs...")
    geometric_matched = 0
    stage2b_started_at = time.perf_counter()
    stage2b_geo_checks = 0
    stage2b_geo_seconds = 0.0

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
            _append_similarity_record(
                similarity_records=similarity_records,
                photo_ids=photo_ids,
                left_idx=i,
                right_idx=j,
                pair_source="dinov2_topk_cross_room",
                dinov2_similarity=float(sem_sim),
                is_connected=False,
            )
            continue

        # Skip pairs with very low semantic similarity (obviously unrelated)
        if sem_sim < MIN_SEMANTIC_FOR_GEOMETRIC:
            logger.info(f"  [{idx+1}/{len(geometric_pairs)}] SKIP {photo_ids[i]} <-> {photo_ids[j]} "
                       f"(DINOv2 similarity={sem_sim:.3f} < {MIN_SEMANTIC_FOR_GEOMETRIC})")
            skipped_low_semantic += 1
            # Still record for database but mark as not matched
            _append_similarity_record(
                similarity_records=similarity_records,
                photo_ids=photo_ids,
                left_idx=i,
                right_idx=j,
                pair_source="dinov2_topk_skipped",
                dinov2_similarity=float(sem_sim),
                is_connected=False,
            )
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
                _append_similarity_record(
                    similarity_records=similarity_records,
                    photo_ids=photo_ids,
                    left_idx=i,
                    right_idx=j,
                    pair_source="dinov2_topk_skipped",
                    dinov2_similarity=float(sem_sim),
                    is_connected=False,
                )
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
        pair_started_at = time.perf_counter()
        num_matches, num_inliers, score, direction, diagnostics = match_image_pair(
            images[i],
            images[j],
            return_diagnostics=True,
        )
        pair_elapsed = time.perf_counter() - pair_started_at
        stage2b_geo_checks += 1
        stage2b_geo_seconds += pair_elapsed
        pair_metrics = _extract_pair_runtime_metrics(diagnostics, fallback_pair_time_s=pair_elapsed)
        _accumulate_pair_runtime(stage_timers, pair_metrics)
        pair_timing_count += 1
        _log_pair_timing(
            phase="2b",
            left_photo_id=photo_ids[i],
            right_photo_id=photo_ids[j],
            pair_metrics=pair_metrics,
            num_matches=num_matches,
            num_inliers=num_inliers,
            diagnostics=diagnostics,
        )

        is_matched = strict_geometry_edge_gate(
            num_matches=num_matches,
            num_inliers=num_inliers,
            geometric_score=score,
            diagnostics=diagnostics,
            min_inliers_required=effective_min_inliers,
        )
        if is_matched and position_gap >= POSITION_GAP_THRESHOLD and float(score) < float(MIN_SCORE_FAR_APART):
            is_matched = False
            logger.info(
                "    ✗ Rejected far-gap geometric link (gap=%s, score=%.3f < %.2f)",
                position_gap,
                float(score),
                float(MIN_SCORE_FAR_APART),
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
            room_i_norm = normalize_room_label(room_i)
            room_j_norm = normalize_room_label(room_j)
            same_room_label = (
                room_i_norm != ""
                and room_i_norm != "unknown"
                and room_i_norm == room_j_norm
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
        _append_similarity_record(
            similarity_records=similarity_records,
            photo_ids=photo_ids,
            left_idx=i,
            right_idx=j,
            pair_source=pair_source,
            dinov2_similarity=float(sem_sim),
            geometric_matches=num_matches,
            geometric_inliers=num_inliers,
            geometric_score=float(score) if score is not None else None,
            direction=direction,
            is_connected=is_matched,
            diagnostics=diagnostics,
        )

    logger.info(
        "Stage 2b: %s/%s non-temporal pairs matched (%s cross-room, %s low-semantic, %s semantic recoveries)",
        geometric_matched,
        len(geometric_pairs),
        skipped_cross_room,
        skipped_low_semantic,
        semantic_recovered,
    )
    total_stage2b_seconds = time.perf_counter() - stage2b_started_at
    avg_stage2b_geo_ms = (stage2b_geo_seconds / stage2b_geo_checks * 1000.0) if stage2b_geo_checks > 0 else 0.0
    logger.info(
        "Stage 2b timing: total=%.2fs, geo_checks=%s, geo_time=%.2fs, avg_geo=%.1fms",
        total_stage2b_seconds,
        stage2b_geo_checks,
        stage2b_geo_seconds,
        avg_stage2b_geo_ms,
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
    pair_metrics_lookup = _build_similarity_lookup(similarity_records)

    final_clusters = []
    duplicate_of_map: Dict[int, int] = dict(pre_pipeline_duplicate_of_map)
    for cluster in ordered_clusters:
        # Deduplicate obvious same-angle shots.
        split_clusters, cluster_duplicate_map = deduplicate_and_split_cluster(
            cluster,
            photo_ids,
            embeddings,
            adjacency,
            max_size=max_cluster_size,
            keep_duplicate_singletons=KEEP_DUPLICATE_SINGLETON_CLUSTERS,
            room_labels=room_labels,
            pair_metrics_lookup=pair_metrics_lookup,
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

    total_listing_time = time.perf_counter() - listing_started_at
    time_loftr_per_pair_avg = (
        stage_timers["time_loftr_total"] / pair_timing_count
        if pair_timing_count > 0
        else 0.0
    )
    logger.info(
        "PERF_SUMMARY job_id=%s photos=%s pairs_processed=%s time_dino_total=%.3fs "
        "time_candidate_generation=%.3fs time_loftr_total=%.3fs time_loftr_per_pair_avg=%.3fs "
        "time_f_total=%.3fs time_h_total=%.3fs time_scoring_total=%.3fs "
        "time_resize_total=%.3fs time_tensor_transfer_total=%.3fs time_device_cleanup_total=%.3fs time_postprocess_total=%.3fs "
        "total_listing_time=%.3fs",
        job_id,
        n,
        pair_timing_count,
        stage_timers["time_dino_total"],
        stage_timers["time_candidate_generation"],
        stage_timers["time_loftr_total"],
        time_loftr_per_pair_avg,
        stage_timers["time_f_total"],
        stage_timers["time_h_total"],
        stage_timers["time_scoring_total"],
        stage_timers["time_resize_total"],
        stage_timers["time_tensor_transfer_total"],
        stage_timers["time_device_cleanup_total"],
        stage_timers["time_postprocess_total"],
        total_listing_time,
    )

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
        max_cluster_size=2,  # Hard cap: keep best transition pairs only
        db_session=db_session,
        job_id=job_id,
        room_labels=room_labels,
        return_metadata=return_metadata,
    )
