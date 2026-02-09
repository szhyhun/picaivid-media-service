"""Photo quality scoring for Phase 1."""
import logging
from typing import List

import numpy as np
from PIL import Image

from app.db.models import JobPhoto
from app.services.storage.s3_client import s3_client

logger = logging.getLogger(__name__)


def compute_photo_scores(photos: List[JobPhoto]) -> None:
    """Compute quality scores for all photos.

    Scores are based on:
    - Sharpness (Laplacian variance)
    - Exposure (histogram analysis)
    - Composition (rule of thirds, centered subject)

    Args:
        photos: List of JobPhoto instances to score
    """
    logger.info(f"Computing quality scores for {len(photos)} photos")

    for photo in photos:
        try:
            image = s3_client.download_image(photo.s3_uri)
            _score_photo(photo, image)
        except Exception as e:
            logger.warning(f"Failed to score photo {photo.id}: {e}")
            # Set default scores
            photo.sharpness = 0.5
            photo.exposure_score = 0.5
            photo.composition_score = 0.5
            photo.base_score = 0.5
            photo.final_score = 0.5


def _score_photo(photo: JobPhoto, image: Image.Image) -> None:
    """Score a single photo.

    Args:
        photo: JobPhoto to update
        image: PIL Image
    """
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_array = np.array(image)

    # Compute individual scores
    photo.sharpness = _compute_sharpness(img_array)
    photo.exposure_score = _compute_exposure_score(img_array)
    photo.composition_score = _compute_composition_score(img_array)

    # Compute base score (weighted average)
    photo.base_score = (
        photo.sharpness * 0.4 +
        photo.exposure_score * 0.3 +
        photo.composition_score * 0.3
    )

    # Apply manual bonuses from metadata
    photo.final_score = _apply_manual_bonuses(photo)

    logger.debug(
        f"Photo {photo.id} scores: "
        f"sharp={photo.sharpness:.2f} exp={photo.exposure_score:.2f} "
        f"comp={photo.composition_score:.2f} final={photo.final_score:.2f}"
    )


def _compute_sharpness(img_array: np.ndarray) -> float:
    """Compute sharpness using Laplacian variance.

    Args:
        img_array: Numpy array (H, W, C)

    Returns:
        Sharpness score 0-1
    """
    # Convert to grayscale
    gray = np.mean(img_array, axis=2)

    # Laplacian kernel
    laplacian = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])

    # Convolve (simple implementation)
    from scipy import ndimage
    laplacian_response = ndimage.convolve(gray, laplacian)
    variance = np.var(laplacian_response)

    # Normalize to 0-1 (empirically determined thresholds)
    # Higher variance = sharper image
    score = min(1.0, variance / 500.0)
    return float(score)


def _compute_exposure_score(img_array: np.ndarray) -> float:
    """Compute exposure score from histogram.

    Good exposure has well-distributed histogram with
    minimal clipping at extremes.

    Args:
        img_array: Numpy array (H, W, C)

    Returns:
        Exposure score 0-1
    """
    # Compute luminance
    luminance = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]

    # Check for clipping
    dark_ratio = np.mean(luminance < 20)
    bright_ratio = np.mean(luminance > 235)

    # Penalize clipping
    clipping_penalty = (dark_ratio + bright_ratio) * 2.0

    # Check histogram spread
    mean_lum = np.mean(luminance)
    std_lum = np.std(luminance)

    # Good exposure: mean around 128, reasonable std
    mean_score = 1.0 - abs(mean_lum - 128) / 128
    std_score = min(1.0, std_lum / 60)

    score = (mean_score * 0.5 + std_score * 0.5) * (1.0 - clipping_penalty)
    return float(max(0.0, min(1.0, score)))


def _compute_composition_score(img_array: np.ndarray) -> float:
    """Compute composition score.

    Simple heuristic based on:
    - Edge activity in rule-of-thirds regions
    - Centered subject detection

    Args:
        img_array: Numpy array (H, W, C)

    Returns:
        Composition score 0-1
    """
    h, w = img_array.shape[:2]
    gray = np.mean(img_array, axis=2)

    # Simple edge detection using gradient
    gy, gx = np.gradient(gray)
    edges = np.sqrt(gx**2 + gy**2)

    # Check rule of thirds lines
    third_h = h // 3
    third_w = w // 3

    # Activity near thirds lines
    h_lines = np.mean(edges[third_h-10:third_h+10, :]) + np.mean(edges[2*third_h-10:2*third_h+10, :])
    v_lines = np.mean(edges[:, third_w-10:third_w+10]) + np.mean(edges[:, 2*third_w-10:2*third_w+10])

    thirds_activity = (h_lines + v_lines) / 4

    # Center activity
    center_region = edges[h//3:2*h//3, w//3:2*w//3]
    center_activity = np.mean(center_region)

    # Normalize
    avg_edge = np.mean(edges) + 1e-8
    thirds_score = min(1.0, thirds_activity / avg_edge / 2)
    center_score = min(1.0, center_activity / avg_edge / 2)

    # Combine (prefer thirds but accept centered)
    score = thirds_score * 0.6 + center_score * 0.4
    return float(score)


def _apply_manual_bonuses(photo: JobPhoto) -> float:
    """Apply manual bonuses from metadata.

    Based on OVERVIEW.md scoring policy:
    - +0.30 if hero_global
    - +0.20 if hero_room
    - +0.10 if preferred_opening
    - +0.10 if preferred_closing
    - +0.05 * hero_priority
    - -1.00 if exclude

    Args:
        photo: JobPhoto with manual_metadata

    Returns:
        Final score with bonuses applied
    """
    metadata = photo.manual_metadata or {}
    base = photo.base_score or 0.5

    bonus = 0.0

    if metadata.get("hero_global"):
        bonus += 0.30
    if metadata.get("hero_room"):
        bonus += 0.20
    if metadata.get("preferred_opening"):
        bonus += 0.10
    if metadata.get("preferred_closing"):
        bonus += 0.10
    if "hero_priority" in metadata:
        bonus += 0.05 * int(metadata.get("hero_priority", 0))

    if photo.exclude or metadata.get("exclude"):
        return 0.0

    return min(1.0, base + bonus)
