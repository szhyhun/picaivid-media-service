"""Motion planning for room clusters based on depth analysis."""
import logging
from typing import List

from sqlalchemy.orm import Session

from app.db.models import RoomCluster, AnalysisResult

logger = logging.getLogger(__name__)

# Motion types by confidence tier
MOTION_TYPES = {
    "low": ["static", "micro_push_in", "micro_push_out", "subtle_pan"],
    "medium": ["push_in", "push_out", "pan_left", "pan_right", "reveal"],
    "high": ["dolly_in", "dolly_out", "orbit", "parallax", "multi_view"],
}

# Default durations by motion type (seconds) - minimum 2s
MOTION_DURATIONS = {
    "static": 2.0,
    "micro_push_in": 3.0,
    "micro_push_out": 3.0,
    "subtle_pan": 3.0,
    "push_in": 3.5,
    "push_out": 3.5,
    "pan_left": 3.5,
    "pan_right": 3.5,
    "reveal": 4.0,
    "dolly_in": 4.0,
    "dolly_out": 4.0,
    "orbit": 5.0,
    "parallax": 4.0,
    "multi_view": 5.5,
}


def plan_motion_for_cluster(db: Session, cluster: RoomCluster) -> AnalysisResult:
    """Plan motion strategy for a room cluster.

    Based on OVERVIEW.md:
    - Low confidence: micro push/pan only
    - Medium confidence: interpolation and reveals
    - High confidence: multi-view synthesis

    Args:
        db: Database session
        cluster: RoomCluster to plan motion for

    Returns:
        Created AnalysisResult with motion plan
    """
    tier = cluster.confidence_tier or "low"
    allowed_motions = MOTION_TYPES.get(tier, MOTION_TYPES["low"])

    # Select recommended motion based on room type and tier
    recommended = _select_recommended_motion(cluster, allowed_motions)

    # Determine duration (minimum 2 seconds)
    duration = max(2.0, MOTION_DURATIONS.get(recommended, 3.0))

    # Determine model recommendation
    # SFM-eligible clusters get more advanced motion capabilities
    if cluster.sfm_eligible:
        if tier == "high":
            model_recommendation = "LTX-2 multi-view"  # Full 3D reconstruction
        else:
            model_recommendation = "LTX-2 parallax"    # Partial 3D / parallax reveals
    elif tier in ("medium", "high"):
        model_recommendation = "LTX-2 interpolation"
    else:
        model_recommendation = "LTX-2 single image"

    # Create analysis result
    result = AnalysisResult(
        job_id=cluster.job_id,
        room_cluster_id=cluster.id,
        recommended_motion=recommended,
        allowed_motion_types=allowed_motions,
        recommended_duration=duration,
        tier=tier,
        model_recommendation=model_recommendation,
        debug_metrics={
            "depth_variance": cluster.depth_variance,
            "image_count": cluster.image_count,
            "sfm_eligible": cluster.sfm_eligible,
        },
    )

    db.add(result)

    # Also update cluster with motion info
    cluster.recommended_motion = recommended
    cluster.allowed_motion_types = ",".join(allowed_motions)
    cluster.recommended_duration = duration

    # Select hero photo for cluster
    _select_hero_photo(cluster)

    db.commit()

    logger.info(
        f"Planned motion for cluster {cluster.id}: "
        f"{recommended} ({tier} tier, {duration}s)"
    )

    return result


def _select_recommended_motion(cluster: RoomCluster, allowed_motions: List[str]) -> str:
    """Select the best motion type for a cluster.

    Args:
        cluster: RoomCluster
        allowed_motions: List of allowed motion types

    Returns:
        Selected motion type
    """
    room_type = (cluster.room_type or "").lower()

    # Room-specific preferences
    if "exterior" in room_type or "front" in room_type:
        if "push_in" in allowed_motions:
            return "push_in"
        if "micro_push_in" in allowed_motions:
            return "micro_push_in"

    if "living" in room_type or "family" in room_type:
        if "orbit" in allowed_motions:
            return "orbit"
        if "pan_left" in allowed_motions:
            return "pan_left"

    if "kitchen" in room_type:
        if "reveal" in allowed_motions:
            return "reveal"
        if "push_out" in allowed_motions:
            return "push_out"

    if "bedroom" in room_type:
        if "push_in" in allowed_motions:
            return "push_in"
        if "micro_push_in" in allowed_motions:
            return "micro_push_in"

    if "bathroom" in room_type:
        if "subtle_pan" in allowed_motions:
            return "subtle_pan"
        if "micro_push_in" in allowed_motions:
            return "micro_push_in"

    if "drone" in room_type or "aerial" in room_type:
        if "dolly_out" in allowed_motions:
            return "dolly_out"
        return "static"

    # Default: prefer push_in if available
    for motion in ["push_in", "micro_push_in", "subtle_pan", "static"]:
        if motion in allowed_motions:
            return motion

    return allowed_motions[0] if allowed_motions else "static"


def _select_hero_photo(cluster: RoomCluster) -> None:
    """Select the hero photo for a cluster.

    Based on OVERVIEW.md priority:
    1. hero_room flag
    2. preferred_opening flag
    3. highest final_score

    Args:
        cluster: RoomCluster to update
    """
    if not cluster.photos:
        return

    candidates = []
    for photo in cluster.photos:
        if photo.exclude:
            continue

        metadata = photo.manual_metadata or {}
        priority = 0

        if metadata.get("hero_room"):
            priority = 100
        elif metadata.get("preferred_opening"):
            priority = 50
        else:
            priority = (photo.final_score or 0) * 10

        candidates.append((priority, photo))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        cluster.hero_photo_id = candidates[0][1].id
