"""Motion planning for room clusters based on depth analysis.

LTX-2 Prompting Strategy:
- Always describe camera motion explicitly
- Always describe speed (slow, subtle, cinematic)
- Always constrain geometry in prompts
- Always add negative constraints to prevent warping
- Keep clips 2-4 seconds
- Use CFG scale 3-6
"""
import logging
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.models import RoomCluster, AnalysisResult

if TYPE_CHECKING:
    from app.db.models import JobPhoto

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

# LTX-2 default parameters by tier
LTX2_PARAMS = {
    "low": {"cfg_scale": 4.0, "inference_steps": 30},
    "medium": {"cfg_scale": 4.5, "inference_steps": 40},
    "high": {"cfg_scale": 5.0, "inference_steps": 50},
}

# Motion descriptions for prompt generation
MOTION_DESCRIPTIONS = {
    "static": "static camera with subtle micro-movement",
    "micro_push_in": "slow smooth cinematic push-in camera movement",
    "micro_push_out": "slow smooth cinematic push-out camera movement",
    "subtle_pan": "slow smooth pan from left to right",
    "push_in": "slow forward dolly motion",
    "push_out": "slow backward dolly motion",
    "pan_left": "smooth lateral pan to the left",
    "pan_right": "smooth lateral pan to the right",
    "reveal": "slow reveal transition",
    "dolly_in": "cinematic dolly-in camera movement",
    "dolly_out": "cinematic dolly-out camera movement",
    "orbit": "cinematic slow orbit camera movement",
    "parallax": "subtle parallax effect with natural depth separation",
    "multi_view": "cinematic multi-view synthesis with true perspective",
}

# Base prompt templates by tier
# Note: {view_count_phrase} is dynamically generated based on image_count
PROMPT_TEMPLATES = {
    "low": """Professional real estate interior video.
{motion_description}.
Stable tripod motion.
Natural lighting.
Maintain structural consistency.
Preserve wall alignment.
No object movement.
No distortion.
No warping.
No morphing.
No flicker.
Photorealistic.""",

    "medium": """Smooth cinematic transition {view_count_phrase} of the same room.
{motion_description}.
Maintain consistent room geometry.
Preserve window and door placement.
Preserve object positions.
Stable lighting.
No morphing.
No warping.
No texture melting.
No bending walls.
Photorealistic interior.""",

    "high": """Cinematic {motion_description} inside room.
{view_count_phrase}.
Subtle parallax effect.
Natural depth separation.
Maintain structural realism.
No hallucinated objects.
No geometry distortion.
Stable lighting.
Preserve original layout.
Photorealistic.""",
}

# Room-specific prompt modifiers
ROOM_MODIFIERS = {
    "hallway": """Slow forward camera movement down hallway.
Maintain depth perspective.
No structural distortion.
Stable alignment.""",

    "kitchen": """Slow reveal behind counter.
Preserve appliance positions.
Stable countertop geometry.""",

    "exterior": """Slow approach toward entrance.
Maintain facade geometry.
Preserve landscaping positions.
Natural outdoor lighting.""",

    "drone": """Slow aerial pullback.
Maintain roof geometry.
Stable horizon line.
No ground warping.""",
}

# SEVA minimal prompts (geometry-driven, not prompt-driven)
SEVA_PROMPT = """Realistic indoor room video.
Natural lighting.
Photorealistic."""


def plan_motion_for_cluster(
    db: Session,
    cluster: RoomCluster,
    preloaded_photos: Optional[List["JobPhoto"]] = None,
) -> AnalysisResult:
    """Plan motion strategy for a room cluster.

    Based on OVERVIEW.md:
    - Low confidence: micro push/pan only (LTX-2 single image)
    - Medium confidence: interpolation and reveals (LTX-2 interpolation)
    - High confidence: multi-view synthesis (LTX-2 parallax/multi-view or SEVA)

    Args:
        db: Database session
        cluster: RoomCluster to plan motion for

    Returns:
        Created AnalysisResult with motion plan
    """
    tier = cluster.confidence_tier or "low"
    allowed_motions = MOTION_TYPES.get(tier, MOTION_TYPES["low"])
    ordered_photos = _ordered_cluster_photos(cluster, preloaded_photos)

    # Select recommended motion based on room type and tier
    recommended = _select_recommended_motion(cluster, allowed_motions)
    inferred_motion, motion_guidance, matching_summary = _infer_motion_from_matching(
        db=db,
        job_id=int(cluster.job_id),
        ordered_photos=ordered_photos,
        allowed_motions=allowed_motions,
    )
    if inferred_motion:
        recommended = inferred_motion

    requested_motion, requested_motion_reason = _requested_motion(ordered_photos, cluster, allowed_motions)
    if requested_motion is not None:
        recommended = requested_motion
    if requested_motion_reason:
        motion_guidance = "\n\n".join(part for part in (motion_guidance, requested_motion_reason) if part)

    # Duration policy:
    # - Single-photo clusters: 1-2s
    # - Multi-photo clusters: up to 4s
    duration = _compute_duration_seconds(cluster, recommended)

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
        model_recommendation = "LTX-2 single"

    # Generate LTX-2 prompt template
    prompt_template = _generate_prompt_template(cluster, tier, recommended)
    if motion_guidance:
        prompt_template = f"{prompt_template}\n\n{motion_guidance}"

    # Get LTX-2 parameters
    ltx2_params = LTX2_PARAMS.get(tier, LTX2_PARAMS["low"])

    # Create analysis result with LTX-2 prompt and parameters
    result = AnalysisResult(
        job_id=cluster.job_id,
        room_cluster_id=cluster.id,
        recommended_motion=recommended,
        allowed_motion_types=allowed_motions,
        recommended_duration=duration,
        tier=tier,
        model_recommendation=model_recommendation,
        prompt_template=prompt_template,
        cfg_scale=ltx2_params["cfg_scale"],
        inference_steps=ltx2_params["inference_steps"],
        debug_metrics={
            "image_count": cluster.image_count,
            "sfm_eligible": cluster.sfm_eligible,
            "matching_inferred_motion": inferred_motion,
            "matching_motion_guidance": motion_guidance,
            "matching_summary": matching_summary,
            "requested_motion": _configured_motion(ordered_photos),
            "motion_fallback_reason": requested_motion_reason,
        },
    )

    db.add(result)

    # Also update cluster with motion info
    cluster.recommended_motion = recommended
    cluster.allowed_motion_types = ",".join(allowed_motions)
    cluster.recommended_duration = duration

    # Select hero photo for cluster
    _select_hero_photo(cluster, photos=ordered_photos)

    db.commit()

    logger.info(
        f"Planned motion for cluster {cluster.id}: "
        f"{recommended} ({tier} tier, {duration}s, {model_recommendation}, "
        f"verified_transitions={matching_summary.get('verified_transitions', 0)}/"
        f"{matching_summary.get('total_transitions', 0)})"
    )

    return result


def _generate_view_count_phrase(image_count: int) -> str:
    """Generate natural language phrase for number of views.

    Args:
        image_count: Number of images in cluster

    Returns:
        Human-readable phrase like "between two views" or "across four viewpoints"
    """
    count_words = {
        1: "from a single view",
        2: "between two views",
        3: "across three viewpoints",
        4: "across four viewpoints",
    }

    if image_count in count_words:
        return count_words[image_count]
    else:
        return f"across {image_count} viewpoints"


def _generate_prompt_template(cluster: RoomCluster, tier: str, motion: str) -> str:
    """Generate LTX-2 prompt template for a cluster.

    Args:
        cluster: RoomCluster
        tier: Confidence tier (low, medium, high)
        motion: Selected motion type

    Returns:
        LTX-2 prompt template string
    """
    room_type = (cluster.room_type or "").lower()

    # Check if SEVA should be used (high confidence + SFM eligible)
    if cluster.sfm_eligible and tier == "high":
        return SEVA_PROMPT

    # Get motion description
    motion_desc = MOTION_DESCRIPTIONS.get(motion, "slow cinematic camera movement")

    # Generate view count phrase based on actual image count
    view_phrase = _generate_view_count_phrase(cluster.image_count or 1)

    # Get base template
    template = PROMPT_TEMPLATES.get(tier, PROMPT_TEMPLATES["low"])

    # Format with motion description and view count
    prompt = template.format(
        motion_description=motion_desc,
        view_count_phrase=view_phrase,
    )

    # Add room-specific modifier if applicable
    for room_key, modifier in ROOM_MODIFIERS.items():
        if room_key in room_type:
            prompt = f"{prompt}\n\n{modifier}"
            break

    return prompt


def get_prompt_for_motion(
    cluster: RoomCluster,
    motion: str,
    tier: str,
) -> Tuple[str, Dict[str, Any]]:
    """Get the complete prompt and parameters for video generation.

    This is called by Phase 2 (render) to get the actual prompt to use.

    Args:
        cluster: RoomCluster
        motion: Motion type to generate
        tier: Confidence tier

    Returns:
        Tuple of (prompt_string, params_dict)
    """
    prompt = _generate_prompt_template(cluster, tier, motion)
    params = LTX2_PARAMS.get(tier, LTX2_PARAMS["low"]).copy()

    # Add common parameters
    params["duration"] = _compute_duration_seconds(cluster, motion)
    params["fps"] = 24
    params["resolution"] = "720p"  # Generate at 720p, upscale later

    return prompt, params


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


def _compute_duration_seconds(cluster: RoomCluster, motion: str) -> float:
    """Compute recommended shot duration using sequencing constraints."""
    base = float(MOTION_DURATIONS.get(motion, 3.0))
    image_count = int(cluster.image_count or 1)

    if image_count <= 1:
        # Single-photo clusters should be short.
        # Map base motion duration into [1.0, 2.0] while preserving relative pacing.
        single = 1.0 + max(0.0, min(1.0, (base - 2.0) / 2.0))
        return float(np.clip(single, 1.0, 2.0))

    # Multi-photo clusters can run longer for transitions, but never exceed 4s.
    return float(np.clip(base, 2.0, 4.0))


def _ordered_cluster_photos(
    cluster: RoomCluster,
    preloaded_photos: Optional[List["JobPhoto"]],
) -> List["JobPhoto"]:
    source = preloaded_photos if preloaded_photos is not None else list(cluster.photos or [])
    return sorted(source, key=lambda p: (p.cluster_order if p.cluster_order is not None else 10**9, p.id))


def _configured_motion(photos: List["JobPhoto"]) -> str | None:
    requested = [str((photo.manual_metadata or {}).get("camera_motion", "auto")) for photo in photos]
    requested = [motion for motion in requested if motion and motion != "auto"]
    if not requested or len(set(requested)) != 1:
        return None
    return requested[0]


def _requested_motion(
    photos: List["JobPhoto"],
    cluster: RoomCluster,
    allowed_motions: List[str],
) -> Tuple[str | None, str | None]:
    """Honor one clear user override only when verified geometry permits it."""
    requested = _configured_motion(photos)
    if requested is None:
        return None, None
    mapping = {
        "push_in": "push_in",
        "push_out": "push_out",
        "orbit_right": "orbit",
        "orbit_left": "orbit",
        "orbit": "orbit",
        "multi_view": "multi_view",
    }
    requested = mapping.get(requested, requested)
    requires_multi_view = requested in {"orbit", "multi_view"}
    verified_multi_view = bool(cluster.sfm_eligible and int(cluster.image_count or 0) >= 2)
    if requires_multi_view and not verified_multi_view:
        fallback = next((motion for motion in ("micro_push_in", "subtle_pan", "static") if motion in allowed_motions), allowed_motions[0])
        return fallback, (
            "Requested multi-view motion was not used because this group lacks verified "
            "interpolation-safe geometry; render a single-image move instead."
        )
    if requested in allowed_motions:
        return requested, "User-selected camera motion is supported by verified scene evidence."
    fallback = next((motion for motion in ("micro_push_in", "push_in", "subtle_pan", "static") if motion in allowed_motions), allowed_motions[0])
    return fallback, (
        f"Requested {requested} is unsafe for this confidence tier; using {fallback} to preserve geometry."
    )


def _motion_from_direction(
    dx: float,
    dy: float,
    allowed_motions: List[str],
) -> str | None:
    abs_dx = abs(dx)
    abs_dy = abs(dy)
    if max(abs_dx, abs_dy) < 0.08:
        return None

    # DB direction is content shift; camera motion is opposite.
    if abs_dx >= abs_dy:
        if dx > 0:
            for m in ("orbit", "pan_left", "subtle_pan"):
                if m in allowed_motions:
                    return m
        else:
            for m in ("orbit", "pan_right", "subtle_pan"):
                if m in allowed_motions:
                    return m
    else:
        for m in ("reveal", "subtle_pan", "push_in"):
            if m in allowed_motions:
                return m
    return None


def _infer_motion_from_matching(
    db: Session,
    job_id: int,
    ordered_photos: List["JobPhoto"],
    allowed_motions: List[str],
) -> Tuple[str | None, str, Dict[str, Any]]:
    """Infer dominant motion from scene-relation directions in this cluster."""
    summary: Dict[str, Any] = {
        "total_transitions": max(0, len(ordered_photos) - 1),
        "evaluated_transitions": 0,
        "verified_transitions": 0,
        "avg_relation_confidence": 0.0,
        "avg_dx": 0.0,
        "avg_dy": 0.0,
        "dominant_image_motion": "unknown",
    }

    if len(ordered_photos) < 2:
        return None, "", summary

    weighted_dx = 0.0
    weighted_dy = 0.0
    total_weight = 0.0
    verified_confidence = 0.0

    for idx in range(len(ordered_photos) - 1):
        left = int(ordered_photos[idx].id)
        right = int(ordered_photos[idx + 1].id)
        photo_a = min(left, right)
        photo_b = max(left, right)
        summary["evaluated_transitions"] += 1

        row = db.execute(
            text(
                """
                SELECT direction_dx, direction_dy, relation_confidence
                FROM photo_relations
                WHERE job_id = :job_id
                  AND photo_a_id = :photo_a
                  AND photo_b_id = :photo_b
                  AND continuity_type = 'interpolation_safe'
                  AND is_connected = true
                LIMIT 1
                """
            ),
            {"job_id": int(job_id), "photo_a": int(photo_a), "photo_b": int(photo_b)},
        ).fetchone()
        if row is None:
            continue

        dx = row[0]
        dy = row[1]
        relation_confidence = row[2]
        if dx is None or dy is None:
            continue
        if relation_confidence is None or float(relation_confidence) < 0.45:
            continue

        summary["verified_transitions"] += 1
        evidence = float(relation_confidence)
        verified_confidence += evidence
        dx = float(dx)
        dy = float(dy)
        if left != photo_a:
            dx = -dx
            dy = -dy

        weight = float(max(1.0, evidence * 100.0))
        weighted_dx += dx * weight
        weighted_dy += dy * weight
        total_weight += weight

    if summary["verified_transitions"] > 0:
        summary["avg_relation_confidence"] = float(
            verified_confidence / summary["verified_transitions"]
        )

    if total_weight <= 0.0:
        guidance = (
            "Matching evidence is weak for this cluster; keep motion subtle and prioritize "
            "layout stability over directional movement."
        )
        return None, guidance, summary

    avg_dx = weighted_dx / total_weight
    avg_dy = weighted_dy / total_weight
    summary["avg_dx"] = float(avg_dx)
    summary["avg_dy"] = float(avg_dy)

    inferred = _motion_from_direction(avg_dx, avg_dy, allowed_motions)

    horizontal = abs(avg_dx) >= abs(avg_dy)
    if horizontal:
        camera_dir = "left" if avg_dx > 0 else "right"
    else:
        camera_dir = "up" if avg_dy > 0 else "down"
    summary["dominant_image_motion"] = camera_dir

    if inferred is None:
        guidance = (
            "Matching evidence is available but direction is ambiguous; keep movement gentle and "
            "consistent with verified transitions."
        )
        return None, guidance, summary

    guidance = (
        "Match-guided image-space motion: follow the verified subject movement "
        f"{camera_dir} across transitions "
        f"({summary['verified_transitions']}/{summary['total_transitions']} verified, "
        f"avg relation confidence {summary['avg_relation_confidence']:.2f}); keep movement smooth and consistent."
    )
    return inferred, guidance, summary


def _select_hero_photo(cluster: RoomCluster, photos: Optional[List["JobPhoto"]] = None) -> None:
    """Select the hero photo for a cluster.

    Based on OVERVIEW.md priority:
    1. hero_room flag
    2. preferred_opening flag
    3. highest final_score

    Args:
        cluster: RoomCluster to update
    """
    candidates_source = photos if photos is not None else cluster.photos
    if not candidates_source:
        return

    candidates = []
    for photo in candidates_source:
        if photo.exclude:
            continue

        metadata = photo.manual_metadata or {}
        priority = 0

        if metadata.get("editorial_role") == "hero":
            priority = 1_000
        elif metadata.get("hero_room"):
            priority = 100
        elif metadata.get("preferred_opening"):
            priority = 50
        else:
            priority = (photo.final_score or 0) * 10

        candidates.append((priority, photo))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        cluster.hero_photo_id = candidates[0][1].id
