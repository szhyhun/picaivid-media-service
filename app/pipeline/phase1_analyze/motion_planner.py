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
from typing import List, Dict, Any, Tuple

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


def plan_motion_for_cluster(db: Session, cluster: RoomCluster) -> AnalysisResult:
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

    # Select recommended motion based on room type and tier
    recommended = _select_recommended_motion(cluster, allowed_motions)

    # Determine duration (minimum 2 seconds, max 4 seconds for cost control)
    duration = min(4.0, max(2.0, MOTION_DURATIONS.get(recommended, 3.0)))

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
        f"{recommended} ({tier} tier, {duration}s, {model_recommendation})"
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
    params["duration"] = min(4.0, max(2.0, MOTION_DURATIONS.get(motion, 3.0)))
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
