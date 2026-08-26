"""Clip generation using LTX-2 video model.

Generates video clips for room clusters using multi-frame injection.
"""
import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import imageio
import numpy as np
from PIL import Image
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.models import Job, JobPhoto, RoomCluster, AnalysisResult, Clip
from app.models.ltx2 import ltx2_model, LTX2_FPS
from app.services.storage.s3_client import S3Client

logger = logging.getLogger(__name__)


def generate_clip_for_cluster(
    db: Session,
    job: Job,
    cluster: RoomCluster,
    analysis: AnalysisResult,
    s3_client: S3Client,
) -> Optional[Clip]:
    """Generate a video clip for a room cluster.

    Downloads all photos in the cluster, generates video using LTX-2
    with multi-frame injection, uploads to S3, and creates Clip record.

    Args:
        db: Database session
        job: Job instance
        cluster: RoomCluster to generate clip for
        analysis: AnalysisResult with motion plan and prompt
        s3_client: S3 client for upload/download

    Returns:
        Created Clip record, or None if generation failed
    """
    logger.info(
        f"Generating clip for cluster {cluster.id} "
        f"({cluster.room_type}, {cluster.image_count} photos)"
    )

    try:
        # Get all photos in this cluster (already ordered by overlap detection)
        photos = (
            db.query(JobPhoto)
            .filter(JobPhoto.room_cluster_id == cluster.id)
            .order_by(JobPhoto.cluster_order.asc().nulls_last(), JobPhoto.id.asc())
            .all()
        )

        if not photos:
            logger.warning(f"No photos found for cluster {cluster.id}")
            return None

        # Download all images
        images = _download_cluster_images(photos, s3_client)
        if not images:
            logger.error(f"Failed to download images for cluster {cluster.id}")
            return None

        # Get generation parameters from analysis
        prompt = analysis.prompt_template or _get_default_prompt(
            analysis.recommended_motion, cluster.room_type
        )
        direction_guidance = _build_direction_prompt_guidance(
            db=db,
            job_id=job.id,
            ordered_photo_ids=[p.id for p in photos],
        )
        if direction_guidance:
            prompt = f"{prompt}\n\n{direction_guidance}"
        cfg_scale = analysis.cfg_scale or 4.0
        inference_steps = analysis.inference_steps or 20

        # Compute frame count based on recommended duration
        duration = analysis.recommended_duration or 2.0
        num_frames = int(duration * LTX2_FPS) + 1  # +1 for inclusive

        # Generate seed for reproducibility
        seed = hash(f"{job.id}_{cluster.id}") % (2**32)

        # Generate video with LTX-2
        logger.info(
            f"Generating {num_frames} frames with {len(images)} input images"
        )
        video_frames, fps = ltx2_model.generate_video(
            images=images,
            prompt=prompt,
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
            num_frames=num_frames,
            guide_strength=0.6,
            seed=seed,
            motion_type=analysis.recommended_motion or "push_in",
        )

        # Encode to MP4
        video_bytes = _encode_video(video_frames, fps)

        # Upload to S3
        s3_key = f"jobs/{job.id}/clips/cluster_{cluster.id}.mp4"
        s3_uri = s3_client.upload_bytes(
            s3_key, video_bytes, content_type="video/mp4"
        )

        # Create Clip record
        clip = Clip(
            job_id=job.id,
            room_cluster_id=cluster.id,
            source_photo_ids=[p.id for p in photos],
            motion_type=analysis.recommended_motion,
            model_used=_get_model_name(analysis, ltx2_model.is_mock_mode()),
            prompt_used=prompt,
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
            seed=seed,
            is_3d=cluster.sfm_eligible,
            duration=len(video_frames) / fps,
            s3_uri=s3_uri,
            status="rendered",
            render_metadata={
                "input_images": len(images),
                "num_frames": len(video_frames),
                "fps": fps,
                "resolution": f"{video_frames.shape[2]}x{video_frames.shape[1]}",
                "mock_mode": ltx2_model.is_mock_mode(),
            },
        )

        db.add(clip)
        db.flush()

        logger.info(
            f"Generated clip {clip.id}: {clip.duration:.2f}s, "
            f"s3_uri={s3_uri}"
        )

        return clip

    except Exception as e:
        logger.error(f"Failed to generate clip for cluster {cluster.id}: {e}")
        return None


def _download_cluster_images(
    photos: List[JobPhoto], s3_client: S3Client
) -> List[Image.Image]:
    """Download images for all photos in a cluster.

    Args:
        photos: List of JobPhoto instances
        s3_client: S3 client

    Returns:
        List of PIL Images (in order)
    """
    images = []
    for photo in photos:
        try:
            img = s3_client.download_image(photo.s3_uri)
            images.append(img)
        except Exception as e:
            logger.warning(f"Failed to download photo {photo.id}: {e}")
            continue

    return images


def _encode_video(frames: np.ndarray, fps: int) -> bytes:
    """Encode video frames to MP4 bytes.

    Args:
        frames: Video frames (num_frames, H, W, 3)
        fps: Frame rate

    Returns:
        MP4 video as bytes
    """
    # Create temporary file for video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        temp_path = f.name

    try:
        # Ensure frames are uint8
        if frames.dtype != np.uint8:
            frames = (frames * 255).astype(np.uint8)

        # Write video using imageio
        writer = imageio.get_writer(
            temp_path,
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
        )

        for frame in frames:
            writer.append_data(frame)

        writer.close()

        # Read back as bytes
        with open(temp_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes

    finally:
        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)


def _get_default_prompt(motion_type: str, room_type: str) -> str:
    """Get default prompt if none provided.

    Args:
        motion_type: Type of camera motion
        room_type: Type of room

    Returns:
        Default prompt string
    """
    motion_desc = {
        "push_in": "slow forward dolly motion",
        "push_out": "slow backward dolly motion",
        "dolly_in": "cinematic dolly-in camera movement",
        "dolly_out": "cinematic dolly-out camera movement",
        "pan_left": "smooth lateral pan to the left",
        "pan_right": "smooth lateral pan to the right",
        "micro_push_in": "slow smooth cinematic push-in",
        "subtle_pan": "slow smooth pan",
        "orbit": "cinematic slow orbit camera movement",
        "reveal": "slow reveal transition",
    }.get(motion_type, "slow cinematic camera movement")

    return f"""Professional real estate interior video.
{motion_desc}.
Stable tripod motion.
Natural lighting.
Maintain structural consistency.
Preserve wall alignment.
No object movement.
No distortion.
No warping.
No morphing.
No flicker.
Photorealistic."""


def _camera_direction_recommendation(
    dx: float | None,
    dy: float | None,
    relation_confidence: float | None = None,
    min_relation_confidence: float = 0.45,
) -> str:
    """Translate content-shift direction into camera motion recommendation."""
    if (
        relation_confidence is None
        or relation_confidence < min_relation_confidence
        or dx is None
        or dy is None
    ):
        return "Not geometrically verified"

    abs_x = abs(dx)
    abs_y = abs(dy)
    if abs_x < 0.15 and abs_y < 0.15:
        return "Hold / very subtle move"

    # Direction vector is content shift; camera motion is opposite.
    if abs_x >= abs_y:
        return "Move camera left" if dx > 0 else "Move camera right"
    return "Move camera up" if dy > 0 else "Move camera down"


def _build_direction_prompt_guidance(
    db: Session,
    job_id: int,
    ordered_photo_ids: List[int],
) -> str:
    """Build prompt guidance from transition direction metrics by photo IDs."""
    if len(ordered_photo_ids) < 2:
        return ""

    # Some environments may not have relation rows yet during rollout.
    supports_direction = False
    try:
        col_rows = db.execute(
            text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'photo_relations'
                  AND table_schema = current_schema()
                """
            )
        ).fetchall()
        available_cols = {str(row[0]) for row in col_rows}
        supports_direction = {
            "direction_dx",
            "direction_dy",
        }.issubset(available_cols)
    except Exception as err:
        logger.warning("Could not inspect photo_relations schema for prompt guidance: %s", err)

    if not supports_direction:
        return ""

    lines = []
    for idx in range(len(ordered_photo_ids) - 1):
        from_id = int(ordered_photo_ids[idx])
        to_id = int(ordered_photo_ids[idx + 1])
        photo_a = min(from_id, to_id)
        photo_b = max(from_id, to_id)

        row = db.execute(
            text(
                """
                SELECT
                    direction_dx,
                    direction_dy,
                    relation_confidence
                FROM photo_relations
                WHERE job_id = :job_id
                  AND photo_a_id = :photo_a_id
                  AND photo_b_id = :photo_b_id
                LIMIT 1
                """
            ),
            {
                "job_id": int(job_id),
                "photo_a_id": photo_a,
                "photo_b_id": photo_b,
            },
        ).fetchone()

        if row is None:
            lines.append(f"{from_id} -> {to_id}: keep smooth consistent transition.")
            continue

        dx = row[0]
        dy = row[1]

        # Direction in DB is stored for photo_a -> photo_b; flip when traversing opposite.
        if from_id != photo_a and dx is not None and dy is not None:
            dx = -float(dx)
            dy = -float(dy)
        elif dx is not None and dy is not None:
            dx = float(dx)
            dy = float(dy)

        relation_confidence = row[2]
        rec = _camera_direction_recommendation(
            dx,
            dy,
            float(relation_confidence) if relation_confidence is not None else None,
        )
        confidence_txt = f"{float(relation_confidence):.3f}" if relation_confidence is not None else "N/A"
        lines.append(f"{from_id} -> {to_id}: {rec} (relation_confidence={confidence_txt}).")

    if not lines:
        return ""

    guidance = [
        "Camera path guidance by photo ID:",
        *[f"- {line}" for line in lines],
        "Keep geometry, layout, and lighting consistent across these transitions.",
    ]
    return "\n".join(guidance)


def _get_model_name(analysis: AnalysisResult, is_mock: bool) -> str:
    """Get model name for clip record.

    Args:
        analysis: AnalysisResult
        is_mock: Whether running in mock mode

    Returns:
        Model name string
    """
    if is_mock:
        return "LTX-2 (mock)"

    recommendation = analysis.model_recommendation or "LTX-2 single"
    return recommendation
