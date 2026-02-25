"""FastAPI application for Picaivid Media Service."""
from datetime import datetime
from types import SimpleNamespace

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.session import get_db
from app.db.models import Job, Clip, JobPhoto, RoomCluster, AnalysisResult, PhotoSimilarity
from app.schemas.job import JobMessage, JobStatusResponse
from app.schemas.clip import (
    ClipResponse, ClipListResponse, SourcePhotoInfo, ClusterInfo, AnalysisInfo,
    ClipTransitionStep,
    ClusterDebugResponse, ClusterListDebugResponse, PhotoDebugInfo, PhotoSimilarityInfo,
)
from app.pipeline.orchestrator import PipelineOrchestrator

# Setup logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title="Picaivid Media Service",
    description="Phased video pipeline for real estate media",
    version="0.1.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


GEOMETRY_MIN_VERIFIED_INLIERS = 8
GEOMETRY_MIN_VERIFIED_SCORE = 0.25
OVERLAP_DIRECTION_COMPONENT_THRESHOLD = 0.2


def _safe_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _direction_for_order(
    from_photo_id: int,
    to_photo_id: int,
    sim: PhotoSimilarity | SimpleNamespace | None,
) -> tuple[float | None, float | None]:
    if sim is None:
        return None, None

    dx = _safe_float(getattr(sim, "direction_dx", None))
    dy = _safe_float(getattr(sim, "direction_dy", None))
    if dx is None or dy is None:
        return None, None

    sim_a = int(getattr(sim, "photo_a_id", min(from_photo_id, to_photo_id)))
    if from_photo_id != sim_a:
        return -dx, -dy
    return dx, dy


def _is_geometrically_verified(
    geometric_inliers: int | None,
    geometric_score: float | None,
    dx: float | None,
    dy: float | None,
    min_verified_inliers: int = GEOMETRY_MIN_VERIFIED_INLIERS,
    min_verified_score: float = GEOMETRY_MIN_VERIFIED_SCORE,
) -> bool:
    if geometric_inliers is None or geometric_inliers < min_verified_inliers:
        return False
    if geometric_score is not None and geometric_score < min_verified_score:
        return False
    if dx is None or dy is None:
        return False
    return (dx * dx + dy * dy) > 1e-8


def _overlap_zones_from_direction(
    dx: float | None,
    dy: float | None,
    threshold: float = OVERLAP_DIRECTION_COMPONENT_THRESHOLD,
) -> tuple[str | None, str | None]:
    if dx is None or dy is None:
        return None, None

    if dx > threshold:
        from_x, to_x = "left", "right"
    elif dx < -threshold:
        from_x, to_x = "right", "left"
    else:
        from_x = to_x = "center"

    if dy > threshold:
        from_y, to_y = "top", "bottom"
    elif dy < -threshold:
        from_y, to_y = "bottom", "top"
    else:
        from_y = to_y = "center"

    from_zone = f"{from_x}-{from_y}"
    to_zone = f"{to_x}-{to_y}"
    return from_zone, to_zone


def _camera_direction_recommendation(
    dx: float | None,
    dy: float | None,
    geometric_inliers: int | None = None,
    min_verified_inliers: int = GEOMETRY_MIN_VERIFIED_INLIERS,
) -> str:
    """Translate content-shift direction into camera motion recommendation."""
    if (
        geometric_inliers is None
        or geometric_inliers < min_verified_inliers
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "picaivid-media-service",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "environment": settings.ENVIRONMENT,
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Picaivid Media Service",
        "version": "0.1.0",
    }


@app.post("/internal/jobs", response_model=JobStatusResponse)
async def create_job(
    message: JobMessage,
    db: Session = Depends(get_db),
):
    """Create a new job and start processing.

    This endpoint is called by Rails to trigger video generation.
    For local development, you can call this directly instead of SQS.
    """
    orchestrator = PipelineOrchestrator(db)
    job = orchestrator.create_job_from_message(message)

    # For development: run Phase 1 and Phase 2 synchronously
    if settings.ENVIRONMENT == "development":
        orchestrator.execute(job.id, allowed_phases=[1, 2])
        db.refresh(job)  # Get updated status after processing

    return JobStatusResponse(
        job_id=job.id,
        project_id=job.project_id,
        status=job.status,
        current_phase=job.current_phase,
        error_message=job.error_message,
    )


@app.get("/internal/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: int,
    db: Session = Depends(get_db),
):
    """Get job status."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job.id,
        project_id=job.project_id,
        status=job.status,
        current_phase=job.current_phase,
        error_message=job.error_message,
    )


@app.post("/internal/jobs/{job_id}/run-phase/{phase}")
async def run_phase(
    job_id: int,
    phase: int,
    db: Session = Depends(get_db),
):
    """Manually run a specific phase for a job.

    Useful for development and debugging.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    orchestrator = PipelineOrchestrator(db)
    orchestrator.execute(job_id, start_phase=phase, allowed_phases=[phase])

    db.refresh(job)
    return JobStatusResponse(
        job_id=job.id,
        project_id=job.project_id,
        status=job.status,
        current_phase=job.current_phase,
        error_message=job.error_message,
    )


@app.get("/api/projects/{project_id}/clips", response_model=ClipListResponse)
async def get_project_clips(
    project_id: str,
    db: Session = Depends(get_db),
):
    """Get all clips for a project.

    Returns clips with pre-signed URLs for video playback.
    This endpoint is called by the frontend to display generated videos.
    """
    from app.services.storage.s3_client import s3_client

    # Find the most recent job for this project
    job = (
        db.query(Job)
        .filter(Job.project_id == project_id)
        .order_by(Job.created_at.desc())
        .first()
    )

    if not job:
        return ClipListResponse(
            project_id=project_id,
            job_id=None,
            job_status=None,
            clips=[],
            total_clips=0,
        )

    # Get all clips for this job with related data
    clips = (
        db.query(Clip)
        .outerjoin(RoomCluster, Clip.room_cluster_id == RoomCluster.id)
        .filter(Clip.job_id == job.id)
        .order_by(RoomCluster.sequence_order.asc().nulls_last(), Clip.id.asc())
        .all()
    )

    # Build a map of job_photo_id -> JobPhoto for quick lookup
    job_photos = db.query(JobPhoto).filter(JobPhoto.job_id == job.id).all()
    photo_map = {p.id: p for p in job_photos}

    # Build a map of room_cluster_id -> (RoomCluster, AnalysisResult)
    clusters = db.query(RoomCluster).filter(RoomCluster.job_id == job.id).all()
    cluster_map = {c.id: c for c in clusters}

    analysis_results = db.query(AnalysisResult).filter(AnalysisResult.job_id == job.id).all()
    analysis_map = {a.room_cluster_id: a for a in analysis_results if a.room_cluster_id}

    # Similarity records are used to provide transition direction recommendations.
    try:
        similarities = db.query(PhotoSimilarity).filter(PhotoSimilarity.job_id == job.id).all()
    except ProgrammingError:
        db.rollback()
        legacy_rows = db.execute(
            text(
                """
                SELECT photo_a_id, photo_b_id, pair_source, dinov2_similarity,
                       geometric_inliers, geometric_score, is_connected
                FROM photo_similarities
                WHERE job_id = :job_id
                """
            ),
            {"job_id": job.id},
        ).fetchall()
        similarities = [
            SimpleNamespace(
                photo_a_id=int(r.photo_a_id),
                photo_b_id=int(r.photo_b_id),
                pair_source=r.pair_source,
                dinov2_similarity=r.dinov2_similarity,
                geometric_inliers=r.geometric_inliers,
                geometric_score=r.geometric_score,
                direction_dx=None,
                direction_dy=None,
                is_connected=int(r.is_connected or 0),
            )
            for r in legacy_rows
        ]

    sim_lookup = {}
    for sim in similarities:
        key = (min(sim.photo_a_id, sim.photo_b_id), max(sim.photo_a_id, sim.photo_b_id))
        sim_lookup[key] = sim

    # Generate pre-signed URLs for each clip
    clip_responses = []

    for clip in clips:
        video_url = None
        if clip.s3_uri:
            # Extract key from s3://bucket/key format
            if clip.s3_uri.startswith("s3://"):
                parts = clip.s3_uri[5:].split("/", 1)
                if len(parts) == 2:
                    key = parts[1]
                    video_url = s3_client.generate_presigned_url(key, expires_in=3600)

        # Get source photos info
        source_photos = []
        if clip.source_photo_ids:
            for photo_id in clip.source_photo_ids:
                photo = photo_map.get(photo_id)
                if photo:
                    # Generate thumbnail URL for source photo
                    thumb_url = None
                    if photo.s3_uri:
                        photo_key = photo.s3_uri[5:].split("/", 1)[1] if photo.s3_uri.startswith("s3://") else photo.s3_uri
                        thumb_url = s3_client.generate_presigned_url(photo_key, expires_in=3600)

                    source_photos.append(SourcePhotoInfo(
                        id=photo.id,
                        rails_photo_id=photo.rails_photo_id,
                        filename=photo.filename,
                        s3_uri=photo.s3_uri,
                        thumbnail_url=thumb_url,
                        room_label=photo.room_label,
                        room_override=photo.room_override,
                        position=photo.position,
                        cluster_order=photo.cluster_order,
                        base_score=photo.base_score,
                        final_score=photo.final_score,
                        depth_variance=photo.depth_variance,
                        depth_layers=photo.depth_layers,
                        sharpness=photo.sharpness,
                        exposure_score=photo.exposure_score,
                        composition_score=photo.composition_score,
                        is_duplicate=bool(photo.is_duplicate),
                        duplicate_of_photo_id=photo.duplicate_of_photo_id,
                    ))

        transition_steps = []
        if clip.source_photo_ids and len(clip.source_photo_ids) > 1:
            for idx in range(len(clip.source_photo_ids) - 1):
                from_photo_id = clip.source_photo_ids[idx]
                to_photo_id = clip.source_photo_ids[idx + 1]
                from_photo = photo_map.get(from_photo_id)
                to_photo = photo_map.get(to_photo_id)
                sim_key = (min(from_photo_id, to_photo_id), max(from_photo_id, to_photo_id))
                sim = sim_lookup.get(sim_key)
                ordered_dx, ordered_dy = _direction_for_order(from_photo_id, to_photo_id, sim)
                inliers = _safe_int(sim.geometric_inliers if sim else None)
                geo_score = _safe_float(sim.geometric_score if sim else None)
                geometric_verified = _is_geometrically_verified(
                    geometric_inliers=inliers,
                    geometric_score=geo_score,
                    dx=ordered_dx,
                    dy=ordered_dy,
                )
                overlap_from_zone, overlap_to_zone = _overlap_zones_from_direction(
                    ordered_dx if geometric_verified else None,
                    ordered_dy if geometric_verified else None,
                )
                overlap_summary = None
                if overlap_from_zone and overlap_to_zone:
                    overlap_summary = (
                        f"#{from_photo_id} {overlap_from_zone} overlaps with "
                        f"#{to_photo_id} {overlap_to_zone}"
                    )

                transition_steps.append(
                    ClipTransitionStep(
                        from_photo_id=from_photo_id,
                        to_photo_id=to_photo_id,
                        from_filename=from_photo.filename if from_photo else None,
                        to_filename=to_photo.filename if to_photo else None,
                        dinov2_similarity=sim.dinov2_similarity if sim else None,
                        geometric_inliers=inliers,
                        geometric_score=geo_score,
                        direction_dx=ordered_dx,
                        direction_dy=ordered_dy,
                        recommendation=_camera_direction_recommendation(
                            ordered_dx,
                            ordered_dy,
                            inliers,
                        ),
                        pair_source=sim.pair_source if sim else None,
                        is_connected=bool(sim.is_connected) if sim else False,
                        geometric_verified=geometric_verified,
                        overlap_from_zone=overlap_from_zone,
                        overlap_to_zone=overlap_to_zone,
                        overlap_summary=overlap_summary,
                    )
                )

        # Get cluster info
        cluster_info = None
        if clip.room_cluster_id and clip.room_cluster_id in cluster_map:
            cluster = cluster_map[clip.room_cluster_id]
            cluster_info = ClusterInfo(
                id=cluster.id,
                room_type=cluster.room_type,
                confidence_tier=cluster.confidence_tier,
                image_count=cluster.image_count or 0,
                overlap_score=cluster.overlap_score,
                depth_variance=cluster.depth_variance,
                recommended_motion=cluster.recommended_motion,
                sequence_order=cluster.sequence_order,
            )

        # Get analysis info
        analysis_info = None
        if clip.room_cluster_id and clip.room_cluster_id in analysis_map:
            analysis = analysis_map[clip.room_cluster_id]
            analysis_info = AnalysisInfo(
                tier=analysis.tier,
                recommended_motion=analysis.recommended_motion,
                model_recommendation=analysis.model_recommendation,
                cfg_scale=analysis.cfg_scale,
                inference_steps=analysis.inference_steps,
                debug_metrics=analysis.debug_metrics,
            )

        clip_responses.append(
            ClipResponse(
                id=clip.id,
                job_id=clip.job_id,
                room_cluster_id=clip.room_cluster_id,
                source_photo_ids=clip.source_photo_ids,
                motion_type=clip.motion_type,
                model_used=clip.model_used,
                is_3d=clip.is_3d or False,
                duration=clip.duration,
                prompt_used=clip.prompt_used,
                s3_uri=clip.s3_uri,
                video_url=video_url,
                status=clip.status,
                source_photos=source_photos if source_photos else None,
                transition_steps=transition_steps if transition_steps else None,
                cluster_info=cluster_info,
                analysis_info=analysis_info,
            )
        )

    return ClipListResponse(
        project_id=project_id,
        job_id=job.id,
        job_status=job.status,
        clips=clip_responses,
        total_clips=len(clip_responses),
    )


@app.get("/api/projects/{project_id}/clusters/debug", response_model=ClusterListDebugResponse)
async def get_project_clusters_debug(
    project_id: str,
    db: Session = Depends(get_db),
):
    """Get debug info for all clusters in a project.

    Shows why photos were grouped together:
    - DINOv2 semantic similarity scores
    - Geometric verification results (matches, inliers)
    - Whether connection was from temporal window or semantic top-k
    - Position in cluster (endpoint vs middle)
    """
    from app.services.storage.s3_client import s3_client

    # Find the most recent job for this project
    job = (
        db.query(Job)
        .filter(Job.project_id == project_id)
        .order_by(Job.created_at.desc())
        .first()
    )

    if not job:
        return ClusterListDebugResponse(
            project_id=project_id,
            job_id=None,
            clusters=[],
            total_clusters=0,
        )

    # Get all clips (each clip represents a cluster)
    clips = (
        db.query(Clip)
        .outerjoin(RoomCluster, Clip.room_cluster_id == RoomCluster.id)
        .filter(Clip.job_id == job.id)
        .order_by(RoomCluster.sequence_order.asc().nulls_last(), Clip.id.asc())
        .all()
    )

    # Get all photos for this job
    job_photos = db.query(JobPhoto).filter(JobPhoto.job_id == job.id).all()
    photo_map = {p.id: p for p in job_photos}
    photo_filename_map = {p.id: (p.filename or f"photo-{p.id}") for p in job_photos}

    # Get all similarity records for this job.
    # Fallback to legacy schema query when direction columns are not migrated yet.
    try:
        similarities = db.query(PhotoSimilarity).filter(PhotoSimilarity.job_id == job.id).all()
    except ProgrammingError:
        db.rollback()
        legacy_rows = db.execute(
            text(
                """
                SELECT photo_a_id, photo_b_id, pair_source, dinov2_similarity,
                       geometric_matches, geometric_inliers, geometric_score, is_connected
                FROM photo_similarities
                WHERE job_id = :job_id
                """
            ),
            {"job_id": job.id},
        ).fetchall()
        similarities = [
            SimpleNamespace(
                photo_a_id=int(r.photo_a_id),
                photo_b_id=int(r.photo_b_id),
                pair_source=r.pair_source,
                dinov2_similarity=r.dinov2_similarity,
                geometric_matches=r.geometric_matches,
                geometric_inliers=r.geometric_inliers,
                geometric_score=r.geometric_score,
                direction_dx=None,
                direction_dy=None,
                is_connected=int(r.is_connected or 0),
            )
            for r in legacy_rows
        ]

    # Build similarity lookup: (photo_a, photo_b) -> similarity record
    sim_lookup = {}
    for sim in similarities:
        key = (min(sim.photo_a_id, sim.photo_b_id), max(sim.photo_a_id, sim.photo_b_id))
        sim_lookup[key] = sim

    # Get clusters info
    clusters = db.query(RoomCluster).filter(RoomCluster.job_id == job.id).all()
    cluster_map = {c.id: c for c in clusters}

    cluster_responses = []

    for clip in clips:
        if not clip.source_photo_ids:
            continue

        photo_ids = clip.source_photo_ids
        cluster = cluster_map.get(clip.room_cluster_id) if clip.room_cluster_id else None

        # Build photo debug info
        photos_debug = []
        all_similarities = []

        for idx, photo_id in enumerate(photo_ids):
            photo = photo_map.get(photo_id)
            if not photo:
                continue

            # Generate thumbnail URL
            thumb_url = None
            if photo.s3_uri:
                photo_key = photo.s3_uri[5:].split("/", 1)[1] if photo.s3_uri.startswith("s3://") else photo.s3_uri
                thumb_url = s3_client.generate_presigned_url(photo_key, expires_in=3600)

            # Get similarities to other photos in this cluster
            photo_sims = []
            for other_id in photo_ids:
                if other_id == photo_id:
                    continue
                key = (min(photo_id, other_id), max(photo_id, other_id))
                sim = sim_lookup.get(key)
                if sim:
                    sim_dx = _safe_float(sim.direction_dx)
                    sim_dy = _safe_float(sim.direction_dy)
                    sim_inliers = _safe_int(sim.geometric_inliers)
                    sim_geo_score = _safe_float(sim.geometric_score)
                    sim_verified = _is_geometrically_verified(
                        geometric_inliers=sim_inliers,
                        geometric_score=sim_geo_score,
                        dx=sim_dx,
                        dy=sim_dy,
                    )
                    sim_overlap_from_zone, sim_overlap_to_zone = _overlap_zones_from_direction(
                        sim_dx if sim_verified else None,
                        sim_dy if sim_verified else None,
                    )
                    sim_overlap_summary = None
                    if sim_overlap_from_zone and sim_overlap_to_zone:
                        sim_overlap_summary = (
                            f"#{sim.photo_a_id} {sim_overlap_from_zone} overlaps with "
                            f"#{sim.photo_b_id} {sim_overlap_to_zone}"
                        )
                    photo_sims.append(PhotoSimilarityInfo(
                        photo_a_id=sim.photo_a_id,
                        photo_b_id=sim.photo_b_id,
                        photo_a_filename=photo_filename_map.get(sim.photo_a_id),
                        photo_b_filename=photo_filename_map.get(sim.photo_b_id),
                        pair_source=sim.pair_source,
                        dinov2_similarity=sim.dinov2_similarity,
                        geometric_matches=sim.geometric_matches,
                        geometric_inliers=sim_inliers,
                        geometric_score=sim_geo_score,
                        direction_dx=sim_dx,
                        direction_dy=sim_dy,
                        is_connected=bool(sim.is_connected),
                        geometric_verified=sim_verified,
                        overlap_from_zone=sim_overlap_from_zone,
                        overlap_to_zone=sim_overlap_to_zone,
                        overlap_summary=sim_overlap_summary,
                    ))
                    all_similarities.append(sim)

            photos_debug.append(PhotoDebugInfo(
                id=photo.id,
                rails_photo_id=photo.rails_photo_id,
                filename=photo.filename,
                thumbnail_url=thumb_url,
                room_label=photo.room_label,
                position_in_cluster=idx,
                is_endpoint=(idx == 0 or idx == len(photo_ids) - 1),
                is_duplicate=bool(photo.is_duplicate),
                duplicate_of_photo_id=photo.duplicate_of_photo_id,
                duplicate_of_filename=photo_filename_map.get(photo.duplicate_of_photo_id),
                similarities=photo_sims,
            ))

        # Compute summary stats
        avg_dinov2 = None
        avg_geo = None
        if all_similarities:
            dinov2_scores = [s.dinov2_similarity for s in all_similarities if s.dinov2_similarity is not None]
            geo_scores = [s.geometric_score for s in all_similarities if s.geometric_score is not None]
            if dinov2_scores:
                avg_dinov2 = sum(dinov2_scores) / len(dinov2_scores)
            if geo_scores:
                avg_geo = sum(geo_scores) / len(geo_scores)

        cluster_responses.append(ClusterDebugResponse(
            cluster_id=clip.room_cluster_id or clip.id,
            room_type=cluster.room_type if cluster else None,
            photo_ids=photo_ids,
            photo_filenames=[photo_filename_map.get(photo_id, f"photo-{photo_id}") for photo_id in photo_ids],
            photos=photos_debug,
            total_photos=len(photo_ids),
            sequence_order=cluster.sequence_order if cluster else None,
            avg_dinov2_similarity=avg_dinov2,
            avg_geometric_score=avg_geo,
            has_direction_info=any(s.geometric_inliers and s.geometric_inliers >= 3 for s in all_similarities),
        ))

    return ClusterListDebugResponse(
        project_id=project_id,
        job_id=job.id,
        clusters=cluster_responses,
        total_clusters=len(cluster_responses),
    )
