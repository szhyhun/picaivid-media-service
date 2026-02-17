"""FastAPI application for Picaivid Media Service."""
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.session import get_db
from app.db.models import Job, Clip, JobPhoto, RoomCluster, AnalysisResult, PhotoSimilarity
from app.schemas.job import JobMessage, JobStatusResponse
from app.schemas.clip import (
    ClipResponse, ClipListResponse, SourcePhotoInfo, ClusterInfo, AnalysisInfo,
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
    clips = db.query(Clip).filter(Clip.job_id == job.id).all()

    # Build a map of job_photo_id -> JobPhoto for quick lookup
    job_photos = db.query(JobPhoto).filter(JobPhoto.job_id == job.id).all()
    photo_map = {p.id: p for p in job_photos}

    # Build a map of room_cluster_id -> (RoomCluster, AnalysisResult)
    clusters = db.query(RoomCluster).filter(RoomCluster.job_id == job.id).all()
    cluster_map = {c.id: c for c in clusters}

    analysis_results = db.query(AnalysisResult).filter(AnalysisResult.job_id == job.id).all()
    analysis_map = {a.room_cluster_id: a for a in analysis_results if a.room_cluster_id}

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
                        s3_uri=photo.s3_uri,
                        thumbnail_url=thumb_url,
                        room_label=photo.room_label,
                        final_score=photo.final_score,
                        depth_variance=photo.depth_variance,
                        sharpness=photo.sharpness,
                    ))

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
    clips = db.query(Clip).filter(Clip.job_id == job.id).all()

    # Get all photos for this job
    job_photos = db.query(JobPhoto).filter(JobPhoto.job_id == job.id).all()
    photo_map = {p.id: p for p in job_photos}

    # Get all similarity records for this job
    similarities = db.query(PhotoSimilarity).filter(PhotoSimilarity.job_id == job.id).all()

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
                    photo_sims.append(PhotoSimilarityInfo(
                        photo_a_id=sim.photo_a_id,
                        photo_b_id=sim.photo_b_id,
                        pair_source=sim.pair_source,
                        dinov2_similarity=sim.dinov2_similarity,
                        geometric_matches=sim.geometric_matches,
                        geometric_inliers=sim.geometric_inliers,
                        geometric_score=sim.geometric_score,
                        is_connected=bool(sim.is_connected),
                    ))
                    all_similarities.append(sim)

            photos_debug.append(PhotoDebugInfo(
                id=photo.id,
                rails_photo_id=photo.rails_photo_id,
                thumbnail_url=thumb_url,
                room_label=photo.room_label,
                position_in_cluster=idx,
                is_endpoint=(idx == 0 or idx == len(photo_ids) - 1),
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
            photos=photos_debug,
            total_photos=len(photo_ids),
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
