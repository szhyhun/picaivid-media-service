"""FastAPI application for Picaivid Media Service."""
from datetime import datetime
import logging
from types import SimpleNamespace
import numpy as np

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
    PairDebugRequest, PairDebugResponse, PairDebugPhotoInfo, PairDebugStoredMetrics,
    PairDebugLiveMetrics, PairDebugPoint,
)
from app.pipeline.orchestrator import PipelineOrchestrator
from app.pipeline.phase1_analyze.learned_matching import match_image_pair

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Segment overlap metrics are expensive to compute (S3 + matcher per pair).
# Keep disabled for normal API reads; enable only when explicitly requested.
ENABLE_ON_DEMAND_SEGMENT_SCORES = False

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
    side_overlap: float | None = None,
    center_overlap: float | None = None,
    overlap_ratio: float | None = None,
    min_verified_inliers: int = GEOMETRY_MIN_VERIFIED_INLIERS,
    min_verified_score: float = GEOMETRY_MIN_VERIFIED_SCORE,
) -> bool:
    if geometric_inliers is None or geometric_inliers < min_verified_inliers:
        return False
    if geometric_score is not None and geometric_score < min_verified_score:
        return False
    if dx is None or dy is None:
        return False
    if (dx * dx + dy * dy) <= 1e-8:
        return False

    if side_overlap is not None or center_overlap is not None:
        side = side_overlap if side_overlap is not None else 0.0
        center = center_overlap if center_overlap is not None else 0.0
        if (
            side < settings.HARD_TRANSITION_MIN_SIDE_OVERLAP
            and center < settings.HARD_TRANSITION_MIN_CENTER_OVERLAP
        ):
            return False

    if (
        overlap_ratio is not None
        and overlap_ratio < settings.HARD_TRANSITION_MIN_OVERLAP_RATIO
    ):
        return False
    return True


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


def _coerce_segment_scores(raw_segment_scores: object) -> dict[str, float | None]:
    raw = raw_segment_scores if isinstance(raw_segment_scores, dict) else {}

    def f(key: str) -> float | None:
        value = raw.get(key)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "from_left_25_50_score": f("from_left_25_50"),
        "from_right_50_75_score": f("from_right_50_75"),
        "to_left_25_50_score": f("to_left_25_50"),
        "to_right_50_75_score": f("to_right_50_75"),
        "cross_left_to_right_score": f("cross_left_to_right"),
        "cross_right_to_left_score": f("cross_right_to_left"),
        "cross_center_to_center_score": f("cross_center_to_center"),
    }


def _segment_scores_from_similarity(sim: PhotoSimilarity | SimpleNamespace | None) -> dict[str, float | None]:
    if sim is None:
        return _coerce_segment_scores({})
    return {
        "from_left_25_50_score": _safe_float(getattr(sim, "from_left_25_50", None)),
        "from_right_50_75_score": _safe_float(getattr(sim, "from_right_50_75", None)),
        "to_left_25_50_score": _safe_float(getattr(sim, "to_left_25_50", None)),
        "to_right_50_75_score": _safe_float(getattr(sim, "to_right_50_75", None)),
        "cross_left_to_right_score": _safe_float(getattr(sim, "cross_left_to_right", None)),
        "cross_right_to_left_score": _safe_float(getattr(sim, "cross_right_to_left", None)),
        "cross_center_to_center_score": _safe_float(getattr(sim, "cross_center_to_center", None)),
    }


def _kornia_metrics_from_similarity(sim: PhotoSimilarity | SimpleNamespace | None) -> dict[str, float | bool | None]:
    if sim is None:
        return {
            "kornia_overlap_ratio": None,
            "kornia_side_overlap": None,
            "kornia_center_overlap": None,
            "kornia_inlier_ratio": None,
            "kornia_transition_overlap_ok": None,
        }
    raw_ok = getattr(sim, "kornia_transition_overlap_ok", None)
    ok: bool | None = None
    if raw_ok is not None:
        ok = bool(raw_ok)
    return {
        "kornia_overlap_ratio": _safe_float(getattr(sim, "kornia_overlap_ratio", None)),
        "kornia_side_overlap": _safe_float(getattr(sim, "kornia_side_overlap", None)),
        "kornia_center_overlap": _safe_float(getattr(sim, "kornia_center_overlap", None)),
        "kornia_inlier_ratio": _safe_float(getattr(sim, "kornia_inlier_ratio", None)),
        "kornia_transition_overlap_ok": ok,
    }


def _compute_pair_segment_scores(
    from_photo: JobPhoto | None,
    to_photo: JobPhoto | None,
    s3_client,
    cache: dict[tuple[int, int], dict[str, float | None]],
) -> dict[str, float | None]:
    if not ENABLE_ON_DEMAND_SEGMENT_SCORES:
        return _coerce_segment_scores({})

    if from_photo is None or to_photo is None:
        return _coerce_segment_scores({})

    # Keep orientation in cache key: segment metrics are directional.
    key = (from_photo.id, to_photo.id)
    if key in cache:
        return cache[key]

    try:
        img_from = s3_client.download_image(from_photo.s3_uri)
        img_to = s3_client.download_image(to_photo.s3_uri)
        _, _, _, _, diagnostics = match_image_pair(
            img_from,
            img_to,
            return_diagnostics=True,
        )
        segment_scores = _coerce_segment_scores(diagnostics.get("segment_scores"))
    except Exception as err:
        logger.debug(
            "Failed to compute segment scores for %s -> %s: %s",
            from_photo.id,
            to_photo.id,
            err,
        )
        segment_scores = _coerce_segment_scores({})

    cache[key] = segment_scores
    return segment_scores


def _s3_key_from_uri(s3_uri: str | None) -> str | None:
    if not s3_uri:
        return None
    if s3_uri.startswith("s3://"):
        parts = s3_uri[5:].split("/", 1)
        if len(parts) == 2:
            return parts[1]
        return None
    return s3_uri


def _sample_points(points: list[dict[str, object]], sample_limit: int) -> list[dict[str, object]]:
    if sample_limit <= 0 or len(points) <= sample_limit:
        return points
    indexes = [int(i) for i in np.linspace(0, len(points) - 1, num=sample_limit)]
    return [points[i] for i in indexes]


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
    pair_segment_cache: dict[tuple[int, int], dict[str, float | None]] = {}

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
                segment_scores = _segment_scores_from_similarity(sim)
                if (
                    segment_scores["cross_left_to_right_score"] is None
                    and segment_scores["cross_right_to_left_score"] is None
                    and segment_scores["cross_center_to_center_score"] is None
                ):
                    segment_scores = _compute_pair_segment_scores(
                        from_photo=from_photo,
                        to_photo=to_photo,
                        s3_client=s3_client,
                        cache=pair_segment_cache,
                    )
                kornia_metrics = _kornia_metrics_from_similarity(sim)
                ordered_dx, ordered_dy = _direction_for_order(from_photo_id, to_photo_id, sim)
                inliers = _safe_int(sim.geometric_inliers if sim else None)
                geo_score = _safe_float(sim.geometric_score if sim else None)
                geometric_verified = _is_geometrically_verified(
                    geometric_inliers=inliers,
                    geometric_score=geo_score,
                    dx=ordered_dx,
                    dy=ordered_dy,
                    side_overlap=max(
                        segment_scores["cross_left_to_right_score"] or 0.0,
                        segment_scores["cross_right_to_left_score"] or 0.0,
                    ),
                    center_overlap=segment_scores["cross_center_to_center_score"],
                    overlap_ratio=_safe_float(kornia_metrics["kornia_overlap_ratio"]),
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
                        from_left_25_50_score=segment_scores["from_left_25_50_score"],
                        from_right_50_75_score=segment_scores["from_right_50_75_score"],
                        to_left_25_50_score=segment_scores["to_left_25_50_score"],
                        to_right_50_75_score=segment_scores["to_right_50_75_score"],
                        cross_left_to_right_score=segment_scores["cross_left_to_right_score"],
                        cross_right_to_left_score=segment_scores["cross_right_to_left_score"],
                        cross_center_to_center_score=segment_scores["cross_center_to_center_score"],
                        kornia_overlap_ratio=kornia_metrics["kornia_overlap_ratio"],
                        kornia_side_overlap=kornia_metrics["kornia_side_overlap"],
                        kornia_center_overlap=kornia_metrics["kornia_center_overlap"],
                        kornia_inlier_ratio=kornia_metrics["kornia_inlier_ratio"],
                        kornia_transition_overlap_ok=kornia_metrics["kornia_transition_overlap_ok"],
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


@app.post("/api/projects/{project_id}/pairs/debug", response_model=PairDebugResponse)
async def debug_pair_geometry(
    project_id: str,
    payload: PairDebugRequest,
    db: Session = Depends(get_db),
):
    """Run on-demand matching for two job photo IDs and return full geometry diagnostics."""
    from app.services.storage.s3_client import s3_client

    if payload.left_photo_id == payload.right_photo_id:
        raise HTTPException(status_code=400, detail="left_photo_id and right_photo_id must be different")

    raw_limit = payload.sample_limit
    if raw_limit is None:
        sample_limit = 250
    else:
        sample_limit = max(int(raw_limit), 0)

    job_query = db.query(Job).filter(Job.project_id == project_id)
    if payload.job_id is not None:
        job_query = job_query.filter(Job.id == payload.job_id)
    job = job_query.order_by(Job.created_at.desc()).first()
    if not job:
        raise HTTPException(status_code=404, detail="No job found for project")

    left_photo = (
        db.query(JobPhoto)
        .filter(JobPhoto.job_id == job.id, JobPhoto.id == payload.left_photo_id)
        .first()
    )
    right_photo = (
        db.query(JobPhoto)
        .filter(JobPhoto.job_id == job.id, JobPhoto.id == payload.right_photo_id)
        .first()
    )
    if not left_photo or not right_photo:
        raise HTTPException(status_code=404, detail=f"Photo IDs must belong to job {job.id}")

    try:
        left_image = s3_client.download_image(left_photo.s3_uri)
        right_image = s3_client.download_image(right_photo.s3_uri)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to load pair images: {err}") from err

    try:
        num_matches, num_inliers, score, direction, diagnostics = match_image_pair(
            left_image,
            right_image,
            return_diagnostics=True,
            debug_options={
                "matcher": payload.matcher,
                "confidence_threshold": payload.confidence_threshold,
                "full_diagnostics": True,
            },
        )
    except Exception as err:
        message = str(err)
        logger.exception(
            "Pair debug matcher failure (project=%s, job=%s, left_photo_id=%s, right_photo_id=%s, matcher=%s): %s",
            project_id,
            job.id if job else None,
            payload.left_photo_id,
            payload.right_photo_id,
            payload.matcher,
            message,
        )
        raise HTTPException(status_code=500, detail=f"Pair debug matcher failed: {err}") from err
    if not isinstance(diagnostics, dict):
        diagnostics = {}

    raw_points = diagnostics.get("raw_matches")
    filter_match_sets = diagnostics.get("filter_match_sets")
    filtered_points = diagnostics.get("filtered_matches")
    inlier_points = diagnostics.get("inlier_matches")
    raw_points_list = raw_points if isinstance(raw_points, list) else []
    filtered_points_list = filtered_points if isinstance(filtered_points, list) else []
    inlier_points_list = inlier_points if isinstance(inlier_points, list) else []
    sampled_raw_points = _sample_points(raw_points_list, sample_limit)
    sampled_filtered_points = _sample_points(filtered_points_list, sample_limit)
    sampled_inlier_points = _sample_points(inlier_points_list, sample_limit)

    sim = (
        db.query(PhotoSimilarity)
        .filter(PhotoSimilarity.job_id == job.id)
        .filter(
            PhotoSimilarity.photo_a_id == min(left_photo.id, right_photo.id),
            PhotoSimilarity.photo_b_id == max(left_photo.id, right_photo.id),
        )
        .first()
    )

    stored_metrics = None
    if sim is not None:
        stored_metrics = PairDebugStoredMetrics(
            pair_source=sim.pair_source,
            dinov2_similarity=_safe_float(sim.dinov2_similarity),
            geometric_matches=_safe_int(sim.geometric_matches),
            geometric_inliers=_safe_int(sim.geometric_inliers),
            geometric_score=_safe_float(sim.geometric_score),
            direction_dx=_safe_float(sim.direction_dx),
            direction_dy=_safe_float(sim.direction_dy),
            cross_left_to_right=_safe_float(getattr(sim, "cross_left_to_right", None)),
            cross_right_to_left=_safe_float(getattr(sim, "cross_right_to_left", None)),
            cross_center_to_center=_safe_float(getattr(sim, "cross_center_to_center", None)),
            kornia_overlap_ratio=_safe_float(getattr(sim, "kornia_overlap_ratio", None)),
            kornia_side_overlap=_safe_float(getattr(sim, "kornia_side_overlap", None)),
            kornia_center_overlap=_safe_float(getattr(sim, "kornia_center_overlap", None)),
            kornia_inlier_ratio=_safe_float(getattr(sim, "kornia_inlier_ratio", None)),
            kornia_transition_overlap_ok=(
                bool(getattr(sim, "kornia_transition_overlap_ok"))
                if getattr(sim, "kornia_transition_overlap_ok", None) is not None
                else None
            ),
            is_connected=bool(sim.is_connected) if sim.is_connected is not None else None,
        )

    left_key = _s3_key_from_uri(left_photo.s3_uri)
    right_key = _s3_key_from_uri(right_photo.s3_uri)

    live_metrics = PairDebugLiveMetrics(
        matcher=str(diagnostics.get("matcher")) if diagnostics.get("matcher") is not None else None,
        checkpoint=str(diagnostics.get("checkpoint")) if diagnostics.get("checkpoint") is not None else None,
        confidence_threshold=_safe_float(diagnostics.get("confidence_threshold")),
        geometry_model=str(diagnostics.get("geometry_model")) if diagnostics.get("geometry_model") is not None else None,
        raw_correspondence_count=_safe_int(diagnostics.get("raw_correspondence_count")),
        raw_matches=[
            PairDebugPoint(
                x0=float(p.get("x0", 0.0)),
                y0=float(p.get("y0", 0.0)),
                x1=float(p.get("x1", 0.0)),
                y1=float(p.get("y1", 0.0)),
                dx=float(p.get("dx", 0.0)),
                dy=float(p.get("dy", 0.0)),
            )
            for p in sampled_raw_points
        ],
        filter_match_sets={
            str(name): [
                PairDebugPoint(
                    x0=float(p.get("x0", 0.0)),
                    y0=float(p.get("y0", 0.0)),
                    x1=float(p.get("x1", 0.0)),
                    y1=float(p.get("y1", 0.0)),
                    dx=float(p.get("dx", 0.0)),
                    dy=float(p.get("dy", 0.0)),
                )
                for p in _sample_points(points if isinstance(points, list) else [], sample_limit)
            ]
            for name, points in (filter_match_sets.items() if isinstance(filter_match_sets, dict) else [])
        },
        filtered_match_count=len(filtered_points_list),
        filtered_matches=[
            PairDebugPoint(
                x0=float(p.get("x0", 0.0)),
                y0=float(p.get("y0", 0.0)),
                x1=float(p.get("x1", 0.0)),
                y1=float(p.get("y1", 0.0)),
                dx=float(p.get("dx", 0.0)),
                dy=float(p.get("dy", 0.0)),
            )
            for p in sampled_filtered_points
        ],
        filtered_final_score=_safe_float(diagnostics.get("filtered_final_score")),
        threshold_trials=[t for t in (diagnostics.get("threshold_trials") or []) if isinstance(t, dict)],
        loftr_input_width=_safe_int(diagnostics.get("loftr_input_width")),
        loftr_input_height=_safe_int(diagnostics.get("loftr_input_height")),
        ransac_reproj_threshold=_safe_float(diagnostics.get("ransac_reproj_threshold")),
        num_matches=int(num_matches),
        num_inliers=int(num_inliers),
        geometric_score=float(score),
        direction_dx=_safe_float(direction[0]) if direction else None,
        direction_dy=_safe_float(direction[1]) if direction else None,
        match_width=_safe_int(diagnostics.get("match_width")),
        match_height=_safe_int(diagnostics.get("match_height")),
        segment_scores={
            str(k): float(v) for k, v in (diagnostics.get("segment_scores") or {}).items() if v is not None
        },
        score_components={
            str(k): float(v) for k, v in (diagnostics.get("score_components") or {}).items() if v is not None
        },
        oracle=(diagnostics.get("oracle") or {}) if isinstance(diagnostics.get("oracle"), dict) else {},
        filter_config=(diagnostics.get("filter_config") or {}) if isinstance(diagnostics.get("filter_config"), dict) else {},
        filter_scores=(diagnostics.get("filter_scores") or {}) if isinstance(diagnostics.get("filter_scores"), dict) else {},
        native_matching_scores={
            str(k): float(v) for k, v in (diagnostics.get("native_matching_scores") or {}).items() if v is not None
        },
        native_matching_scores_raw={
            str(k): float(v) for k, v in (diagnostics.get("native_matching_scores_raw") or {}).items() if v is not None
        },
        zju_variant=(
            str(diagnostics.get("zju_variant"))
            if diagnostics.get("zju_variant") is not None
            else None
        ),
        zju_loader=(
            str(diagnostics.get("zju_loader"))
            if diagnostics.get("zju_loader") is not None
            else None
        ),
        zju_checkpoint_path=(
            str(diagnostics.get("zju_checkpoint_path"))
            if diagnostics.get("zju_checkpoint_path") is not None
            else None
        ),
        zju_repo_dir=(
            str(diagnostics.get("zju_repo_dir"))
            if diagnostics.get("zju_repo_dir") is not None
            else None
        ),
        zju_match_type=(
            str(diagnostics.get("zju_match_type"))
            if diagnostics.get("zju_match_type") is not None
            else None
        ),
        zju_model_class=(
            str(diagnostics.get("zju_model_class"))
            if diagnostics.get("zju_model_class") is not None
            else None
        ),
        hf_visualization_data_url=(
            str(diagnostics.get("hf_visualization_data_url"))
            if diagnostics.get("hf_visualization_data_url") is not None
            else None
        ),
        inlier_match_count=len(inlier_points_list),
        inlier_matches=[
            PairDebugPoint(
                x0=float(p.get("x0", 0.0)),
                y0=float(p.get("y0", 0.0)),
                x1=float(p.get("x1", 0.0)),
                y1=float(p.get("y1", 0.0)),
                dx=float(p.get("dx", 0.0)),
                dy=float(p.get("dy", 0.0)),
            )
            for p in sampled_inlier_points
        ],
    )

    return PairDebugResponse(
        project_id=project_id,
        job_id=job.id,
        left_photo=PairDebugPhotoInfo(
            id=left_photo.id,
            rails_photo_id=left_photo.rails_photo_id,
            filename=left_photo.filename,
            room_label=left_photo.room_label,
            position=left_photo.position,
            s3_uri=left_photo.s3_uri,
            image_url=s3_client.generate_presigned_url(left_key, expires_in=3600) if left_key else None,
            thumbnail_url=s3_client.generate_presigned_url(left_key, expires_in=3600) if left_key else None,
        ),
        right_photo=PairDebugPhotoInfo(
            id=right_photo.id,
            rails_photo_id=right_photo.rails_photo_id,
            filename=right_photo.filename,
            room_label=right_photo.room_label,
            position=right_photo.position,
            s3_uri=right_photo.s3_uri,
            image_url=s3_client.generate_presigned_url(right_key, expires_in=3600) if right_key else None,
            thumbnail_url=s3_client.generate_presigned_url(right_key, expires_in=3600) if right_key else None,
        ),
        stored_metrics=stored_metrics,
        live_metrics=live_metrics,
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
    pair_segment_cache: dict[tuple[int, int], dict[str, float | None]] = {}

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
                    segment_scores = _segment_scores_from_similarity(sim)
                    if (
                        segment_scores["cross_left_to_right_score"] is None
                        and segment_scores["cross_right_to_left_score"] is None
                        and segment_scores["cross_center_to_center_score"] is None
                    ):
                        segment_scores = _compute_pair_segment_scores(
                            from_photo=photo_map.get(sim.photo_a_id),
                            to_photo=photo_map.get(sim.photo_b_id),
                            s3_client=s3_client,
                            cache=pair_segment_cache,
                        )
                    kornia_metrics = _kornia_metrics_from_similarity(sim)
                    sim_dx = _safe_float(sim.direction_dx)
                    sim_dy = _safe_float(sim.direction_dy)
                    sim_inliers = _safe_int(sim.geometric_inliers)
                    sim_geo_score = _safe_float(sim.geometric_score)
                    sim_verified = _is_geometrically_verified(
                        geometric_inliers=sim_inliers,
                        geometric_score=sim_geo_score,
                        dx=sim_dx,
                        dy=sim_dy,
                        side_overlap=max(
                            segment_scores["cross_left_to_right_score"] or 0.0,
                            segment_scores["cross_right_to_left_score"] or 0.0,
                        ),
                        center_overlap=segment_scores["cross_center_to_center_score"],
                        overlap_ratio=_safe_float(kornia_metrics["kornia_overlap_ratio"]),
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
                        from_left_25_50_score=segment_scores["from_left_25_50_score"],
                        from_right_50_75_score=segment_scores["from_right_50_75_score"],
                        to_left_25_50_score=segment_scores["to_left_25_50_score"],
                        to_right_50_75_score=segment_scores["to_right_50_75_score"],
                        cross_left_to_right_score=segment_scores["cross_left_to_right_score"],
                        cross_right_to_left_score=segment_scores["cross_right_to_left_score"],
                        cross_center_to_center_score=segment_scores["cross_center_to_center_score"],
                        kornia_overlap_ratio=kornia_metrics["kornia_overlap_ratio"],
                        kornia_side_overlap=kornia_metrics["kornia_side_overlap"],
                        kornia_center_overlap=kornia_metrics["kornia_center_overlap"],
                        kornia_inlier_ratio=kornia_metrics["kornia_inlier_ratio"],
                        kornia_transition_overlap_ok=kornia_metrics["kornia_transition_overlap_ok"],
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
