"""FastAPI application for Picaivid Media Service."""
from datetime import datetime
import logging
import time
import numpy as np

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.session import get_db
from app.db.models import Job, Clip, JobPhoto, RoomCluster, AnalysisResult, PhotoSimilarity, PhotoPoseAlignment, TransitionSequence
from app.schemas.job import JobMessage, JobStatusResponse
from app.schemas.clip import (
    ClipResponse, ClipListResponse, SourcePhotoInfo, ClusterInfo, AnalysisInfo,
    ClipTransitionStep, TransitionSequenceInfo, TransitionSequenceStepInfo,
    ClusterDebugResponse, ClusterListDebugResponse, PhotoDebugInfo, PhotoSimilarityInfo,
    PairDebugRequest, PairDebugResponse, PairDebugPhotoInfo, PairDebugStoredMetrics,
    PairDebugLiveMetrics, PairDebugPoint, SemanticRegionInfo, ClusterSuggestionInfo,
)
from app.pipeline.orchestrator import PipelineOrchestrator
from app.pipeline.phase1_analyze.mast3r_pipeline import MAST3R_ENGINE_NAME, debug_pair_mast3r
from app.models.warmup import warmup_core_models

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


@app.on_event("startup")
async def startup_warm_models() -> None:
    warmup_core_models(context="api", include_mast3r=True, include_legacy=False)

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


def _pair_debug_point_from_dict(point: dict[str, object]) -> PairDebugPoint:
    return PairDebugPoint(
        x0=float(point.get("x0", 0.0)),
        y0=float(point.get("y0", 0.0)),
        x1=float(point.get("x1", 0.0)),
        y1=float(point.get("y1", 0.0)),
        dx=float(point.get("dx", 0.0)),
        dy=float(point.get("dy", 0.0)),
        label_a=str(point.get("label_a")) if point.get("label_a") is not None else None,
        label_b=str(point.get("label_b")) if point.get("label_b") is not None else None,
        region_id_a=_safe_int(point.get("region_id_a")),
        region_id_b=_safe_int(point.get("region_id_b")),
        region_type_a=str(point.get("region_type_a")) if point.get("region_type_a") is not None else None,
        region_type_b=str(point.get("region_type_b")) if point.get("region_type_b") is not None else None,
        is_anchor_match=bool(point.get("is_anchor_match")) if point.get("is_anchor_match") is not None else None,
        is_window_match=bool(point.get("is_window_match")) if point.get("is_window_match") is not None else None,
        is_background_match=bool(point.get("is_background_match")) if point.get("is_background_match") is not None else None,
        is_object_match=bool(point.get("is_object_match")) if point.get("is_object_match") is not None else None,
        same_associated_object_match=bool(point.get("same_associated_object_match")) if point.get("same_associated_object_match") is not None else None,
        anchor_object_match=bool(point.get("anchor_object_match")) if point.get("anchor_object_match") is not None else None,
        cross_object_match=bool(point.get("cross_object_match")) if point.get("cross_object_match") is not None else None,
        association_score=_safe_float(point.get("association_score")),
        semantic_accept=bool(point.get("semantic_accept")) if point.get("semantic_accept") is not None else None,
    )


def _side_shift_label(side_a: int | None, side_b: int | None) -> str | None:
    if side_a is None or side_b is None:
        return None
    labels = {0: "LEFT", 1: "CENTER", 2: "RIGHT"}
    return f"{labels.get(int(side_a), '?')} -> {labels.get(int(side_b), '?')}"


def _serialize_transition_sequence(sequence: TransitionSequence) -> TransitionSequenceInfo:
    photo_ids = [int(step.photo_id) for step in sequence.steps]
    return TransitionSequenceInfo(
        sequence_rank=int(sequence.sequence_rank or 0),
        sequence_score=float(sequence.sequence_score or 0.0),
        certification_status=str(sequence.certification_status or "usable"),
        room_type_hint=sequence.room_type_hint,
        source_cluster_ids=[int(value) for value in (sequence.source_cluster_ids or [])],
        motion_hint=sequence.motion_hint,
        photo_ids=photo_ids,
        steps=[
            TransitionSequenceStepInfo(
                step_index=int(step.step_index or 0),
                photo_id=int(step.photo_id),
                photo_similarity_id=_safe_int(step.photo_similarity_id),
            )
            for step in sequence.steps
        ],
    )


def _build_pair_debug_strict_gate(
    num_matches: int | None,
    num_inliers: int | None,
    geometric_score: float | None,
    diagnostics: dict | None,
    min_inliers_required: int | None = None,
) -> dict:
    """Mirror MASt3R pair acceptance diagnostics."""
    has_diagnostics = isinstance(diagnostics, dict)
    retrieval_score = float((diagnostics or {}).get("retrieval_score") or 0.0)
    reciprocal_match_count = int((diagnostics or {}).get("reciprocal_match_count") or num_matches or 0)
    pointmap_consistency = float((diagnostics or {}).get("pointmap_consistency") or 0.0)
    parallax_score = float((diagnostics or {}).get("parallax_score") or 0.0)
    graph_edge_score = float((diagnostics or {}).get("graph_edge_score") or geometric_score or 0.0)
    geometry_model = str((diagnostics or {}).get("geometry_model") or "").strip().lower()

    required_matches = int(min_inliers_required or settings.MAST3R_MIN_RECIPROCAL_MATCHES)
    checks = [
        ("has_counts", reciprocal_match_count >= 0),
        ("min_retrieval_score", retrieval_score >= float(settings.MAST3R_MIN_RETRIEVAL_SCORE)),
        ("min_reciprocal_matches", reciprocal_match_count >= required_matches),
        ("min_pointmap_consistency", pointmap_consistency >= float(settings.MAST3R_MIN_POINTMAP_CONSISTENCY)),
        ("min_parallax_score", parallax_score >= float(settings.MAST3R_MIN_PARALLAX_SCORE)),
        ("min_graph_edge_score", graph_edge_score >= float(settings.MAST3R_MIN_GRAPH_EDGE_SCORE)),
        ("has_diagnostics", has_diagnostics),
        ("geometry_model_allowed", geometry_model == "mast3r_pointmap_consistency"),
    ]

    fail_reason = "passed"
    for reason, passed in checks:
        if not passed:
            fail_reason = reason
            break

    return {
        "would_connect": fail_reason == "passed",
        "reason": fail_reason,
        "required": {
            "min_reciprocal_matches": required_matches,
            "min_retrieval_score": float(settings.MAST3R_MIN_RETRIEVAL_SCORE),
            "min_pointmap_consistency": float(settings.MAST3R_MIN_POINTMAP_CONSISTENCY),
            "min_parallax_score": float(settings.MAST3R_MIN_PARALLAX_SCORE),
            "min_graph_edge_score": float(settings.MAST3R_MIN_GRAPH_EDGE_SCORE),
            "allowed_geometry_models": ["mast3r_pointmap_consistency"],
        },
        "actual": {
            "num_matches": int(num_matches or 0),
            "num_inliers": int(num_inliers or 0),
            "retrieval_score": float(retrieval_score),
            "reciprocal_match_count": int(reciprocal_match_count),
            "pointmap_consistency": float(pointmap_consistency),
            "parallax_score": float(parallax_score),
            "graph_edge_score": float(graph_edge_score),
            "geometric_score": float(geometric_score or 0.0),
            "geometry_model": geometry_model or "none",
        },
        "checks": {reason: bool(passed) for reason, passed in checks},
    }


def _direction_for_order(
    from_photo_id: int,
    to_photo_id: int,
    sim: PhotoSimilarity | None,
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


def _segment_scores_from_similarity(sim: PhotoSimilarity | None) -> dict[str, float | None]:
    return _coerce_segment_scores({})


def _compute_pair_segment_scores(
    from_photo: JobPhoto | None,
    to_photo: JobPhoto | None,
    s3_client,
    cache: dict[tuple[int, int], dict[str, float | None]],
) -> dict[str, float | None]:
    return _coerce_segment_scores({})


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


def _primary_graph_component_id(photo_ids: list[int], component_by_photo: dict[int, int]) -> int | None:
    counts: dict[int, int] = {}
    for photo_id in photo_ids:
        component_id = component_by_photo.get(int(photo_id))
        if component_id is None:
            continue
        counts[component_id] = counts.get(component_id, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]


def _cluster_suggestions(
    cluster_photo_ids: list[int],
    cluster_id: int,
    component_by_photo: dict[int, int],
    photos_by_component: dict[int, list[int]],
    sim_lookup: dict[tuple[int, int], PhotoSimilarity],
    photo_map: dict[int, JobPhoto],
    paired_cluster_for_photo: dict[int, int],
) -> list[ClusterSuggestionInfo]:
    component_id = _primary_graph_component_id(cluster_photo_ids, component_by_photo)
    if component_id is None:
        return []

    chosen_edge_score = None
    if len(cluster_photo_ids) == 2:
        chosen_sim = sim_lookup.get((min(cluster_photo_ids), max(cluster_photo_ids)))
        if chosen_sim is not None:
            chosen_edge_score = (
                _safe_float(getattr(chosen_sim, "graph_edge_score", None))
                or _safe_float(getattr(chosen_sim, "pair_rank", None))
                or 0.0
            )

    suggestions: list[ClusterSuggestionInfo] = []
    cluster_set = {int(photo_id) for photo_id in cluster_photo_ids}
    for candidate_id in photos_by_component.get(int(component_id), []):
        candidate_id = int(candidate_id)
        if candidate_id in cluster_set:
            continue

        best_related_photo_id = None
        best_score = None
        for selected_photo_id in cluster_photo_ids:
            sim = sim_lookup.get((min(int(selected_photo_id), candidate_id), max(int(selected_photo_id), candidate_id)))
            if sim is None:
                continue
            edge_score = (
                _safe_float(getattr(sim, "graph_edge_score", None))
                or _safe_float(getattr(sim, "pair_rank", None))
                or 0.0
            )
            if best_score is None or edge_score > best_score:
                best_score = edge_score
                best_related_photo_id = int(selected_photo_id)

        if best_score is None:
            continue

        if candidate_id in paired_cluster_for_photo and paired_cluster_for_photo[candidate_id] != int(cluster_id):
            reason = "already consumed by a stronger pair"
        elif best_score < float(settings.MAST3R_SUGGESTION_MIN_EDGE_SCORE):
            reason = "failed pair safety threshold"
        elif chosen_edge_score is not None and best_score < chosen_edge_score:
            reason = "lower than chosen edge"
        else:
            reason = "failed pair safety threshold"

        candidate_photo = photo_map.get(candidate_id)
        suggestions.append(
            ClusterSuggestionInfo(
                photo_id=candidate_id,
                filename=candidate_photo.filename if candidate_photo is not None else None,
                room_label=(candidate_photo.room_override or candidate_photo.room_label) if candidate_photo is not None else None,
                score=float(best_score),
                related_to_photo_id=best_related_photo_id,
                reason=reason,
            )
        )

    suggestions.sort(key=lambda item: (float(item.score or 0.0), -int(item.photo_id)), reverse=True)
    return suggestions[:8]


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

    # Similarity records are MASt3R graph edges used for transition direction recommendations.
    similarities = db.query(PhotoSimilarity).filter(PhotoSimilarity.job_id == job.id).all()

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
                ordered_dx, ordered_dy = _direction_for_order(from_photo_id, to_photo_id, sim)
                inliers = _safe_int(sim.reciprocal_match_count if sim else None)
                geo_score = _safe_float(sim.graph_edge_score if sim else None)
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
                    overlap_ratio=_safe_float(sim.overlap_ratio if sim else None),
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
                        geometric_inliers=inliers,
                        geometric_score=geo_score,
                        pair_rank=_safe_float(sim.pair_rank if sim else None),
                        certification_status=sim.certification_status if sim else None,
                        rejection_reason=sim.rejection_reason if sim else None,
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
    request_started_at = time.perf_counter()
    logger.info(
        "pair_debug start project=%s job_id=%s left_photo_id=%s right_photo_id=%s matcher=%s threshold=%.3f sample_limit=%s",
        project_id,
        payload.job_id,
        payload.left_photo_id,
        payload.right_photo_id,
        payload.matcher,
        float(payload.confidence_threshold),
        payload.sample_limit,
    )

    if payload.left_photo_id == payload.right_photo_id:
        raise HTTPException(status_code=400, detail="left_photo_id and right_photo_id must be different")

    raw_limit = payload.sample_limit
    if raw_limit is None:
        sample_limit = 1000
    else:
        sample_limit = max(int(raw_limit), 0)

    lookup_started_at = time.perf_counter()
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
        ownership_issues: list[str] = []
        for side, pid, scoped_photo in (
            ("left_photo_id", payload.left_photo_id, left_photo),
            ("right_photo_id", payload.right_photo_id, right_photo),
        ):
            if scoped_photo is not None:
                continue
            owner = (
                db.query(JobPhoto.id, JobPhoto.job_id, Job.project_id)
                .join(Job, Job.id == JobPhoto.job_id)
                .filter(JobPhoto.id == pid)
                .first()
            )
            if owner is None:
                ownership_issues.append(
                    f"{side}={pid} not found in media-service DB"
                )
                continue
            owner_project_id = str(owner.project_id)
            owner_job_id = int(owner.job_id)
            if owner_project_id != project_id:
                ownership_issues.append(
                    f"{side}={pid} belongs to project {owner_project_id} (job {owner_job_id}), "
                    f"not requested project {project_id} (job {job.id})"
                )
            else:
                ownership_issues.append(
                    f"{side}={pid} belongs to job {owner_job_id} in project {project_id}, "
                    f"but requested debug job is {job.id}"
                )
        raise HTTPException(status_code=422, detail="; ".join(ownership_issues))
    lookup_seconds = time.perf_counter() - lookup_started_at

    try:
        s3_started_at = time.perf_counter()
        left_s3_started_at = time.perf_counter()
        left_image = s3_client.download_image(left_photo.s3_uri)
        left_s3_seconds = time.perf_counter() - left_s3_started_at
        right_s3_started_at = time.perf_counter()
        right_image = s3_client.download_image(right_photo.s3_uri)
        right_s3_seconds = time.perf_counter() - right_s3_started_at
        s3_total_seconds = time.perf_counter() - s3_started_at
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to load pair images: {err}") from err

    try:
        matcher_started_at = time.perf_counter()
        normalized_matcher = str(payload.matcher or "").strip().lower()
        if normalized_matcher in {"", "current", "default", MAST3R_ENGINE_NAME}:
            num_matches, num_inliers, score, direction, diagnostics = debug_pair_mast3r(
                left_image,
                right_image,
            )
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported matcher '{payload.matcher}'. MASt3R is the only active pair-debug matcher.",
            )
        matcher_seconds = time.perf_counter() - matcher_started_at
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
        lowered = message.lower()
        if any(
            token in lowered
            for token in (
                "repo path not found",
                "checkpoint not found",
                "codebook",
                "missing env",
                "failed importing repo modules",
                "requires a cuda gpu worker",
                "required file not found locally",
                "required repo not found locally",
            )
        ):
            raise HTTPException(status_code=422, detail=f"Selected debug matcher is unavailable locally: {err}") from err
        raise HTTPException(status_code=500, detail=f"Pair debug matcher failed: {err}") from err
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    timing_diagnostics = diagnostics.get("timing") if isinstance(diagnostics.get("timing"), dict) else {}
    pair_model_seconds = float(timing_diagnostics.get("time_pair_total_s", 0.0) or 0.0)
    mast3r_inference_seconds = float(timing_diagnostics.get("time_mast3r_inference_s", 0.0) or 0.0)
    postprocess_started_at = time.perf_counter()

    raw_points = diagnostics.get("raw_matches")
    inlier_points = diagnostics.get("inlier_matches")
    raw_points_list = raw_points if isinstance(raw_points, list) else []
    inlier_points_list = inlier_points if isinstance(inlier_points, list) else []
    sampled_raw_points = _sample_points(raw_points_list, sample_limit)
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
            match_engine=getattr(sim, "match_engine", None),
            retrieval_score=_safe_float(getattr(sim, "retrieval_score", None)),
            reciprocal_match_count=_safe_int(getattr(sim, "reciprocal_match_count", None)),
            pointmap_consistency=_safe_float(getattr(sim, "pointmap_consistency", None)),
            alignment_residual=_safe_float(getattr(sim, "alignment_residual", None)),
            reprojection_error=_safe_float(getattr(sim, "reprojection_error", None)),
            parallax_score=_safe_float(getattr(sim, "parallax_score", None)),
            graph_component_id=_safe_int(getattr(sim, "graph_component_id", None)),
            graph_edge_score=_safe_float(getattr(sim, "graph_edge_score", None)),
            overlap_ratio=_safe_float(getattr(sim, "overlap_ratio", None)),
            combined_geometry_score=_safe_float(getattr(sim, "combined_geometry_score", None)),
            order_proximity=_safe_float(getattr(sim, "order_proximity", None)),
            pair_rank=_safe_float(getattr(sim, "pair_rank", None)),
            certification_status=getattr(sim, "certification_status", None),
            rejection_reason=getattr(sim, "rejection_reason", None),
            direction_dx=_safe_float(sim.direction_dx),
            direction_dy=_safe_float(sim.direction_dy),
            is_connected=bool(sim.is_connected) if sim.is_connected is not None else None,
        )

    left_key = _s3_key_from_uri(left_photo.s3_uri)
    right_key = _s3_key_from_uri(right_photo.s3_uri)
    strict_gate = _build_pair_debug_strict_gate(
        num_matches=num_matches,
        num_inliers=num_inliers,
        geometric_score=score,
        diagnostics=diagnostics if isinstance(diagnostics, dict) else None,
    )
    failed_checks = [
        reason
        for reason, passed in (strict_gate.get("checks") or {}).items()
        if not bool(passed)
    ]
    live_rejection_reason = failed_checks[0] if failed_checks else None
    live_status = "reject" if live_rejection_reason else stored_metrics.certification_status if stored_metrics and stored_metrics.certification_status else "usable"
    live_graph_edge_score = _safe_float(diagnostics.get("graph_edge_score"))
    if live_graph_edge_score is None:
        native_scores = diagnostics.get("native_matching_scores")
        if isinstance(native_scores, dict):
            live_graph_edge_score = _safe_float(native_scores.get("graph_edge_score"))
    live_retrieval_score = _safe_float(diagnostics.get("retrieval_score"))
    if live_retrieval_score is None and stored_metrics is not None:
        live_retrieval_score = stored_metrics.retrieval_score
    live_parallax_score = _safe_float(diagnostics.get("parallax_score"))
    if live_parallax_score is None and stored_metrics is not None:
        live_parallax_score = stored_metrics.parallax_score

    live_metrics = PairDebugLiveMetrics(
        matcher=str(diagnostics.get("matcher")) if diagnostics.get("matcher") is not None else None,
        checkpoint=str(diagnostics.get("checkpoint")) if diagnostics.get("checkpoint") is not None else None,
        match_engine=MAST3R_ENGINE_NAME,
        retrieval_score=live_retrieval_score,
        reciprocal_match_count=_safe_int(diagnostics.get("reciprocal_match_count")) or int(num_matches),
        pointmap_consistency=_safe_float(diagnostics.get("pointmap_consistency")),
        alignment_residual=_safe_float(diagnostics.get("alignment_residual")),
        reprojection_error=_safe_float(diagnostics.get("reprojection_error")),
        parallax_score=live_parallax_score,
        graph_component_id=_safe_int(getattr(sim, "graph_component_id", None)),
        graph_edge_score=live_graph_edge_score,
        confidence_threshold=_safe_float(diagnostics.get("confidence_threshold")),
        geometry_model=str(diagnostics.get("geometry_model")) if diagnostics.get("geometry_model") is not None else None,
        raw_correspondence_count=_safe_int(diagnostics.get("raw_correspondence_count")),
        raw_matches=[_pair_debug_point_from_dict(p) for p in sampled_raw_points if isinstance(p, dict)],
        threshold_trials=[t for t in (diagnostics.get("threshold_trials") or []) if isinstance(t, dict)],
        num_matches=int(num_matches),
        threshold_match_count=_safe_int(diagnostics.get("threshold_match_count")) or int(num_matches),
        active_match_count=_safe_int(diagnostics.get("active_match_count")) or int(num_matches),
        num_inliers=int(num_inliers),
        geometric_score=float(score),
        pair_rank=float(live_graph_edge_score if live_graph_edge_score is not None else score),
        certification_status=live_status,
        rejection_reason=live_rejection_reason,
        overlap_ratio=_safe_float(diagnostics.get("overlap_ratio")),
        combined_geometry_score=_safe_float(diagnostics.get("combined_geometry_score")),
        order_proximity=_safe_float(getattr(sim, "order_proximity", None)),
        direction_dx=_safe_float(direction[0]) if direction else None,
        direction_dy=_safe_float(direction[1]) if direction else None,
        segment_scores={},
        score_components={
            "pointmap_consistency": float(diagnostics.get("pointmap_consistency") or 0.0),
            "overlap_ratio": float(diagnostics.get("overlap_ratio") or 0.0),
            "graph_edge_score": float(live_graph_edge_score or 0.0),
        },
        timing={},
        oracle=(diagnostics.get("oracle") or {}) if isinstance(diagnostics.get("oracle"), dict) else {},
        native_matching_scores={
            str(k): float(v) for k, v in (diagnostics.get("native_matching_scores") or {}).items() if v is not None
        },
        native_matching_scores_raw={},
        strict_gate=strict_gate,
        inlier_match_count=len(inlier_points_list),
        inlier_matches=[_pair_debug_point_from_dict(p) for p in sampled_inlier_points if isinstance(p, dict)],
    )
    response_build_seconds = time.perf_counter() - postprocess_started_at
    request_total_seconds = time.perf_counter() - request_started_at
    live_metrics.timing = {
        "endpoint_total_ms": float(request_total_seconds * 1000.0),
        "endpoint_lookup_ms": float(lookup_seconds * 1000.0),
        "endpoint_s3_total_ms": float(s3_total_seconds * 1000.0),
        "endpoint_s3_left_ms": float(left_s3_seconds * 1000.0),
        "endpoint_s3_right_ms": float(right_s3_seconds * 1000.0),
        "endpoint_matcher_ms": float(matcher_seconds * 1000.0),
        "endpoint_response_build_ms": float(response_build_seconds * 1000.0),
        "model_pair_ms": float(pair_model_seconds * 1000.0),
        "model_mast3r_inference_ms": float(mast3r_inference_seconds * 1000.0),
        "model_cache_hit": bool(timing_diagnostics.get("model_cache_hit", False)),
        "model_device": str(timing_diagnostics.get("model_device", "cuda")),
        "cuda_available": bool(timing_diagnostics.get("cuda_available", True)),
        "preferred_device": str(timing_diagnostics.get("preferred_device", "cuda")),
        "endpoint_unaccounted_ms": float(
            max(
                0.0,
                request_total_seconds * 1000.0
                - (
                    lookup_seconds * 1000.0
                    + s3_total_seconds * 1000.0
                    + matcher_seconds * 1000.0
                    + response_build_seconds * 1000.0
                ),
            )
        ),
    }
    logger.info(
        "pair_debug_timing project=%s job=%s pair=%s<->%s total_ms=%.1f lookup_ms=%.1f s3_total_ms=%.1f s3_left_ms=%.1f s3_right_ms=%.1f matcher_ms=%.1f response_build_ms=%.1f mast3r_pair_ms=%.1f mast3r_inference_ms=%.1f model_cache_hit=%s model_device=%s cuda_available=%s preferred_device=%s",
        project_id,
        job.id,
        payload.left_photo_id,
        payload.right_photo_id,
        request_total_seconds * 1000.0,
        lookup_seconds * 1000.0,
        s3_total_seconds * 1000.0,
        left_s3_seconds * 1000.0,
        right_s3_seconds * 1000.0,
        matcher_seconds * 1000.0,
        response_build_seconds * 1000.0,
        pair_model_seconds * 1000.0,
        mast3r_inference_seconds * 1000.0,
        str(bool(timing_diagnostics.get("model_cache_hit", False))),
        str(timing_diagnostics.get("model_device", "cuda")),
        str(timing_diagnostics.get("cuda_available", True)),
        str(timing_diagnostics.get("preferred_device", "cuda")),
    )
    slow_threshold_s = 2.0
    if request_total_seconds >= slow_threshold_s:
        stage_breakdown = {
            "lookup": lookup_seconds,
            "s3_total": s3_total_seconds,
            "matcher_total": matcher_seconds,
            "response_build": response_build_seconds,
        }
        dominant_stage = max(stage_breakdown, key=stage_breakdown.get)
        logger.warning(
            "pair_debug_slow project=%s job=%s pair=%s<->%s total_ms=%.1f dominant_stage=%s dominant_ms=%.1f threshold_ms=%.1f",
            project_id,
            job.id,
            payload.left_photo_id,
            payload.right_photo_id,
            request_total_seconds * 1000.0,
            dominant_stage,
            stage_breakdown[dominant_stage] * 1000.0,
            slow_threshold_s * 1000.0,
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
    - MASt3R retrieval and graph edge scores
    - pointmap/geometry diagnostics
    - whether extra same-component photos were kept as debug suggestions
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
    similarities = db.query(PhotoSimilarity).filter(PhotoSimilarity.job_id == job.id).all()

    transition_sequences = (
        db.query(TransitionSequence)
        .filter(TransitionSequence.job_id == job.id)
        .all()
    )
    pose_rows = (
        db.query(PhotoPoseAlignment)
        .filter(PhotoPoseAlignment.job_id == job.id)
        .all()
    )
    component_by_photo = {
        int(row.photo_id): int(row.graph_component_id)
        for row in pose_rows
        if row.graph_component_id is not None
    }
    photos_by_component: dict[int, list[int]] = {}
    for photo_id, component_id in component_by_photo.items():
        photos_by_component.setdefault(int(component_id), []).append(int(photo_id))
    for component_id in list(photos_by_component.keys()):
        photos_by_component[component_id].sort()
    sequences_by_cluster: dict[int, list[TransitionSequenceInfo]] = {}
    for sequence in transition_sequences:
        serialized = _serialize_transition_sequence(sequence)
        for cluster_id in serialized.source_cluster_ids:
            sequences_by_cluster.setdefault(int(cluster_id), []).append(serialized)

    # Build similarity lookup: (photo_a, photo_b) -> similarity record
    sim_lookup = {}
    for sim in similarities:
        key = (min(sim.photo_a_id, sim.photo_b_id), max(sim.photo_a_id, sim.photo_b_id))
        sim_lookup[key] = sim

    # Get clusters info
    clusters = db.query(RoomCluster).filter(RoomCluster.job_id == job.id).all()
    cluster_map = {c.id: c for c in clusters}
    paired_cluster_for_photo: dict[int, int] = {}
    for clip in clips:
        clip_photo_ids = [int(photo_id) for photo_id in (clip.source_photo_ids or [])]
        if len(clip_photo_ids) != 2:
            continue
        cluster_key = int(clip.room_cluster_id or clip.id)
        for photo_id in clip_photo_ids:
            paired_cluster_for_photo[int(photo_id)] = cluster_key

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
                        photo_a_filename=photo_filename_map.get(sim.photo_a_id),
                        photo_b_filename=photo_filename_map.get(sim.photo_b_id),
                        pair_source=sim.pair_source,
                        match_engine=getattr(sim, "match_engine", None),
                        retrieval_score=_safe_float(getattr(sim, "retrieval_score", None)),
                        reciprocal_match_count=_safe_int(getattr(sim, "reciprocal_match_count", None)),
                        pointmap_consistency=_safe_float(getattr(sim, "pointmap_consistency", None)),
                        alignment_residual=_safe_float(getattr(sim, "alignment_residual", None)),
                        reprojection_error=_safe_float(getattr(sim, "reprojection_error", None)),
                        parallax_score=_safe_float(getattr(sim, "parallax_score", None)),
                        graph_component_id=_safe_int(getattr(sim, "graph_component_id", None)),
                        graph_edge_score=_safe_float(getattr(sim, "graph_edge_score", None)),
                        overlap_ratio=_safe_float(getattr(sim, "overlap_ratio", None)),
                        combined_geometry_score=_safe_float(getattr(sim, "combined_geometry_score", None)),
                        order_proximity=_safe_float(getattr(sim, "order_proximity", None)),
                        pair_rank=_safe_float(getattr(sim, "pair_rank", None)),
                        certification_status=getattr(sim, "certification_status", None),
                        rejection_reason=getattr(sim, "rejection_reason", None),
                        direction_dx=_safe_float(sim.direction_dx),
                        direction_dy=_safe_float(sim.direction_dy),
                        is_connected=bool(sim.is_connected),
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

        avg_pair_rank = None
        if all_similarities:
            rank_scores = [s.pair_rank for s in all_similarities if getattr(s, "pair_rank", None) is not None]
            if rank_scores:
                avg_pair_rank = sum(rank_scores) / len(rank_scores)

        cluster_key = int(clip.room_cluster_id or clip.id)
        suggestions = _cluster_suggestions(
            cluster_photo_ids=[int(photo_id) for photo_id in photo_ids],
            cluster_id=cluster_key,
            component_by_photo=component_by_photo,
            photos_by_component=photos_by_component,
            sim_lookup=sim_lookup,
            photo_map=photo_map,
            paired_cluster_for_photo=paired_cluster_for_photo,
        )

        cluster_responses.append(ClusterDebugResponse(
            cluster_id=cluster_key,
            room_type=cluster.room_type if cluster else None,
            photo_ids=photo_ids,
            photo_filenames=[photo_filename_map.get(photo_id, f"photo-{photo_id}") for photo_id in photo_ids],
            photos=photos_debug,
            total_photos=len(photo_ids),
            sequence_order=cluster.sequence_order if cluster else None,
            avg_pair_rank=avg_pair_rank,
            has_direction_info=any(getattr(s, "reciprocal_match_count", None) and s.reciprocal_match_count >= 3 for s in all_similarities),
            sequences=sequences_by_cluster.get(cluster_key, []),
            suggestions=suggestions,
        ))

    return ClusterListDebugResponse(
        project_id=project_id,
        job_id=job.id,
        clusters=cluster_responses,
        total_clusters=len(cluster_responses),
    )
