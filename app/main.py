"""FastAPI application for Picaivid Media Service."""
from datetime import datetime
import uuid
import logging
from secrets import compare_digest
from typing import Dict
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, HTTPException, Request
from sqlalchemy import text
from sqlalchemy.orm import Session
from starlette.responses import JSONResponse

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.models import (
    AnalysisResult,
    Clip,
    Job,
    JobPhoto,
    PhotoRelation,
    PhotoSceneGeometry,
    RoomCluster,
    SceneComponent,
    SceneComponentMembership,
)
from app.db.models.scene_truth import TRUTH_SPLITS, TRUTH_STATUSES, SceneTruthSet
from app.db.session import get_db
from app.models.warmup import warmup_core_models
from app.pipeline.orchestrator import PipelineOrchestrator
from app.pipeline.progress import STALE_AFTER_SECONDS
from app.schemas.job_progress import (
    CancelResponse,
    JobProgressResponse,
    ProjectPurgeResponse,
)
from app.db.models.timeline import (
    SOURCE_SECONDS as TL_SOURCE_SECONDS,
    source_seconds_for,
)
from app.db.models.timeline import Timeline, TimelineClip
from app.pipeline.phase1_analyze.camera_move import (
    axes_of,
    compose_label,
    reverse_label,
    reverse_vector,
    vector_from_label,
)
from app.pipeline.phase1_analyze.directorial_prompt import (
    build_shot_prompt,
    effective_move_label,
)
from app.pipeline.phase1_analyze.pairing import persisted_pair_score
from app.pipeline.phase3_timeline.music import MusicAnalysisError, analyze_music_uri
from app.pipeline.phase3_timeline.materialize import (
    latest_timeline,
    materialize_timeline,
    populate_timeline as populate_timeline_rows,
)
from app.schemas.timeline import (
    CandidateListResponse,
    CandidatePhoto,
    CandidateVideo,
    TimelinePhoto,
    TimelineMusic,
    TimelineMusicRequest,
    TimelineListResponse,
    TimelineSummary,
    TimelineResponse,
    TimelineShot,
    TimelineUpdateRequest,
)
from app.schemas.clip import (
    AnalysisInfo,
    ClipListResponse,
    ClipResponse,
    ClipTransitionStep,
    ClusterInfo,
    MotionDecisionDebug,
    PhotoGeometryDebug,
    PhotoRelationDebug,
    PhotoRelationDebugRequest,
    PhotoRelationDebugResponse,
    SceneComponentSummary,
    SceneDebugResponse,
    ShotPlanResponse,
    SourcePhotoInfo,
)
from app.schemas.job import JobMessage, JobStatusResponse
from app.schemas.scene_truth import (
    RoomInstancePayload,
    SceneTruthSetPayload,
    SceneTruthSetResponse,
    TruthPhoto,
    TruthSetListResponse,
    TruthSetSummary,
)
from app.services.storage.s3_client import s3_client

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Picaivid Media Service",
    description="Phased video pipeline for real estate media",
    version="0.2.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
)


def _internal_service_key() -> str:
    return (settings.MEDIA_SERVICE_API_KEY or settings.API_KEY or "").strip()


def _valid_internal_service_key(supplied: str) -> bool:
    expected = _internal_service_key()
    return bool(
        expected
        and supplied
        and len(supplied) == len(expected)
        and compare_digest(supplied, expected)
    )


@app.middleware("http")
async def require_rails_service_key(request: Request, call_next):
    """Keep the media HTTP surface private to Rails.

    The worker itself consumes SQS and has no HTTP listener. This middleware
    protects the optional media API used by Rails for timeline reads, edits and
    diagnostics; a browser request cannot bypass Rails authentication.
    """
    if request.url.path.startswith(("/api/", "/internal/")):
        supplied = request.headers.get("X-Internal-Service-Key", "").strip()
        if not _internal_service_key():
            logger.error("MEDIA_SERVICE_API_KEY is not configured")
            return JSONResponse(
                {"detail": "Media service internal authentication is not configured."},
                status_code=503,
            )
        if not _valid_internal_service_key(supplied):
            return JSONResponse(
                {"detail": "Media service accepts requests from Rails only."},
                status_code=401,
            )
    return await call_next(request)


@app.on_event("startup")
async def startup_warm_models() -> None:
    # Inference belongs to the worker. Keeping the API lightweight avoids loading
    # a second multi-gigabyte model beside the local MPS worker.
    warmup_core_models(context="api", include_vggt=False, include_legacy=False)


def _safe_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _timeline_generation_prompt(
    clip: TimelineClip,
    index: int,
    total: int,
    *,
    room_has_pair: bool = False,
) -> str:
    """Return the prompt shown in Studio for one stored timeline shot."""
    if clip.clip is not None and clip.clip.prompt_used:
        return clip.clip.prompt_used
    return build_shot_prompt(
        clip,
        is_first=index == 0,
        is_last=index == total - 1,
        room_has_pair=room_has_pair,
    )


def _safe_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _s3_key_from_uri(s3_uri: str | None) -> str | None:
    if not s3_uri:
        return None
    if s3_uri.startswith("s3://"):
        parts = s3_uri[5:].split("/", 1)
        return parts[1] if len(parts) == 2 else None
    return s3_uri


def _signed_url(s3_uri: str | None, expires_in: int = 3600) -> str | None:
    """Presign an object. Generation submits need a longer window than the UI:
    a Runware job can sit in a queue for minutes and then fetch the image."""
    key = _s3_key_from_uri(s3_uri)
    return s3_client.generate_presigned_url(key, expires_in=expires_in) if key else None


def _music_preview_url(uri: str | None) -> str | None:
    """Return a browser-readable, non-storage URI for the local editor."""
    if not uri:
        return None
    if uri.startswith('s3://'):
        return _signed_url(uri)
    if uri.startswith(('https://', 'http://')):
        return uri
    return None


def _photo_geometry_map(db: Session, job_id: int) -> dict[int, PhotoSceneGeometry]:
    return {
        int(row.photo_id): row
        for row in db.query(PhotoSceneGeometry).filter(PhotoSceneGeometry.job_id == job_id).all()
    }


def _membership_map(db: Session, job_id: int) -> dict[int, SceneComponentMembership]:
    return {
        int(row.photo_id): row
        for row in db.query(SceneComponentMembership).filter(SceneComponentMembership.job_id == job_id).all()
    }


def _serialize_photo_debug(
    photo: JobPhoto,
    geometry: PhotoSceneGeometry | None,
    membership: SceneComponentMembership | None,
) -> PhotoGeometryDebug:
    return PhotoGeometryDebug(
        photo_id=int(photo.id),
        rails_photo_id=str(photo.rails_photo_id),
        filename=photo.filename,
        thumbnail_url=_signed_url(photo.s3_uri),
        room_label=photo.room_override or photo.room_label,
        scene_component_id=_safe_int(getattr(membership, "scene_component_id", None)),
        order_index=_safe_int(getattr(membership, "order_index", None)),
        photo_role=getattr(membership, "photo_role", None),
        pose_confidence=_safe_float(getattr(geometry, "pose_confidence", None)),
        depth_confidence=_safe_float(getattr(geometry, "depth_confidence", None)),
        point_confidence=_safe_float(getattr(geometry, "point_confidence", None)),
        visibility_score=_safe_float(getattr(geometry, "visibility_score", None)),
        reprojection_error=_safe_float(getattr(geometry, "reprojection_error", None)),
        camera_center=getattr(geometry, "camera_center", None),
        view_direction=getattr(geometry, "view_direction", None),
        depth_artifact_uri=getattr(geometry, "depth_artifact_uri", None),
        point_map_artifact_uri=getattr(geometry, "point_map_artifact_uri", None),
    )


def _serialize_relation_debug(
    relation: PhotoRelation,
    photo_map: dict[int, JobPhoto],
) -> PhotoRelationDebug:
    return PhotoRelationDebug(
        photo_a_id=int(relation.photo_a_id),
        photo_b_id=int(relation.photo_b_id),
        photo_a_filename=photo_map.get(int(relation.photo_a_id)).filename if photo_map.get(int(relation.photo_a_id)) else None,
        photo_b_filename=photo_map.get(int(relation.photo_b_id)).filename if photo_map.get(int(relation.photo_b_id)) else None,
        scene_component_id=_safe_int(relation.scene_component_id),
        same_component=relation.scene_component_id is not None,
        overlap_score=_safe_float(relation.overlap_score),
        track_support=_safe_float(relation.track_support),
        reprojection_score=_safe_float(relation.reprojection_score),
        relation_confidence=_safe_float(relation.relation_confidence),
        baseline_distance=_safe_float(relation.baseline_distance),
        direction_dx=_safe_float(relation.direction_dx),
        direction_dy=_safe_float(relation.direction_dy),
        continuity_type=relation.continuity_type,
        is_bridge_edge=bool(relation.is_bridge_edge),
        is_connected=bool(relation.is_connected),
        relative_transform=relation.relative_transform,
        debug_metrics=relation.debug_metrics or {},
    )


def _shot_plan_response(project_id: str, job: Job | None, db: Session) -> ShotPlanResponse:
    if job is None:
        return ShotPlanResponse(project_id=project_id, job_id=None)
    analyses = (
        db.query(AnalysisResult)
        .filter(AnalysisResult.job_id == job.id)
        .order_by(AnalysisResult.room_cluster_id.asc().nulls_last())
        .all()
    )
    for analysis in analyses:
        plan = (analysis.debug_metrics or {}).get("shot_plan") if isinstance(analysis.debug_metrics, dict) else None
        if isinstance(plan, dict):
            return ShotPlanResponse(
                project_id=project_id,
                job_id=int(job.id),
                planner_version=str(plan.get("planner_version", "v2.0")),
                runtime_provenance=plan.get("runtime_provenance") or {},
                target_length_seconds=_safe_float(plan.get("target_length_seconds")),
                target_group_budget=plan.get("target_group_budget"),
                sequence_edges=plan.get("sequence_edges") or [],
                ordered_shots=plan.get("ordered_shots") or [],
            )
    return ShotPlanResponse(project_id=project_id, job_id=int(job.id))


def _camera_direction_recommendation(dx: float | None, dy: float | None, confidence: float | None) -> str:
    if confidence is None or confidence < 0.45 or dx is None or dy is None:
        return "Keep movement subtle"
    abs_x = abs(dx)
    abs_y = abs(dy)
    if abs_x < 0.15 and abs_y < 0.15:
        return "Hold / micro move"
    if abs_x >= abs_y:
        return "Move camera left" if dx > 0 else "Move camera right"
    return "Move camera up" if dy > 0 else "Move camera down"


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "picaivid-media-service",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "environment": settings.ENVIRONMENT,
        "analysis_engine": settings.ANALYSIS_MATCH_ENGINE,
    }


@app.get("/")
async def root():
    return {"message": "Picaivid Media Service", "version": "0.2.0"}


@app.post("/internal/jobs", response_model=JobStatusResponse)
async def create_job(message: JobMessage, db: Session = Depends(get_db)):
    orchestrator = PipelineOrchestrator(db)
    job = orchestrator.create_job_from_message(message)
    if settings.ENVIRONMENT == "development":
        orchestrator.execute(job.id, allowed_phases=[1, 2])
        db.refresh(job)
    return JobStatusResponse(
        job_id=job.id,
        project_id=job.project_id,
        status=job.status,
        current_phase=job.current_phase,
        error_message=job.error_message,
    )


@app.get("/internal/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: int, db: Session = Depends(get_db)):
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
async def run_phase(job_id: int, phase: int, db: Session = Depends(get_db)):
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
async def get_project_clips(project_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.project_id == project_id).order_by(Job.created_at.desc()).first()
    if not job:
        return ClipListResponse(project_id=project_id, job_id=None, job_status=None, clips=[], total_clips=0)

    clips = (
        db.query(Clip)
        .outerjoin(RoomCluster, Clip.room_cluster_id == RoomCluster.id)
        .filter(Clip.job_id == job.id)
        .order_by(RoomCluster.sequence_order.asc().nulls_last(), Clip.id.asc())
        .all()
    )
    job_photos = db.query(JobPhoto).filter(JobPhoto.job_id == job.id).all()
    photo_map = {int(photo.id): photo for photo in job_photos}
    geometry_map = _photo_geometry_map(db, int(job.id))
    membership_map = _membership_map(db, int(job.id))
    cluster_map = {cluster.id: cluster for cluster in db.query(RoomCluster).filter(RoomCluster.job_id == job.id).all()}
    analysis_map = {
        analysis.room_cluster_id: analysis
        for analysis in db.query(AnalysisResult).filter(AnalysisResult.job_id == job.id).all()
        if analysis.room_cluster_id is not None
    }
    relation_lookup: Dict[tuple[int, int], PhotoRelation] = {
        (min(int(relation.photo_a_id), int(relation.photo_b_id)), max(int(relation.photo_a_id), int(relation.photo_b_id))): relation
        for relation in db.query(PhotoRelation).filter(PhotoRelation.job_id == job.id).all()
    }

    responses: list[ClipResponse] = []
    for clip in clips:
        source_photos = []
        for photo_id in clip.source_photo_ids or []:
            photo = photo_map.get(int(photo_id))
            if photo is None:
                continue
            geometry = geometry_map.get(int(photo.id))
            membership = membership_map.get(int(photo.id))
            source_photos.append(
                SourcePhotoInfo(
                    id=int(photo.id),
                    rails_photo_id=str(photo.rails_photo_id),
                    filename=photo.filename,
                    s3_uri=photo.s3_uri,
                    thumbnail_url=_signed_url(photo.s3_uri),
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
                    pose_confidence=_safe_float(getattr(geometry, "pose_confidence", None)),
                    photo_role=getattr(membership, "photo_role", None),
                )
            )

        transition_steps: list[ClipTransitionStep] = []
        for index in range(max(0, len(clip.source_photo_ids or []) - 1)):
            from_photo_id = int(clip.source_photo_ids[index])
            to_photo_id = int(clip.source_photo_ids[index + 1])
            relation = relation_lookup.get((min(from_photo_id, to_photo_id), max(from_photo_id, to_photo_id)))
            if relation is None:
                continue
            dx = _safe_float(relation.direction_dx)
            dy = _safe_float(relation.direction_dy)
            if from_photo_id != int(relation.photo_a_id) and dx is not None and dy is not None:
                dx = -dx
                dy = -dy
            transition_steps.append(
                ClipTransitionStep(
                    from_photo_id=from_photo_id,
                    to_photo_id=to_photo_id,
                    from_filename=photo_map.get(from_photo_id).filename if photo_map.get(from_photo_id) else None,
                    to_filename=photo_map.get(to_photo_id).filename if photo_map.get(to_photo_id) else None,
                    relation_confidence=_safe_float(relation.relation_confidence),
                    overlap_score=_safe_float(relation.overlap_score),
                    track_support=_safe_float(relation.track_support),
                    reprojection_score=_safe_float(relation.reprojection_score),
                    direction_dx=dx,
                    direction_dy=dy,
                    recommendation=_camera_direction_recommendation(dx, dy, _safe_float(relation.relation_confidence)),
                    continuity_type=relation.continuity_type,
                    is_connected=bool(relation.is_connected),
                    is_bridge_edge=bool(relation.is_bridge_edge),
                )
            )

        cluster = cluster_map.get(clip.room_cluster_id) if clip.room_cluster_id else None
        analysis = analysis_map.get(clip.room_cluster_id) if clip.room_cluster_id else None
        cluster_info = (
            ClusterInfo(
                id=int(cluster.id),
                room_type=cluster.room_type,
                confidence_tier=cluster.confidence_tier,
                image_count=int(cluster.image_count or 0),
                overlap_score=_safe_float(cluster.overlap_score),
                depth_variance=_safe_float(cluster.depth_variance),
                recommended_motion=cluster.recommended_motion,
                sequence_order=_safe_int(cluster.sequence_order),
                geometry_confidence=_safe_float(cluster.geometry_confidence),
                scene_component_id=_safe_int(cluster.scene_component_id),
            )
            if cluster is not None
            else None
        )
        analysis_info = (
            AnalysisInfo(
                tier=analysis.tier,
                recommended_motion=analysis.recommended_motion,
                model_recommendation=analysis.model_recommendation,
                cfg_scale=analysis.cfg_scale,
                inference_steps=analysis.inference_steps,
                debug_metrics=analysis.debug_metrics,
            )
            if analysis is not None
            else None
        )
        responses.append(
            ClipResponse(
                id=int(clip.id),
                job_id=int(clip.job_id),
                room_cluster_id=_safe_int(clip.room_cluster_id),
                source_photo_ids=clip.source_photo_ids,
                motion_type=clip.motion_type,
                model_used=clip.model_used,
                is_3d=bool(clip.is_3d),
                duration=clip.duration,
                prompt_used=clip.prompt_used,
                s3_uri=clip.s3_uri,
                video_url=_signed_url(clip.s3_uri),
                status=clip.status,
                source_photos=source_photos or None,
                transition_steps=transition_steps or None,
                cluster_info=cluster_info,
                analysis_info=analysis_info,
            )
        )

    return ClipListResponse(
        project_id=project_id,
        job_id=int(job.id),
        job_status=job.status,
        clips=responses,
        total_clips=len(responses),
    )


def _job_progress_payload(project_id: str, job) -> JobProgressResponse:
    """Derive the UI-facing state for a job.

    `stalled` is the important one: a job whose status still says it is working
    but whose heartbeat has gone quiet means the worker died. Deriving it here
    keeps every consumer honest about the difference between slow and dead.
    """
    if job is None:
        return JobProgressResponse(project_id=project_id, state="idle")

    now = datetime.utcnow()
    status = str(job.status or "")
    heartbeat_age = (
        (now - job.heartbeat_at).total_seconds() if job.heartbeat_at else None
    )
    elapsed = (now - job.created_at).total_seconds() if job.created_at else None

    if job.canceled_at is not None or status == "cancelled":
        state = "cancelled"
    elif status == "failed":
        state = "failed"
    elif status in {"pending", ""}:
        state = "queued"
    elif status.endswith("_complete") or status == "complete":
        state = "complete"
    elif heartbeat_age is not None and heartbeat_age > STALE_AFTER_SECONDS:
        state = "stalled"
    elif heartbeat_age is None and elapsed is not None and elapsed > STALE_AFTER_SECONDS:
        # Claimed by a worker that never reported anything at all -- the exact
        # signature of a worker that cannot run the requested phase.
        state = "stalled"
    else:
        state = "running"

    current, total = job.progress_current, job.progress_total
    percent = None
    if total:
        percent = round(min(100.0, max(0.0, (current or 0) * 100.0 / total)), 1)
    if state == "complete":
        percent = 100.0

    return JobProgressResponse(
        project_id=project_id,
        job_id=int(job.id),
        state=state,
        status=status or None,
        phase=job.current_phase,
        phase_label=job.progress_label,
        current=current,
        total=total,
        percent=percent,
        elapsed_seconds=round(elapsed, 1) if elapsed is not None else None,
        seconds_since_heartbeat=(
            round(heartbeat_age, 1) if heartbeat_age is not None else None
        ),
        stale_after_seconds=STALE_AFTER_SECONDS,
        cancel_requested=bool(job.cancel_requested),
        error_message=job.error_message,
        started_at=job.created_at,
    )


@app.get("/api/projects/{project_id}/shot-line")
async def get_project_shot_line(
    project_id: str,
    seconds: float = 45.0,
    bpm: float = 100.0,
    db: Session = Depends(get_db),
):
    """The ordered shot line: what the film would be, before anything is rendered.

    Exists so the story order can be reviewed before paying for video
    generation, which is by far the most expensive step.
    """
    from scripts.build_shot_line import load_rooms
    from app.pipeline.phase1_analyze.shot_line import (
        build_shot_line,
        schedule_resolved_close_ups,
    )
    from app.pipeline.phase1_analyze.detail_crop import (
        is_detail_eligible,
        resolve_persisted_detail_candidates,
    )

    try:
        rooms, source = load_rooms(db, project_id)
    except SystemExit:
        return {"project_id": project_id, "shots": [], "rooms": 0,
                "source": "no analysis yet"}

    # Build selected wides first. Detail detections were saved during Phase 1;
    # the review endpoint only selects that persisted evidence and never loads a
    # second model or downloads original source images.
    line = build_shot_line(
        rooms, target_seconds=seconds, bpm=bpm, include_close_ups=False)
    if line.shots:
        room_photo_ids = {
            room.room_id: room.photo_ids
            for room in rooms
            if is_detail_eligible(room.room_type)
        }
        selected_room_ids = {shot.room_id for shot in line.shots}
        candidate_photo_ids = {
            photo_id
            for room_id, photo_ids in room_photo_ids.items()
            if room_id in selected_room_ids
            for photo_id in photo_ids
        }
        geometry_rows = db.query(PhotoSceneGeometry).filter(
            PhotoSceneGeometry.photo_id.in_(candidate_photo_ids)
        ).all()
        resolved = resolve_persisted_detail_candidates(
            line.shots,
            room_photo_ids,
            {int(row.photo_id): row.local_metrics or {} for row in geometry_rows},
        )
        line.shots = schedule_resolved_close_ups(line.shots, resolved)

    photo_rows = db.query(JobPhoto).filter(
        JobPhoto.id.in_([p for shot in line.shots for p in shot.photo_ids])
    ).all() if line.shots else []
    photo_map = {int(p.id): p for p in photo_rows}

    def photo_payload(pid: int):
        photo = photo_map.get(int(pid))
        if photo is None:
            return {"id": pid, "thumbnail_url": None, "position": None}
        return {
            "id": int(photo.id),
            "thumbnail_url": _signed_url(photo.s3_uri),
            "position": photo.position,
            "room_label": photo.room_label,
        }

    return {
        "project_id": project_id,
        "source": source,
        "rooms": len(rooms),
        "bpm": line.bpm,
        "shot_seconds": line.shots[0].duration_seconds if line.shots else 0,
        "generation_seconds": (
            line.shots[0].generation_duration_seconds if line.shots else 0
        ),
        "duration": round(line.duration, 2),
        "hero_room_id": line.hero_room_id,
        "keyframe_fraction": round(line.keyframe_fraction(), 3),
        "dropped_rooms": line.dropped_rooms,
        "shots": [{
            "index": s.index,
            "act": s.act,
            "room_id": s.room_id,
            "room_type": s.room_type,
            "kind": s.kind,
            "is_hero": s.is_hero,
            "start_seconds": s.start_seconds,
            "duration_seconds": s.duration_seconds,
            "generation_duration_seconds": s.generation_duration_seconds,
            "is_close_up": s.is_close_up,
            "crop_box": list(s.crop_box) if s.crop_box else None,
            "crop_score": round(s.crop_score, 4) if s.crop_score is not None else None,
            "source_size": list(s.source_size) if s.source_size else None,
            "detail_subject": s.detail_subject,
            "detector_score": round(s.detector_score, 4) if s.detector_score is not None else None,
            "subject_box": list(s.subject_box) if s.subject_box else None,
            "detail_framing": s.detail_framing,
            # Fractions per second, so the UI can draw the motion rather than
            # only name it.
            "move": s.move.label,
            "move_vector": {
                "zoom": round(s.move.zoom, 4),
                "pan": round(s.move.pan, 4),
                "tilt": round(s.move.tilt, 4),
                "roll_deg": round(s.move.roll_deg, 3),
            },
            "photos": [photo_payload(p) for p in s.photo_ids],
        } for s in line.shots],
    }


@app.get("/api/projects/{project_id}/progress", response_model=JobProgressResponse)
async def get_project_progress(project_id: str, db: Session = Depends(get_db)):
    """Live progress for the newest job on a project. Safe to poll."""
    job = (
        db.query(Job)
        .filter(Job.project_id == project_id)
        .order_by(Job.id.desc())
        .first()
    )
    return _job_progress_payload(project_id, job)


@app.post("/api/projects/{project_id}/cancel", response_model=CancelResponse)
async def cancel_project_job(project_id: str, db: Session = Depends(get_db)):
    """Request cancellation of the newest job.

    Cooperative: this only raises the flag. The worker notices it at a loop
    boundary and unwinds cleanly, so a cancel never leaves a half-written scene
    graph behind. A job that is already finished is left alone.
    """
    job = (
        db.query(Job)
        .filter(Job.project_id == project_id)
        .order_by(Job.id.desc())
        .first()
    )
    if job is None:
        return CancelResponse(
            project_id=project_id, job_id=None, cancelled=False,
            detail="No job to cancel.",
        )

    status = str(job.status or "")
    if status.endswith("_complete") or status in {"complete", "failed", "cancelled"}:
        return CancelResponse(
            project_id=project_id, job_id=int(job.id), cancelled=False,
            detail=f"Job already finished ({status}).",
        )

    job.cancel_requested = True
    db.commit()
    logger.info("JOB_CANCEL_REQUESTED job_id=%s project_id=%s", job.id, project_id)
    return CancelResponse(
        project_id=project_id, job_id=int(job.id), cancelled=True,
        detail="Cancellation requested; the worker will stop at the next checkpoint.",
    )


@app.delete("/api/projects/{project_id}", response_model=ProjectPurgeResponse)
async def purge_project(project_id: str, db: Session = Depends(get_db)):
    """Remove every media-service row belonging to a project.

    Rails owns `projects` and cannot cascade into this service's tables:
    `projects.id` is a uuid while `jobs.project_id` is a varchar, so no foreign
    key can exist between them. Deleting a project in Rails therefore left its
    jobs, photos, relations and whole scene graph behind -- eight projects had
    accumulated sixteen orphaned jobs, a third of the table, before this
    endpoint existed.

    Every job-scoped table already cascades from `jobs`, so removing the job
    rows is enough for them. `scene_truth_sets` is project-scoped with no job
    reference and is deleted directly.

    Idempotent by design: purging an unknown or already-purged project reports
    zeros rather than failing, so a retried delete is always safe.
    """
    job_ids = [int(row[0]) for row in
               db.query(Job.id).filter(Job.project_id == project_id).all()]

    counts: dict[str, int] = {}
    if job_ids:
        for field, model in (
            ("job_photos", JobPhoto),
            ("photo_relations", PhotoRelation),
            ("scene_components", SceneComponent),
            ("photo_scene_geometry", PhotoSceneGeometry),
            ("analysis_results", AnalysisResult),
            ("room_clusters", RoomCluster),
        ):
            counts[field] = (
                db.query(model).filter(model.job_id.in_(job_ids)).count()
            )

    truth = (
        db.query(SceneTruthSet)
        .filter(SceneTruthSet.project_id == project_id)
        .delete(synchronize_session=False)
    )
    # The cascade does the rest; deleting jobs row-by-row would emit thousands
    # of statements for a listing that has a full pairwise relation matrix.
    deleted_jobs = (
        db.query(Job)
        .filter(Job.project_id == project_id)
        .delete(synchronize_session=False)
    )
    db.commit()

    logger.info(
        "PROJECT_PURGED project_id=%s jobs=%s photos=%s relations=%s truth_sets=%s",
        project_id, deleted_jobs, counts.get("job_photos", 0),
        counts.get("photo_relations", 0), truth,
    )
    return ProjectPurgeResponse(
        project_id=project_id,
        jobs=int(deleted_jobs),
        truth_sets=int(truth),
        detail=("Nothing to purge." if not deleted_jobs and not truth
                else "Media-service data removed."),
        **counts,
    )


def _latest_job_id(db: Session, project_id: str) -> int | None:
    row = (db.query(Job.id).filter(Job.project_id == project_id)
           .order_by(Job.id.desc()).first())
    return int(row[0]) if row else None


def _timeline_for_project(db: Session, project_id: str,
                          timeline_id: int | None = None):
    if timeline_id is not None:
        timeline = (db.query(Timeline)
                    .join(Job, Timeline.job_id == Job.id)
                    .filter(Timeline.id == int(timeline_id),
                            Job.project_id == project_id)
                    .first())
        if timeline is None:
            raise HTTPException(status_code=404, detail="Studio timeline not found.")
        return timeline
    job_id = _latest_job_id(db, project_id)
    if job_id is None:
        return None
    return latest_timeline(db, job_id)


def _timeline_summary(timeline) -> TimelineSummary:
    grid = timeline.beat_grid or {}
    return TimelineSummary(
        id=int(timeline.id),
        version=int(timeline.version or 1),
        status=timeline.status or "draft",
        name=str(grid.get("name") or f"Video {timeline.version or 1}"),
        bpm=timeline.bpm,
        duration=float(timeline.total_duration or 0.0),
        target_seconds=_safe_float(grid.get("target_seconds")),
    )


def _clip_error(clip: TimelineClip) -> str | None:
    """The stored failure reason, surfaced only while the shot is failed."""
    if clip.clip is None or clip.state != "failed":
        return None
    reason = (clip.clip.render_metadata or {}).get("error")
    return str(reason)[:500] if reason else None


def _clip_cost(clip: TimelineClip) -> float | None:
    if clip.clip is None:
        return None
    return _safe_float((clip.clip.render_metadata or {}).get("provider_cost_usd"))


def _timeline_payload(db: Session, project_id: str, timeline) -> TimelineResponse:
    """Serialise a stored timeline. Timing is recomputed here, never trusted."""
    if timeline is None:
        return TimelineResponse(project_id=project_id,
                                detail="No timeline yet. POST to create one.")

    clips = timeline.ordered_clips()
    timeline.retime()
    paired_room_ids = {
        c.room_id for c in clips
        if c.room_id and c.kind == "pair_keyframes"
    }

    photo_ids = [pid for c in clips for pid in (c.photo_ids or [])]
    photo_rows = (db.query(JobPhoto).filter(JobPhoto.id.in_(photo_ids)).all()
                  if photo_ids else [])
    photo_map = {int(p.id): p for p in photo_rows}

    def photo(pid: int) -> TimelinePhoto:
        row = photo_map.get(int(pid))
        if row is None:
            return TimelinePhoto(id=int(pid))
        return TimelinePhoto(id=int(row.id), thumbnail_url=_signed_url(row.s3_uri),
                             position=row.position, room_label=row.room_label)

    grid = timeline.grid_seconds
    music_meta = (timeline.beat_grid or {}).get("music")
    music = None
    if isinstance(music_meta, dict):
        source = music_meta.get("source")
        if source in {"catalog", "upload"}:
            timeline_start = _safe_float(music_meta.get("timeline_start")) or 0.0
            music = TimelineMusic(
                source=source,
                title=str(music_meta.get("title") or "Custom track"),
                bpm=timeline.bpm,
                beat_offset=float((timeline.beat_grid or {}).get("beat_offset") or 0.0),
                start_offset=max(0.0, -timeline_start),
                timeline_start=timeline_start,
                duration=_safe_float(music_meta.get("duration")),
                preview_url=_music_preview_url(timeline.job.music_uri if timeline.job else None),
            )
    shots = []
    for index, c in enumerate(clips):
        # Ready clips expose the immutable prompt that was paid for. Planned
        # clips expose the current prompt so the user can inspect it first.
        generation_prompt = _timeline_generation_prompt(
            c,
            index,
            len(clips),
            room_has_pair=c.room_id in paired_room_ids,
        )
        shots.append(TimelineShot(
            shot_uid=c.shot_uid, order_index=c.order_index, state=c.state,
            room_id=c.room_id, room_type=c.room_type, act=c.act, kind=c.kind,
            is_close_up=bool(c.is_close_up),
            in_time=c.in_time or 0.0, duration=c.duration_for(grid),
            trim_start=c.trim_start or 0.0, max_duration=c.max_duration(),
            duration_override=c.duration_override,
            # The move the prompt will actually use, not the raw stored label:
            # for an unlocked single, position still decides the push or pull.
            move=(effective_move_label(
                c,
                is_first=index == 0,
                is_last=index == len(clips) - 1,
                room_has_pair=c.room_id in paired_room_ids,
            )
                  if c.kind != "pair_keyframes" else c.move_label),
            move_vector=c.move_vector,
            crop_box=c.crop_box, subject_box=c.subject_box,
            source_size=c.source_size, detail_subject=c.detail_subject,
            prompt_override=c.prompt_override,
            prompt_note=c.prompt_note,
            move_locked=bool(c.move_locked),
            generation_prompt=generation_prompt,
            video_url=_signed_url(c.clip.s3_uri) if c.clip else None,
            source_clip_id=int(c.clip_id) if c.clip_id is not None else None,
            error=_clip_error(c),
            cost_usd=_clip_cost(c),
            photos=[photo(pid) for pid in (c.photo_ids or [])],
        ))
    return TimelineResponse(
        project_id=project_id, timeline_id=timeline.id,
        analysis_status=timeline.job.status if timeline.job else None,
        analysis_ready=True, version=timeline.version,
        status=timeline.status or "draft", bpm=timeline.bpm,
        beats_per_shot=timeline.beats_per_shot, shot_seconds=grid,
        source_seconds=TL_SOURCE_SECONDS, duration=timeline.total_duration or 0.0,
        shots=shots, music=music,
    )


@app.get("/api/projects/{project_id}/timelines", response_model=TimelineListResponse)
async def list_timelines(project_id: str, db: Session = Depends(get_db)):
    """List every saved video cut for one listing."""
    job_id = _latest_job_id(db, project_id)
    if job_id is None:
        return TimelineListResponse(project_id=project_id)
    timelines = (db.query(Timeline).filter(Timeline.job_id == job_id)
                 .order_by(Timeline.id.asc()).all())
    return TimelineListResponse(
        project_id=project_id,
        timelines=[_timeline_summary(timeline) for timeline in timelines],
    )


@app.get("/api/projects/{project_id}/timeline", response_model=TimelineResponse)
async def get_timeline(project_id: str, timeline_id: int | None = None,
                       db: Session = Depends(get_db)):
    """The stored, editable timeline. Empty until one is materialised."""
    timeline = _timeline_for_project(db, project_id, timeline_id)
    if timeline is None:
        job_id = _latest_job_id(db, project_id)
        if job_id is None:
            return TimelineResponse(
                project_id=project_id,
                detail="No analysis job for this listing. Run analysis first.",
            )

        job = db.get(Job, job_id)
        status = str(job.status) if job and job.status else None
        ready = status == "analysis_complete"
        detail = (
            "Analysis is ready. Create a video to build its timeline."
            if ready
            else f"Analysis is not ready yet ({status or 'unknown'})."
        )
        return TimelineResponse(
            project_id=project_id,
            analysis_status=status,
            analysis_ready=ready,
            detail=detail,
        )
    return _timeline_payload(db, project_id, timeline)


@app.delete("/api/projects/{project_id}/timeline")
async def delete_timeline(project_id: str, timeline_id: int | None = None,
                          db: Session = Depends(get_db)):
    """Delete one saved studio timeline and its editable shot rows."""
    timeline = _timeline_for_project(db, project_id, timeline_id)
    if timeline is None:
        raise HTTPException(status_code=404, detail="Timeline not found.")

    deleted_id = int(timeline.id)
    db.delete(timeline)
    db.commit()
    return {
        "project_id": project_id,
        "timeline_id": deleted_id,
        "detail": "Timeline deleted.",
    }


@app.post("/api/projects/{project_id}/timeline", response_model=TimelineResponse)
async def create_timeline(project_id: str, seconds: float = 45.0, bpm: float = 100.0,
                          force: bool = False, name: str | None = None,
                          populate: bool = True,
                          db: Session = Depends(get_db)):
    """Materialise the computed shot line into an editable timeline.

    The first video is populated from the analyzed rooms by default. Later
    studio creation passes `populate=false` so the new cut starts blank.
    Returns the existing one unless `force` -- re-materialising discards manual
    arrangement, so it has to be asked for rather than happening on a refresh.
    """
    job_id = _latest_job_id(db, project_id)
    if job_id is None:
        raise HTTPException(status_code=404, detail="No job for this project.")

    existing = latest_timeline(db, job_id)
    if existing is not None and not force:
        return _timeline_payload(db, project_id, existing)

    rooms = []
    if populate:
        from scripts.build_shot_line import load_rooms
        try:
            rooms, _ = load_rooms(db, project_id)
        except SystemExit as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    timeline = materialize_timeline(db, job_id, rooms,
                                    target_seconds=seconds, bpm=bpm, name=name)
    return _timeline_payload(db, project_id, timeline)


@app.post("/api/projects/{project_id}/timeline/populate", response_model=TimelineResponse)
async def populate_timeline(project_id: str, timeline_id: int | None = None,
                            db: Session = Depends(get_db)):
    """Fill one blank studio with the current analyzed shot plan."""
    timeline = _timeline_for_project(db, project_id, timeline_id)
    if timeline is None:
        raise HTTPException(status_code=404, detail="No timeline to populate.")
    if timeline.timeline_clips:
        raise HTTPException(status_code=409,
                            detail="Timeline already contains shots.")

    from scripts.build_shot_line import load_rooms
    try:
        rooms, _ = load_rooms(db, project_id)
    except SystemExit as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    grid = timeline.beat_grid or {}
    target_seconds = _safe_float(grid.get("target_seconds")) or 45.0
    timeline = populate_timeline_rows(
        db,
        timeline,
        rooms,
        target_seconds=target_seconds,
        bpm=timeline.bpm,
    )
    return _timeline_payload(db, project_id, timeline)


@app.put("/api/projects/{project_id}/timeline", response_model=TimelineResponse)
async def update_timeline(project_id: str, payload: TimelineUpdateRequest,
                          timeline_id: int | None = None,
                          db: Session = Depends(get_db)):
    """Replace the shot list with the order given.

    A drag is a reorder, an insert and a delete at once, so the whole list is
    sent and applied atomically; a half-applied drag is worse than a refused one.
    Rows are matched by `shot_uid` so a generated clip survives being moved.
    """
    job_id = _latest_job_id(db, project_id)
    timeline = _timeline_for_project(db, project_id, timeline_id) if job_id else None
    if timeline is None:
        raise HTTPException(status_code=404, detail="No timeline to update.")

    if payload.bpm or payload.beats_per_shot:
        grid = dict(timeline.beat_grid or {})
        if payload.bpm:
            grid["bpm"] = payload.bpm
        if payload.beats_per_shot:
            grid["beats_per_shot"] = payload.beats_per_shot
        timeline.beat_grid = grid

    if payload.music_timeline_start is not None:
        grid = dict(timeline.beat_grid or {})
        music_meta = grid.get("music")
        if isinstance(music_meta, dict):
            music_meta = dict(music_meta)
            timeline_start = float(payload.music_timeline_start)
            music_meta["timeline_start"] = timeline_start
            music_meta["start_offset"] = max(0.0, -timeline_start)
            grid["music"] = music_meta
            timeline.beat_grid = grid

    existing = {c.shot_uid: c for c in timeline.timeline_clips if c.shot_uid}
    keep: set[str] = set()

    for index, incoming in enumerate(payload.shots):
        row = existing.get(incoming.shot_uid) if incoming.shot_uid else None
        source_clip = None
        if incoming.source_clip_id is not None:
            allowed_statuses = ["complete", "rendered"] if row is not None else ["complete"]
            source_clip = (db.query(Clip)
                           .filter(Clip.id == int(incoming.source_clip_id),
                                   Clip.job_id == job_id,
                                   Clip.status.in_(allowed_statuses),
                                   Clip.s3_uri.isnot(None))
                           .first())
            if source_clip is None:
                raise HTTPException(
                    status_code=422,
                    detail="source_clip_id must be a completed video from this listing.",
                )
        if row is None:
            # A photo dropped onto the timeline. It is a real row immediately --
            # duration comes from the beat grid, not from a video that does not
            # exist yet, so the schedule is correct before generation finishes.
            restored_clip = source_clip
            shot_uid = incoming.shot_uid or str(uuid.uuid4())
            if restored_clip is None and incoming.shot_uid:
                # Undo can send back a shot_uid whose TimelineClip row was
                # removed. Keep the identity stable and reuse the rendered
                # Clip row when one exists, instead of turning the undo into a
                # silent regeneration.
                restored_clip = (db.query(Clip)
                                 .filter(Clip.job_id == job_id,
                                         Clip.shot_uid == incoming.shot_uid)
                                 .order_by(Clip.id.desc()).first())
            row = TimelineClip(timeline_id=timeline.id,
                               shot_uid=shot_uid,
                               transition_type="cut")
            db.add(row)
            row.room_id = incoming.room_id
            row.room_type = incoming.room_type
            row.act = incoming.act
            row.move_label = incoming.move or (
                source_clip.motion_type if source_clip is not None else None
            ) or "push in + pan right"
            if restored_clip is not None:
                row.clip_id = restored_clip.id
        elif source_clip is not None:
            row.clip_id = source_clip.id
            if not incoming.move:
                row.move_label = source_clip.motion_type or row.move_label
        row.order_index = index
        incoming_ids = list(incoming.photo_ids)
        # Reversing the frames reverses the move. Detected here rather than
        # asked for, because it is a fact about the shot, not an option: the
        # camera that travelled A -> B now travels B -> A. It matters beyond
        # display -- a pair prompt follows reference order, but the moment it
        # is downgraded to a single image this label starts driving the camera,
        # and a stale one would point the wrong way.
        if (len(incoming_ids) == 2 and row.photo_ids
                and list(row.photo_ids) == list(reversed(incoming_ids))):
            row.move_label = reverse_label(row.move_label)
            row.move_vector = reverse_vector(row.move_vector)
        row.photo_ids = incoming_ids
        row.kind = incoming.kind
        row.trim_start = max(0.0, float(incoming.trim_start or 0.0))
        row.duration_override = incoming.duration_override
        row.prompt_override = incoming.prompt_override
        row.prompt_note = incoming.prompt_note
        # A customer composing a move changes the stored label and the vector the
        # renderer reads. The editor is sent the *effective* move, so a save that
        # echoes it back is not a choice and must not lock the shot -- otherwise
        # merely dragging a clip would freeze every move on the timeline.
        roll = float((row.move_vector or {}).get("roll_deg") or 0.0)
        if incoming.move is None and row.move_locked:
            # An explicit clear is the way back to the planner's own choice.
            # Without it a composed move is a one-way door, and the customer
            # who overrode one shot can never see what we would have done.
            row.move_locked = False
            row.move_chosen = None
            row.move_vector = vector_from_label(row.move_label, roll)
        elif incoming.move and incoming.kind != "pair_keyframes":
            chosen = compose_label(axes_of(incoming.move))
            effective = effective_move_label(
                row, is_first=index == 0, is_last=index == len(payload.shots) - 1)
            if chosen != effective:
                # `move_label` keeps the measured move on purpose. Overwriting
                # it would destroy the only thing a reset could restore.
                row.move_locked = True
                row.move_chosen = chosen
                row.move_vector = vector_from_label(chosen, roll)
        keep.add(row.shot_uid)

    for uid, row in existing.items():
        if uid not in keep:
            db.delete(row)

    db.flush()
    db.expire(timeline, ["timeline_clips"])
    timeline.retime()
    db.commit()
    db.refresh(timeline)
    return _timeline_payload(db, project_id, timeline)


def _valid_music_uri(uri: str) -> bool:
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return bool(parsed.netloc and parsed.path.strip("/"))
    if parsed.scheme == "https":
        return bool(parsed.netloc)
    return parsed.scheme == "http" and settings.ENVIRONMENT == "development" and bool(parsed.netloc)


def _center_trim_for(clip: TimelineClip, grid_seconds: float) -> float:
    """Return the stable middle start for the current beat-length window."""
    source = source_seconds_for(clip.kind)
    wanted = float(clip.duration_override or grid_seconds)
    wanted = max(0.1, min(wanted, source))
    return max(0.0, (source - wanted) / 2.0)


@app.put("/api/projects/{project_id}/timeline/music", response_model=TimelineResponse)
async def update_timeline_music(project_id: str, payload: TimelineMusicRequest,
                                timeline_id: int | None = None,
                                db: Session = Depends(get_db)):
    """Select a catalog/uploaded track and retime the editable timeline."""
    if not _valid_music_uri(payload.music_uri):
        raise HTTPException(
            status_code=422,
            detail="music_uri must be an s3:// URI, an https:// URL, or local http:// URL.",
        )

    job_id = _latest_job_id(db, project_id)
    timeline = _timeline_for_project(db, project_id, timeline_id) if job_id else None
    if timeline is None:
        raise HTTPException(status_code=404, detail="No timeline to update.")
    job = db.get(Job, job_id)

    old_grid = timeline.grid_seconds
    centered = {
        clip.id: abs((clip.trim_start or 0.0) - _center_trim_for(clip, old_grid)) <= 0.025
        for clip in timeline.ordered_clips()
    }
    if payload.bpm is None:
        try:
            analysis = analyze_music_uri(payload.music_uri)
        except MusicAnalysisError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        bpm = analysis.bpm
        beat_offset = analysis.beat_offset
        duration = analysis.duration
    else:
        bpm = float(payload.bpm)
        beat_offset = float(payload.beat_offset or 0.0)
        duration = payload.duration
    if payload.beat_offset is not None:
        beat_offset = float(payload.beat_offset)

    title = " ".join(payload.title.split()) or "Custom track"
    grid = dict(timeline.beat_grid or {})
    grid.update({
        "bpm": bpm,
        "beat_offset": beat_offset,
        "music": {
            "source": payload.source,
            "title": title,
            "duration": duration,
            "timeline_start": float(payload.timeline_start or 0.0),
        },
    })
    timeline.beat_grid = grid
    if job is not None:
        job.music_uri = payload.music_uri
        job.bpm = int(round(bpm))
        job.beat_offset = beat_offset

    new_grid = timeline.grid_seconds
    for clip in timeline.ordered_clips():
        # Selecting music should keep the automatic middle window centred, but
        # never erase a trim the editor deliberately moved.
        if centered.get(clip.id):
            clip.trim_start = _center_trim_for(clip, new_grid)
    timeline.retime()
    db.commit()
    db.refresh(timeline)
    return _timeline_payload(db, project_id, timeline)


@app.get("/api/projects/{project_id}/timeline/candidates",
         response_model=CandidateListResponse)
async def get_timeline_candidates(project_id: str, timeline_id: int | None = None,
                                  db: Session = Depends(get_db)):
    """Unused photos and reusable generated videos for the source-media rail.

    Each carries the verified partner it could form a pair shot with, when one
    exists and is also unused -- two real endpoints beat one, so the UI should
    offer that rather than making the user find it.
    """
    job_id = _latest_job_id(db, project_id)
    if job_id is None:
        return CandidateListResponse(project_id=project_id)

    timeline = _timeline_for_project(db, project_id, timeline_id)
    used = {int(pid) for c in (timeline.timeline_clips if timeline else [])
            for pid in (c.photo_ids or [])}
    used_clip_ids = {int(c.clip_id) for c in (timeline.timeline_clips if timeline else [])
                     if c.clip_id is not None}

    photos = (db.query(JobPhoto)
              .filter(JobPhoto.job_id == job_id)
              .filter(JobPhoto.exclude.isnot(True))
              .filter(JobPhoto.is_duplicate.isnot(True))
              .order_by(JobPhoto.position).all())
    free = [p for p in photos if int(p.id) not in used]
    free_ids = {int(p.id) for p in free}

    partner: dict[int, int] = {}
    if free_ids:
        for a, b, metrics in db.execute(text(
            "SELECT photo_a_id, photo_b_id, debug_metrics FROM photo_relations "
            "WHERE job_id = :j"), {"j": job_id}).fetchall():
            a, b = int(a), int(b)
            if a not in free_ids or b not in free_ids:
                continue
            if a in partner or b in partner:
                continue
            if persisted_pair_score(a, b, metrics or {}) > 0.0:
                partner[a] = b
                partner[b] = a

    video_clips = (db.query(Clip)
                   .filter(Clip.job_id == job_id,
                           Clip.status == "complete",
                           Clip.s3_uri.isnot(None))
                   .order_by(Clip.id.asc()).all())
    video_clips = [clip for clip in video_clips if int(clip.id) not in used_clip_ids]
    video_photo_ids = {
        int(photo_id)
        for clip in video_clips
        for photo_id in (clip.source_photo_ids or [])
    }
    video_photo_rows = (db.query(JobPhoto)
                        .filter(JobPhoto.id.in_(video_photo_ids)).all()
                        if video_photo_ids else [])
    video_photo_map = {int(photo.id): photo for photo in video_photo_rows}
    videos = []
    for clip in video_clips:
        source_ids = [int(photo_id) for photo_id in (clip.source_photo_ids or [])]
        first_photo = video_photo_map.get(source_ids[0]) if source_ids else None
        videos.append(CandidateVideo(
            id=int(clip.id),
            video_url=_signed_url(clip.s3_uri),
            thumbnail_url=_signed_url(first_photo.s3_uri) if first_photo else None,
            duration=float(clip.duration or TL_SOURCE_SECONDS),
            kind="pair_keyframes" if len(source_ids) == 2 else "single_image",
            room_label=first_photo.room_label if first_photo else None,
            room_id=None,
            source_photo_ids=source_ids,
            motion_type=clip.motion_type,
        ))

    return CandidateListResponse(
        project_id=project_id,
        photos=[CandidatePhoto(id=int(p.id), thumbnail_url=_signed_url(p.s3_uri),
                               position=p.position, room_label=p.room_label,
                               pair_with=partner.get(int(p.id)))
                for p in free],
        videos=videos,
    )


@app.post("/api/projects/{project_id}/timeline/generate")
async def generate_timeline(project_id: str):
    """Generation is dispatched by Rails through SQS, never from this API."""
    raise HTTPException(
        status_code=410,
        detail="Timeline generation must be queued by Rails.",
    )


@app.post("/api/projects/{project_id}/timeline/assemble")
async def assemble_timeline(project_id: str):
    """Assembly is dispatched by Rails through SQS, never from this API."""
    raise HTTPException(
        status_code=410,
        detail="Timeline assembly must be queued by Rails.",
    )


@app.get("/api/projects/{project_id}/scenes/debug", response_model=SceneDebugResponse)
async def get_project_scenes_debug(project_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.project_id == project_id).order_by(Job.created_at.desc()).first()
    if not job:
        return SceneDebugResponse(project_id=project_id, job_id=None, components=[], photo_geometries=[], motion_decisions=[], shot_plan=None)

    photos = db.query(JobPhoto).filter(JobPhoto.job_id == job.id).all()
    photo_map = {int(photo.id): photo for photo in photos}
    geometry_map = _photo_geometry_map(db, int(job.id))
    membership_map = _membership_map(db, int(job.id))
    components = db.query(SceneComponent).filter(SceneComponent.job_id == job.id).order_by(SceneComponent.id.asc()).all()
    clusters = db.query(RoomCluster).filter(RoomCluster.job_id == job.id).order_by(RoomCluster.sequence_order.asc().nulls_last()).all()
    analysis_map = {
        analysis.room_cluster_id: analysis
        for analysis in db.query(AnalysisResult).filter(AnalysisResult.job_id == job.id).all()
        if analysis.room_cluster_id is not None
    }

    component_summaries: list[SceneComponentSummary] = []
    for component in components:
        memberships = [membership for membership in component.memberships if membership.photo_id in photo_map]
        ordered_memberships = sorted(memberships, key=lambda membership: (membership.order_index or 0, membership.photo_id))
        ordered_photo_ids = [int(membership.photo_id) for membership in ordered_memberships]
        bridge_photo_ids = [int(membership.photo_id) for membership in memberships if membership.photo_role == "bridge"]
        outlier_photo_ids = [
            int(membership.photo_id)
            for membership in memberships
            if geometry_map.get(int(membership.photo_id)) is not None
            and _safe_float(geometry_map[int(membership.photo_id)].pose_confidence) is not None
            and float(geometry_map[int(membership.photo_id)].pose_confidence) < 0.45
        ]
        recommended_motion = next(
            (cluster.recommended_motion for cluster in clusters if int(cluster.scene_component_id or 0) == int(component.id) and cluster.recommended_motion),
            component.motion_affordance,
        )
        component_summaries.append(
            SceneComponentSummary(
                component_id=int(component.id),
                component_key=component.component_key,
                scene_type=component.scene_type,
                photo_ids=[int(membership.photo_id) for membership in memberships],
                ordered_photo_ids=ordered_photo_ids,
                hero_photo_id=_safe_int(component.hero_photo_id),
                bridge_photo_ids=bridge_photo_ids,
                outlier_photo_ids=outlier_photo_ids,
                geometry_confidence=_safe_float(component.geometry_confidence),
                connectivity_confidence=_safe_float(component.connectivity_confidence),
                track_coverage=_safe_float(component.track_coverage),
                avg_reprojection_error=_safe_float(component.avg_reprojection_error),
                depth_range=_safe_float(component.depth_range),
                motion_affordance=component.motion_affordance,
                recommended_motion=recommended_motion,
                debug_metrics=component.debug_metrics or {},
            )
        )

    photo_geometries = [
        _serialize_photo_debug(photo, geometry_map.get(int(photo.id)), membership_map.get(int(photo.id)))
        for photo in photos
    ]

    motion_decisions: list[MotionDecisionDebug] = []
    for cluster in clusters:
        analysis = analysis_map.get(cluster.id)
        motion_decisions.append(
            MotionDecisionDebug(
                cluster_id=int(cluster.id),
                room_type=cluster.room_type,
                scene_component_id=_safe_int(cluster.scene_component_id),
                confidence_tier=cluster.confidence_tier,
                recommended_motion=cluster.recommended_motion,
                recommended_duration=_safe_float(cluster.recommended_duration),
                model_recommendation=analysis.model_recommendation if analysis is not None else None,
                geometry_confidence=_safe_float(cluster.geometry_confidence),
                motion_affordance=(analysis.debug_metrics or {}).get("matching_inferred_motion") if analysis and isinstance(analysis.debug_metrics, dict) else cluster.recommended_motion,
                decision_metrics=analysis.debug_metrics if analysis is not None and isinstance(analysis.debug_metrics, dict) else {},
            )
        )

    return SceneDebugResponse(
        project_id=project_id,
        job_id=int(job.id),
        components=component_summaries,
        photo_geometries=photo_geometries,
        motion_decisions=motion_decisions,
        shot_plan=_shot_plan_response(project_id, job, db),
    )


@app.get("/api/projects/{project_id}/shot_plan", response_model=ShotPlanResponse)
async def get_project_shot_plan(project_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.project_id == project_id).order_by(Job.created_at.desc()).first()
    if not job:
        raise HTTPException(status_code=404, detail="No job found for project")
    return _shot_plan_response(project_id, job, db)


@app.post("/api/projects/{project_id}/relations/debug", response_model=PhotoRelationDebugResponse)
async def debug_photo_relation(project_id: str, payload: PhotoRelationDebugRequest, db: Session = Depends(get_db)):
    job_query = db.query(Job).filter(Job.project_id == project_id)
    if payload.job_id is not None:
        job_query = job_query.filter(Job.id == payload.job_id)
    job = job_query.order_by(Job.created_at.desc()).first()
    if not job:
        raise HTTPException(status_code=404, detail="No job found for project")

    photo_map = {
        int(photo.id): photo
        for photo in db.query(JobPhoto)
        .filter(JobPhoto.job_id == job.id)
        .filter(JobPhoto.id.in_([payload.left_photo_id, payload.right_photo_id]))
        .all()
    }
    left_photo = photo_map.get(int(payload.left_photo_id))
    right_photo = photo_map.get(int(payload.right_photo_id))
    if left_photo is None or right_photo is None:
        raise HTTPException(status_code=404, detail="Requested photos were not found in the latest job")

    geometry_map = _photo_geometry_map(db, int(job.id))
    membership_map = _membership_map(db, int(job.id))
    relation = (
        db.query(PhotoRelation)
        .filter(PhotoRelation.job_id == job.id)
        .filter(PhotoRelation.photo_a_id == min(int(left_photo.id), int(right_photo.id)))
        .filter(PhotoRelation.photo_b_id == max(int(left_photo.id), int(right_photo.id)))
        .first()
    )

    return PhotoRelationDebugResponse(
        project_id=project_id,
        job_id=int(job.id),
        relation=_serialize_relation_debug(relation, photo_map) if relation is not None else None,
        left_photo=_serialize_photo_debug(left_photo, geometry_map.get(int(left_photo.id)), membership_map.get(int(left_photo.id))),
        right_photo=_serialize_photo_debug(right_photo, geometry_map.get(int(right_photo.id)), membership_map.get(int(right_photo.id))),
    )


def _truth_validation_warnings(
    payload: SceneTruthSetPayload,
    valid_photo_keys: set[str],
    describe: dict[str, str] | None = None,
) -> list[str]:
    """Validate a truth set per the Scene-Graph V2 plan (Stage 0, §3.2).

    `describe` maps photo keys to what the labeler actually sees on the tile
    ("#39"); warnings quoting raw UUIDs are useless while labeling.
    """
    def shown(photo_key: str) -> str:
        if describe and photo_key in describe:
            return describe[photo_key]
        return f"unknown photo {photo_key[:8]}"

    warnings: list[str] = []
    seen: dict[str, str] = {}
    instance_names: set[str] = set()

    for instance in payload.room_instances:
        name = (instance.instance or "").strip()
        if not name:
            warnings.append("A room instance has no name")
            continue
        if name in instance_names:
            warnings.append(f"Duplicate room instance name: {name}")
        instance_names.add(name)
        for photo_key in instance.photo_keys:
            if photo_key not in valid_photo_keys:
                warnings.append(f"{name}: photo {shown(photo_key)} does not belong to this listing")
                continue
            previous = seen.get(photo_key)
            if previous is not None:
                warnings.append(f"Photo {shown(photo_key)} is in two rooms ({previous} and {name})")
            seen[photo_key] = name

    for group in payload.open_plan_groups:
        for name in group:
            if name not in instance_names:
                warnings.append(f"Open-plan group references unknown room: {name}")
        if len(group) < 2:
            warnings.append("An open-plan group needs at least two rooms")

    def _same_open_plan(left: str, right: str) -> bool:
        """Two rooms the labeler linked as one connected open space."""
        left_room, right_room = seen.get(left), seen.get(right)
        if left_room is None or right_room is None:
            return False
        return any(left_room in group and right_room in group for group in payload.open_plan_groups)

    def _check_pairs(pairs: list[list[str]], label: str, require_same_room: bool | None, allow_open_plan: bool = False) -> None:
        for pair in pairs:
            if len(pair) != 2:
                warnings.append(f"{label}: entries must be photo pairs")
                continue
            left, right = pair[0], pair[1]
            for photo_key in (left, right):
                if photo_key not in valid_photo_keys:
                    warnings.append(f"{label}: photo {shown(photo_key)} does not belong to this listing")
            if left == right:
                warnings.append(f"{label}: a photo cannot pair with itself ({shown(left)})")
                continue
            same_room = seen.get(left) is not None and seen.get(left) == seen.get(right)
            # A cinematic pair may legitimately span an open-plan seam (kitchen ->
            # living room); a duplicate may not, since it is the same view twice.
            acceptable = same_room or (allow_open_plan and _same_open_plan(left, right))
            if require_same_room is True and not acceptable:
                warnings.append(f"{label}: photos {shown(left)} + {shown(right)} are not in the same room instance")
            if require_same_room is False and same_room:
                warnings.append(f"{label}: photos {shown(left)} + {shown(right)} are marked must-not-group but share room {seen.get(left)}")

    _check_pairs(payload.duplicates, "Duplicates", True)
    _check_pairs(payload.preferred_pairs, "Preferred pairs", True, allow_open_plan=True)
    _check_pairs(payload.must_not_group, "Must-not-group", False)
    # A story bridge is only meaningful between two *different* rooms.
    _check_pairs(payload.story_bridges, "Story bridges", False)
    return warnings


def _rails_truth_photos(db: Session, project_id: str) -> tuple[list[TruthPhoto], set[str], int]:
    """Photos straight from the Rails table, for listings not analyzed yet.

    Labeling only needs the photos; the analysis job just adds the v1 overlay.
    Ground truth is keyed by the Rails photo id either way, so labels made here
    stay valid once the listing is analyzed.
    """
    rows = db.execute(
        text(
            "SELECT id, position, filename, room_type, s3_object_key "
            "FROM photos WHERE project_id = :project_id "
            "ORDER BY position NULLS LAST, created_at"
        ),
        {"project_id": project_id},
    ).fetchall()
    result = [
        TruthPhoto(
            photo_key=str(row.id),
            photo_id=0,  # no job_photos row yet
            position=int(row.position) if row.position is not None else index,
            filename=row.filename,
            thumbnail_url=_signed_url(row.s3_object_key),
            room_label=row.room_type,
            predicted_component_id=None,
        )
        for index, row in enumerate(rows)
    ]
    return result, {photo.photo_key for photo in result}, 0


def _truth_photos(db: Session, job: Job) -> tuple[list[TruthPhoto], set[str], int]:
    photos = (
        db.query(JobPhoto)
        .filter(JobPhoto.job_id == job.id)
        .order_by(JobPhoto.position.asc().nulls_last(), JobPhoto.id.asc())
        .all()
    )
    membership_map = _membership_map(db, int(job.id))
    component_ids: set[int] = set()
    result: list[TruthPhoto] = []
    for index, photo in enumerate(photos):
        membership = membership_map.get(int(photo.id))
        component_id = _safe_int(getattr(membership, "scene_component_id", None))
        if component_id is not None:
            component_ids.add(component_id)
        result.append(
            TruthPhoto(
                photo_key=str(photo.rails_photo_id),
                photo_id=int(photo.id),
                position=_safe_int(photo.position) if photo.position is not None else index,
                filename=photo.filename,
                thumbnail_url=_signed_url(photo.s3_uri),
                room_label=photo.room_override or photo.room_label,
                predicted_component_id=component_id,
            )
        )
    return result, {photo.photo_key for photo in result}, len(component_ids)


def _truth_response(
    project_id: str,
    job: Job | None,
    truth: SceneTruthSet | None,
    db: Session,
    warnings: list[str] | None = None,
) -> SceneTruthSetResponse:
    photos, valid_keys, predicted_components = (
        _truth_photos(db, job) if job is not None else _rails_truth_photos(db, project_id)
    )
    room_instances = [
        RoomInstancePayload(
            instance=entry.get("instance", ""),
            photo_keys=[str(value) for value in entry.get("photo_keys", [])],
        )
        for entry in (truth.room_instances if truth else []) or []
    ]
    labeled_keys = {key for entry in room_instances for key in entry.photo_keys}
    # Labels referencing photos that are no longer in the listing (deleted or
    # replaced between analysis runs) are reported rather than silently dropped.
    stale = sorted(labeled_keys - valid_keys)
    return SceneTruthSetResponse(
        project_id=project_id,
        job_id=int(job.id) if job is not None else None,
        listing_slug=(truth.listing_slug if truth else "") or "",
        split=(truth.split if truth else "calibration") or "calibration",
        status=(truth.status if truth else "draft") or "draft",
        room_instances=room_instances,
        open_plan_groups=(truth.open_plan_groups if truth else []) or [],
        duplicates=(truth.duplicates if truth else []) or [],
        must_not_group=(truth.must_not_group if truth else []) or [],
        preferred_pairs=(truth.preferred_pairs if truth else []) or [],
        story_bridges=(truth.story_bridges if truth else []) or [],
        notes=(truth.notes if truth else "") or "",
        labeled_by=(truth.labeled_by if truth else "") or "",
        photos=photos,
        photo_count=len(photos),
        labeled_count=len(labeled_keys & valid_keys),
        predicted_component_count=predicted_components,
        warnings=warnings or [],
        stale_photo_keys=stale,
        revision=int(truth.revision or 1) if truth else 1,
        updated_at=truth.updated_at.isoformat() if truth is not None and truth.updated_at else None,
    )


def _latest_job(db: Session, project_id: str) -> Job | None:
    return db.query(Job).filter(Job.project_id == project_id).order_by(Job.created_at.desc()).first()


@app.get("/api/projects/{project_id}/truth", response_model=SceneTruthSetResponse)
async def get_project_truth(project_id: str, db: Session = Depends(get_db)):
    """Labeling state for a project: photos, saved labels, and current predictions."""
    job = _latest_job(db, project_id)
    truth = db.query(SceneTruthSet).filter(SceneTruthSet.project_id == project_id).first()
    photos, valid_keys, _components = (
        _truth_photos(db, job) if job is not None else _rails_truth_photos(db, project_id)
    )
    if not photos:
        raise HTTPException(status_code=404, detail="This project has no photos yet")
    # Surface problems while labeling rather than only when "complete" is refused.
    warnings: list[str] = []
    if truth is not None:
        stored = SceneTruthSetPayload(
            listing_slug=truth.listing_slug or "",
            split=truth.split or "calibration",
            status=truth.status or "draft",
            room_instances=[
                RoomInstancePayload(instance=entry.get("instance", ""),
                                    photo_keys=[str(v) for v in entry.get("photo_keys", [])])
                for entry in (truth.room_instances or [])
            ],
            open_plan_groups=truth.open_plan_groups or [],
            duplicates=truth.duplicates or [],
            must_not_group=truth.must_not_group or [],
            preferred_pairs=truth.preferred_pairs or [],
            story_bridges=truth.story_bridges or [],
            notes=truth.notes or "",
            labeled_by=truth.labeled_by or "",
        )
        describe = {photo.photo_key: f"#{photo.position}" for photo in photos}
        warnings = _truth_validation_warnings(stored, valid_keys, describe)
    return _truth_response(project_id, job, truth, db, warnings)


@app.put("/api/projects/{project_id}/truth", response_model=SceneTruthSetResponse)
async def put_project_truth(project_id: str, payload: SceneTruthSetPayload, db: Session = Depends(get_db)):
    """Upsert the human ground truth for a project (survives analysis re-runs)."""
    job = _latest_job(db, project_id)
    if payload.split not in TRUTH_SPLITS:
        raise HTTPException(status_code=422, detail=f"split must be one of {TRUTH_SPLITS}")
    if payload.status not in TRUTH_STATUSES:
        raise HTTPException(status_code=422, detail=f"status must be one of {TRUTH_STATUSES}")

    _photos, valid_photo_keys, _components = (
        _truth_photos(db, job) if job is not None else _rails_truth_photos(db, project_id)
    )
    if not valid_photo_keys:
        raise HTTPException(status_code=404, detail="This project has no photos yet")
    describe = {photo.photo_key: f"#{photo.position}" for photo in _photos}
    warnings = _truth_validation_warnings(payload, valid_photo_keys, describe)
    if payload.status == "complete" and warnings:
        raise HTTPException(status_code=422, detail={"error": "Cannot mark complete while warnings remain", "warnings": warnings})

    truth = db.query(SceneTruthSet).filter(SceneTruthSet.project_id == project_id).first()
    if truth is None:
        truth = SceneTruthSet(project_id=project_id, revision=0)
        db.add(truth)

    room_instances = [
        {"instance": entry.instance.strip(), "photo_keys": sorted(set(entry.photo_keys))}
        for entry in payload.room_instances
        if entry.instance and entry.instance.strip()
    ]
    truth.last_job_id = int(job.id) if job is not None else truth.last_job_id
    truth.listing_slug = payload.listing_slug.strip()
    truth.split = payload.split
    truth.status = payload.status
    truth.room_instances = room_instances
    truth.open_plan_groups = [list(group) for group in payload.open_plan_groups]
    truth.duplicates = [sorted(pair) for pair in payload.duplicates if len(pair) == 2]
    truth.must_not_group = [sorted(pair) for pair in payload.must_not_group if len(pair) == 2]
    truth.preferred_pairs = [sorted(pair) for pair in payload.preferred_pairs if len(pair) == 2]
    truth.story_bridges = [sorted(pair) for pair in payload.story_bridges if len(pair) == 2]
    truth.notes = payload.notes
    truth.labeled_by = payload.labeled_by.strip()
    truth.photo_count = len(valid_photo_keys)
    truth.labeled_count = len({key for entry in room_instances for key in entry["photo_keys"]} & valid_photo_keys)
    truth.revision = int(truth.revision or 0) + 1
    db.commit()
    db.refresh(truth)
    return _truth_response(project_id, job, truth, db, warnings)


@app.get("/api/truth/sets", response_model=TruthSetListResponse)
async def list_truth_sets(db: Session = Depends(get_db)):
    """Corpus progress across every labeled listing."""
    rows = db.query(SceneTruthSet).order_by(SceneTruthSet.updated_at.desc().nulls_last()).all()
    summaries = [
        TruthSetSummary(
            project_id=str(row.project_id),
            last_job_id=_safe_int(row.last_job_id),
            listing_slug=row.listing_slug or "",
            split=row.split or "calibration",
            status=row.status or "draft",
            photo_count=int(row.photo_count or 0),
            labeled_count=int(row.labeled_count or 0),
            room_count=len(row.room_instances or []),
            revision=int(row.revision or 1),
            updated_at=row.updated_at.isoformat() if row.updated_at else None,
        )
        for row in rows
    ]
    return TruthSetListResponse(
        sets=summaries,
        total_labeled_photos=sum(summary.labeled_count for summary in summaries),
        complete_count=sum(1 for summary in summaries if summary.status == "complete"),
    )


@app.get("/api/projects/{project_id}/truth/export")
async def export_project_truth(project_id: str, db: Session = Depends(get_db)):
    """Fixture JSON in the Scene-Graph V2 plan schema.

    Emits stable rails_photo_ids as the join key, with 0-based upload positions
    and filenames alongside for human readability.
    """
    job = _latest_job(db, project_id)
    truth = db.query(SceneTruthSet).filter(SceneTruthSet.project_id == project_id).first()
    if truth is None:
        raise HTTPException(status_code=404, detail="No ground truth has been saved for this project")

    photos, valid_keys, _components = (
        _truth_photos(db, job) if job is not None else _rails_truth_photos(db, project_id)
    )
    position_by_key = {photo.photo_key: photo.position for photo in photos}

    def _sorted_keys(keys: list) -> list[str]:
        known = [str(key) for key in keys if str(key) in position_by_key]
        return sorted(known, key=lambda key: position_by_key[key])

    return {
        "listing": truth.listing_slug or project_id,
        "project_id": project_id,
        "job_ids": [int(job.id)] if job is not None else [],
        "split": truth.split,
        "revision": int(truth.revision or 1),
        "photos": [
            {"rails_photo_id": photo.photo_key, "position": photo.position, "filename": photo.filename}
            for photo in photos
        ],
        "room_instances": [
            {
                "instance": entry.get("instance", ""),
                "rails_photo_ids": _sorted_keys(entry.get("photo_keys", [])),
                "positions": [position_by_key[key] for key in _sorted_keys(entry.get("photo_keys", []))],
            }
            for entry in (truth.room_instances or [])
        ],
        "open_plan_groups": truth.open_plan_groups or [],
        "duplicates": [_sorted_keys(pair) for pair in (truth.duplicates or [])],
        "must_not_group": [_sorted_keys(pair) for pair in (truth.must_not_group or [])],
        "preferred_cinematic_pairs": [_sorted_keys(pair) for pair in (truth.preferred_pairs or [])],
        "story_bridges": [_sorted_keys(pair) for pair in (truth.story_bridges or [])],
        "notes": truth.notes or "",
        "labeled_by": truth.labeled_by or "",
        "stale_photo_keys": sorted(
            {key for entry in (truth.room_instances or []) for key in entry.get("photo_keys", [])} - valid_keys
        ),
        "tool": "picaivid scene-truth debug page",
    }
