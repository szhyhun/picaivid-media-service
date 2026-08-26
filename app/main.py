"""FastAPI application for Picaivid Media Service."""
from datetime import datetime
import logging
from typing import Dict

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

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


@app.on_event("startup")
async def startup_warm_models() -> None:
    # Inference belongs to the worker. Keeping the API lightweight avoids loading
    # a second multi-gigabyte model beside the local MPS worker.
    warmup_core_models(context="api", include_vggt=False, include_legacy=False)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def _s3_key_from_uri(s3_uri: str | None) -> str | None:
    if not s3_uri:
        return None
    if s3_uri.startswith("s3://"):
        parts = s3_uri[5:].split("/", 1)
        return parts[1] if len(parts) == 2 else None
    return s3_uri


def _signed_url(s3_uri: str | None) -> str | None:
    key = _s3_key_from_uri(s3_uri)
    return s3_client.generate_presigned_url(key, expires_in=3600) if key else None


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
