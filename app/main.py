"""FastAPI application for Picaivid Media Service."""
from datetime import datetime
import logging
from typing import Dict

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
