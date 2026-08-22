"""Pydantic schemas for clips and VGGT scene-debug responses."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class SourcePhotoInfo(BaseModel):
    id: int
    rails_photo_id: str
    filename: Optional[str] = None
    s3_uri: Optional[str] = None
    thumbnail_url: Optional[str] = None
    room_label: Optional[str] = None
    room_override: Optional[str] = None
    position: Optional[int] = None
    cluster_order: Optional[int] = None
    base_score: Optional[float] = None
    final_score: Optional[float] = None
    depth_variance: Optional[float] = None
    depth_layers: Optional[int] = None
    sharpness: Optional[float] = None
    exposure_score: Optional[float] = None
    composition_score: Optional[float] = None
    is_duplicate: bool = False
    duplicate_of_photo_id: Optional[int] = None
    pose_confidence: Optional[float] = None
    photo_role: Optional[str] = None


class ClipTransitionStep(BaseModel):
    from_photo_id: int
    to_photo_id: int
    from_filename: Optional[str] = None
    to_filename: Optional[str] = None
    relation_confidence: Optional[float] = None
    overlap_score: Optional[float] = None
    track_support: Optional[float] = None
    reprojection_score: Optional[float] = None
    direction_dx: Optional[float] = None
    direction_dy: Optional[float] = None
    recommendation: Optional[str] = None
    continuity_type: Optional[str] = None
    is_connected: bool = False
    is_bridge_edge: bool = False


class ClusterInfo(BaseModel):
    id: int
    room_type: Optional[str] = None
    confidence_tier: Optional[str] = None
    image_count: int = 0
    overlap_score: Optional[float] = None
    depth_variance: Optional[float] = None
    recommended_motion: Optional[str] = None
    sequence_order: Optional[int] = None
    geometry_confidence: Optional[float] = None
    scene_component_id: Optional[int] = None


class AnalysisInfo(BaseModel):
    model_config = {"protected_namespaces": ()}

    tier: Optional[str] = None
    recommended_motion: Optional[str] = None
    model_recommendation: Optional[str] = None
    cfg_scale: Optional[float] = None
    inference_steps: Optional[int] = None
    debug_metrics: Optional[Dict[str, Any]] = None


class ClipResponse(BaseModel):
    model_config = {"protected_namespaces": (), "from_attributes": True}

    id: int
    job_id: int
    room_cluster_id: Optional[int] = None
    source_photo_ids: Optional[List[int]] = None
    motion_type: Optional[str] = None
    model_used: Optional[str] = None
    is_3d: bool = False
    duration: Optional[float] = None
    prompt_used: Optional[str] = None
    s3_uri: Optional[str] = None
    video_url: Optional[str] = None
    status: str = "pending"
    source_photos: Optional[List[SourcePhotoInfo]] = None
    transition_steps: Optional[List[ClipTransitionStep]] = None
    cluster_info: Optional[ClusterInfo] = None
    analysis_info: Optional[AnalysisInfo] = None


class ClipListResponse(BaseModel):
    project_id: str
    job_id: Optional[int] = None
    job_status: Optional[str] = None
    clips: List[ClipResponse]
    total_clips: int


class SceneComponentSummary(BaseModel):
    component_id: int
    component_key: str
    scene_type: str
    photo_ids: List[int]
    ordered_photo_ids: List[int]
    hero_photo_id: Optional[int] = None
    bridge_photo_ids: List[int] = []
    outlier_photo_ids: List[int] = []
    geometry_confidence: Optional[float] = None
    connectivity_confidence: Optional[float] = None
    track_coverage: Optional[float] = None
    avg_reprojection_error: Optional[float] = None
    depth_range: Optional[float] = None
    motion_affordance: Optional[str] = None
    recommended_motion: Optional[str] = None
    debug_metrics: Dict[str, Any] = {}


class PhotoGeometryDebug(BaseModel):
    photo_id: int
    rails_photo_id: str
    filename: Optional[str] = None
    thumbnail_url: Optional[str] = None
    room_label: Optional[str] = None
    scene_component_id: Optional[int] = None
    order_index: Optional[int] = None
    photo_role: Optional[str] = None
    pose_confidence: Optional[float] = None
    depth_confidence: Optional[float] = None
    point_confidence: Optional[float] = None
    visibility_score: Optional[float] = None
    reprojection_error: Optional[float] = None
    camera_center: Optional[List[float]] = None
    view_direction: Optional[List[float]] = None
    depth_artifact_uri: Optional[str] = None
    point_map_artifact_uri: Optional[str] = None


class PhotoRelationDebug(BaseModel):
    photo_a_id: int
    photo_b_id: int
    photo_a_filename: Optional[str] = None
    photo_b_filename: Optional[str] = None
    scene_component_id: Optional[int] = None
    same_component: bool = False
    overlap_score: Optional[float] = None
    track_support: Optional[float] = None
    reprojection_score: Optional[float] = None
    relation_confidence: Optional[float] = None
    baseline_distance: Optional[float] = None
    direction_dx: Optional[float] = None
    direction_dy: Optional[float] = None
    continuity_type: Optional[str] = None
    is_bridge_edge: bool = False
    is_connected: bool = False
    relative_transform: Optional[Dict[str, Any]] = None
    debug_metrics: Dict[str, Any] = {}


class MotionDecisionDebug(BaseModel):
    cluster_id: int
    room_type: Optional[str] = None
    scene_component_id: Optional[int] = None
    confidence_tier: Optional[str] = None
    recommended_motion: Optional[str] = None
    recommended_duration: Optional[float] = None
    model_recommendation: Optional[str] = None
    geometry_confidence: Optional[float] = None
    motion_affordance: Optional[str] = None
    decision_metrics: Dict[str, Any] = {}


class ShotPlanResponse(BaseModel):
    project_id: str
    job_id: Optional[int] = None
    planner_version: str = "v2.0"
    runtime_provenance: Dict[str, Any] = {}
    target_length_seconds: Optional[float] = None
    target_group_budget: Optional[List[int]] = None
    sequence_edges: List[Dict[str, Any]] = []
    ordered_shots: List[Dict[str, Any]] = []


class SceneDebugResponse(BaseModel):
    project_id: str
    job_id: Optional[int] = None
    components: List[SceneComponentSummary]
    photo_geometries: List[PhotoGeometryDebug]
    motion_decisions: List[MotionDecisionDebug]
    shot_plan: Optional[ShotPlanResponse] = None


class PhotoRelationDebugRequest(BaseModel):
    left_photo_id: int
    right_photo_id: int
    job_id: Optional[int] = None


class PhotoRelationDebugResponse(BaseModel):
    project_id: str
    job_id: int
    relation: Optional[PhotoRelationDebug] = None
    left_photo: PhotoGeometryDebug
    right_photo: PhotoGeometryDebug
