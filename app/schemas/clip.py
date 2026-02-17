"""Pydantic schemas for clip responses."""
from typing import List, Optional, Any, Dict
from pydantic import BaseModel


class SourcePhotoInfo(BaseModel):
    """Info about a source photo used in a clip."""
    id: int
    rails_photo_id: str
    s3_uri: Optional[str] = None
    thumbnail_url: Optional[str] = None
    room_label: Optional[str] = None
    final_score: Optional[float] = None
    depth_variance: Optional[float] = None
    sharpness: Optional[float] = None


class ClusterInfo(BaseModel):
    """Info about the room cluster for a clip."""
    id: int
    room_type: Optional[str] = None
    confidence_tier: Optional[str] = None
    image_count: int = 0
    overlap_score: Optional[float] = None
    depth_variance: Optional[float] = None
    recommended_motion: Optional[str] = None


class AnalysisInfo(BaseModel):
    """Analysis result info for a clip."""
    model_config = {"protected_namespaces": ()}

    tier: Optional[str] = None
    recommended_motion: Optional[str] = None
    model_recommendation: Optional[str] = None
    cfg_scale: Optional[float] = None
    inference_steps: Optional[int] = None
    debug_metrics: Optional[Dict[str, Any]] = None


class ClipResponse(BaseModel):
    """Single clip response for API."""
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
    video_url: Optional[str] = None  # Pre-signed URL for frontend
    status: str = "pending"

    # Extended info for debugging
    source_photos: Optional[List[SourcePhotoInfo]] = None
    cluster_info: Optional[ClusterInfo] = None
    analysis_info: Optional[AnalysisInfo] = None


class ClipListResponse(BaseModel):
    """List of clips response for API."""
    project_id: str
    job_id: Optional[int] = None
    job_status: Optional[str] = None
    clips: List[ClipResponse]
    total_clips: int
