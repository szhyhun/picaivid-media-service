"""Pydantic schemas for clip responses."""
from typing import List, Optional, Any, Dict
from pydantic import BaseModel


class SourcePhotoInfo(BaseModel):
    """Info about a source photo used in a clip."""
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


class ClusterInfo(BaseModel):
    """Info about the room cluster for a clip."""
    id: int
    room_type: Optional[str] = None
    confidence_tier: Optional[str] = None
    image_count: int = 0
    overlap_score: Optional[float] = None
    depth_variance: Optional[float] = None
    recommended_motion: Optional[str] = None
    sequence_order: Optional[int] = None


class ClipTransitionStep(BaseModel):
    """Directional recommendation between consecutive photos in a clip."""
    from_photo_id: int
    to_photo_id: int
    from_filename: Optional[str] = None
    to_filename: Optional[str] = None
    dinov2_similarity: Optional[float] = None
    geometric_inliers: Optional[int] = None
    geometric_score: Optional[float] = None
    direction_dx: Optional[float] = None
    direction_dy: Optional[float] = None
    recommendation: Optional[str] = None
    pair_source: Optional[str] = None
    is_connected: bool = False
    geometric_verified: bool = False
    overlap_from_zone: Optional[str] = None
    overlap_to_zone: Optional[str] = None
    overlap_summary: Optional[str] = None


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
    transition_steps: Optional[List[ClipTransitionStep]] = None
    cluster_info: Optional[ClusterInfo] = None
    analysis_info: Optional[AnalysisInfo] = None


class ClipListResponse(BaseModel):
    """List of clips response for API."""
    project_id: str
    job_id: Optional[int] = None
    job_status: Optional[str] = None
    clips: List[ClipResponse]
    total_clips: int


# Debug schemas for photo similarity and clustering analysis
class PhotoSimilarityInfo(BaseModel):
    """Similarity info between two photos."""
    photo_a_id: int
    photo_b_id: int
    photo_a_filename: Optional[str] = None
    photo_b_filename: Optional[str] = None
    pair_source: Optional[str] = None  # dinov2_topk, temporal_window, both
    dinov2_similarity: Optional[float] = None
    geometric_matches: Optional[int] = None
    geometric_inliers: Optional[int] = None
    geometric_score: Optional[float] = None
    direction_dx: Optional[float] = None
    direction_dy: Optional[float] = None
    is_connected: bool = False
    geometric_verified: bool = False
    overlap_from_zone: Optional[str] = None
    overlap_to_zone: Optional[str] = None
    overlap_summary: Optional[str] = None


class PhotoDebugInfo(BaseModel):
    """Debug info for a single photo in a cluster."""
    id: int
    rails_photo_id: str
    filename: Optional[str] = None
    thumbnail_url: Optional[str] = None
    room_label: Optional[str] = None
    position_in_cluster: int  # 0-indexed position
    is_endpoint: bool = False  # True if first or last in cluster
    is_duplicate: bool = False
    duplicate_of_photo_id: Optional[int] = None
    duplicate_of_filename: Optional[str] = None
    # Similarity to neighbors
    similarities: List[PhotoSimilarityInfo] = []


class ClusterDebugResponse(BaseModel):
    """Debug info for a cluster - shows why photos were grouped."""
    cluster_id: int
    room_type: Optional[str] = None
    photo_ids: List[int]  # Ordered list
    photo_filenames: List[str] = []  # Ordered list aligned with photo_ids
    photos: List[PhotoDebugInfo]
    total_photos: int
    sequence_order: Optional[int] = None
    # Summary stats
    avg_dinov2_similarity: Optional[float] = None
    avg_geometric_score: Optional[float] = None
    has_direction_info: bool = False


class ClusterListDebugResponse(BaseModel):
    """List of clusters with debug info."""
    project_id: str
    job_id: Optional[int] = None
    clusters: List[ClusterDebugResponse]
    total_clusters: int
