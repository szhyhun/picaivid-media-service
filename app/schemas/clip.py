"""Pydantic schemas for clip responses."""
from typing import List, Optional, Any, Dict, Literal
from pydantic import BaseModel, Field


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
    from_left_25_50_score: Optional[float] = None
    from_right_50_75_score: Optional[float] = None
    to_left_25_50_score: Optional[float] = None
    to_right_50_75_score: Optional[float] = None
    cross_left_to_right_score: Optional[float] = None
    cross_right_to_left_score: Optional[float] = None
    cross_center_to_center_score: Optional[float] = None
    kornia_overlap_ratio: Optional[float] = None
    kornia_side_overlap: Optional[float] = None
    kornia_center_overlap: Optional[float] = None
    kornia_inlier_ratio: Optional[float] = None
    kornia_transition_overlap_ok: Optional[bool] = None


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
    from_left_25_50_score: Optional[float] = None
    from_right_50_75_score: Optional[float] = None
    to_left_25_50_score: Optional[float] = None
    to_right_50_75_score: Optional[float] = None
    cross_left_to_right_score: Optional[float] = None
    cross_right_to_left_score: Optional[float] = None
    cross_center_to_center_score: Optional[float] = None
    kornia_overlap_ratio: Optional[float] = None
    kornia_side_overlap: Optional[float] = None
    kornia_center_overlap: Optional[float] = None
    kornia_inlier_ratio: Optional[float] = None
    kornia_transition_overlap_ok: Optional[bool] = None


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


class PairDebugRequest(BaseModel):
    left_photo_id: int
    right_photo_id: int
    job_id: Optional[int] = None
    sample_limit: int = 250
    confidence_threshold: Optional[float] = Field(default=None, ge=0.1, le=1.0)
    matcher: Literal[
        "current",
        "loftr_kornia_indoor_native",
    ] = "current"


class PairDebugPhotoInfo(BaseModel):
    id: int
    rails_photo_id: str
    filename: Optional[str] = None
    room_label: Optional[str] = None
    position: Optional[int] = None
    s3_uri: Optional[str] = None
    image_url: Optional[str] = None
    thumbnail_url: Optional[str] = None


class PairDebugPoint(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float
    dx: float
    dy: float


class PairDebugStoredMetrics(BaseModel):
    pair_source: Optional[str] = None
    dinov2_similarity: Optional[float] = None
    geometric_matches: Optional[int] = None
    geometric_inliers: Optional[int] = None
    geometric_score: Optional[float] = None
    direction_dx: Optional[float] = None
    direction_dy: Optional[float] = None
    cross_left_to_right: Optional[float] = None
    cross_right_to_left: Optional[float] = None
    cross_center_to_center: Optional[float] = None
    kornia_overlap_ratio: Optional[float] = None
    kornia_side_overlap: Optional[float] = None
    kornia_center_overlap: Optional[float] = None
    kornia_inlier_ratio: Optional[float] = None
    kornia_transition_overlap_ok: Optional[bool] = None
    is_connected: Optional[bool] = None


class PairDebugLiveMetrics(BaseModel):
    matcher: Optional[str] = None
    checkpoint: Optional[str] = None
    confidence_threshold: Optional[float] = None
    geometry_model: Optional[str] = None
    raw_correspondence_count: Optional[int] = None
    raw_matches: List[PairDebugPoint] = []
    threshold_trials: List[Dict[str, Any]] = []
    loftr_input_width: Optional[int] = None
    loftr_input_height: Optional[int] = None
    ransac_reproj_threshold: Optional[float] = None
    num_matches: int = 0
    threshold_match_count: int = 0
    active_match_count: int = 0
    num_inliers: int = 0
    geometric_score: float = 0.0
    direction_dx: Optional[float] = None
    direction_dy: Optional[float] = None
    match_width: Optional[int] = None
    match_height: Optional[int] = None
    segment_scores: Dict[str, float] = {}
    score_components: Dict[str, float] = {}
    timing: Dict[str, Any] = {}
    oracle: Dict[str, Any] = {}
    native_matching_scores: Dict[str, float] = {}
    native_matching_scores_raw: Dict[str, float] = {}
    zju_variant: Optional[str] = None
    zju_loader: Optional[str] = None
    zju_checkpoint_path: Optional[str] = None
    zju_repo_dir: Optional[str] = None
    zju_match_type: Optional[str] = None
    zju_model_class: Optional[str] = None
    hf_visualization_data_url: Optional[str] = None
    inlier_match_count: int = 0
    inlier_matches: List[PairDebugPoint] = []


class PairDebugResponse(BaseModel):
    project_id: str
    job_id: int
    left_photo: PairDebugPhotoInfo
    right_photo: PairDebugPhotoInfo
    stored_metrics: Optional[PairDebugStoredMetrics] = None
    live_metrics: PairDebugLiveMetrics
