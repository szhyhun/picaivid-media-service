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
    geometric_inliers: Optional[int] = None
    geometric_score: Optional[float] = None
    pair_rank: Optional[float] = None
    certification_status: Optional[str] = None
    rejection_reason: Optional[str] = None
    overlap_ratio: Optional[float] = None
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


class TransitionSequenceStepInfo(BaseModel):
    step_index: int
    photo_id: int
    photo_similarity_id: Optional[int] = None


class TransitionSequenceInfo(BaseModel):
    sequence_rank: int
    sequence_score: float
    certification_status: str
    room_type_hint: Optional[str] = None
    source_cluster_ids: List[int] = []
    motion_hint: Optional[str] = None
    photo_ids: List[int] = []
    steps: List[TransitionSequenceStepInfo] = []


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
    pair_source: Optional[str] = None  # mast3r_retrieval_graph
    match_engine: Optional[str] = None
    retrieval_score: Optional[float] = None
    reciprocal_match_count: Optional[int] = None
    pointmap_consistency: Optional[float] = None
    alignment_residual: Optional[float] = None
    reprojection_error: Optional[float] = None
    parallax_score: Optional[float] = None
    graph_component_id: Optional[int] = None
    graph_edge_score: Optional[float] = None
    raw_matches: Optional[int] = None
    overlap_ratio: Optional[float] = None
    combined_geometry_score: Optional[float] = None
    effective_overlap: Optional[float] = None
    repeated_room_scope: Optional[str] = None
    repeated_room_penalty_weight: Optional[float] = None
    fixture_disagreement_count: Optional[int] = None
    fixture_instance_penalty: Optional[float] = None
    mirror_similarity: Optional[float] = None
    mirror_penalty: Optional[float] = None
    layout_similarity: Optional[float] = None
    layout_penalty: Optional[float] = None
    center_similarity: Optional[float] = None
    center_support_bonus: Optional[float] = None
    center_mismatch_penalty: Optional[float] = None
    anchor_region_name: Optional[str] = None
    anchor_similarity: Optional[float] = None
    anchor_match_density_a: Optional[float] = None
    anchor_match_density_b: Optional[float] = None
    anchor_support_score: Optional[float] = None
    anchor_support_bonus: Optional[float] = None
    anchor_mismatch_penalty: Optional[float] = None
    window_match_density_a: Optional[float] = None
    window_match_density_b: Optional[float] = None
    window_support_score: Optional[float] = None
    window_dominance_penalty: Optional[float] = None
    repeated_room_instance_penalty: Optional[float] = None
    order_proximity: Optional[float] = None
    pair_rank: Optional[float] = None
    certification_status: Optional[str] = None
    rejection_reason: Optional[str] = None
    direction_dx: Optional[float] = None
    direction_dy: Optional[float] = None
    is_connected: bool = False


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


class ClusterSuggestionInfo(BaseModel):
    photo_id: int
    filename: Optional[str] = None
    room_label: Optional[str] = None
    score: Optional[float] = None
    related_to_photo_id: Optional[int] = None
    reason: Optional[str] = None


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
    avg_pair_rank: Optional[float] = None
    has_direction_info: bool = False
    sequences: List[TransitionSequenceInfo] = []
    suggestions: List[ClusterSuggestionInfo] = []


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
    sample_limit: int = 1000
    confidence_threshold: Optional[float] = Field(default=None, ge=0.1, le=1.0)
    matcher: Literal[
        "current",
        "mast3r_graph",
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
    label_a: Optional[str] = None
    label_b: Optional[str] = None
    region_id_a: Optional[int] = None
    region_id_b: Optional[int] = None
    region_type_a: Optional[str] = None
    region_type_b: Optional[str] = None
    is_anchor_match: Optional[bool] = None
    is_window_match: Optional[bool] = None
    is_background_match: Optional[bool] = None
    is_object_match: Optional[bool] = None
    same_associated_object_match: Optional[bool] = None
    anchor_object_match: Optional[bool] = None
    cross_object_match: Optional[bool] = None
    association_score: Optional[float] = None
    semantic_accept: Optional[bool] = None


class SemanticRegionInfo(BaseModel):
    id: int
    label: str
    region_type: str
    bbox: List[float] = []
    polygon: List[List[float]] = []
    score: Optional[float] = None
    area_ratio: Optional[float] = None


class PairDebugStoredMetrics(BaseModel):
    pair_source: Optional[str] = None
    match_engine: Optional[str] = None
    semantic_backend: Optional[str] = None
    semantic_regions_available: Optional[bool] = None
    retrieval_score: Optional[float] = None
    reciprocal_match_count: Optional[int] = None
    pointmap_consistency: Optional[float] = None
    alignment_residual: Optional[float] = None
    reprojection_error: Optional[float] = None
    parallax_score: Optional[float] = None
    graph_component_id: Optional[int] = None
    graph_edge_score: Optional[float] = None
    raw_matches: Optional[int] = None
    overlap_ratio: Optional[float] = None
    combined_geometry_score: Optional[float] = None
    geometry_soft_penalty: Optional[float] = None
    effective_overlap: Optional[float] = None
    repeated_room_scope: Optional[str] = None
    repeated_room_penalty_weight: Optional[float] = None
    fixture_disagreement_count: Optional[int] = None
    fixture_instance_penalty: Optional[float] = None
    mirror_similarity: Optional[float] = None
    mirror_penalty: Optional[float] = None
    layout_similarity: Optional[float] = None
    layout_penalty: Optional[float] = None
    center_similarity: Optional[float] = None
    center_support_bonus: Optional[float] = None
    center_mismatch_penalty: Optional[float] = None
    anchor_region_name: Optional[str] = None
    anchor_label_a: Optional[str] = None
    anchor_label_b: Optional[str] = None
    anchor_similarity: Optional[float] = None
    object_similarity: Optional[float] = None
    anchor_match_density_a: Optional[float] = None
    anchor_match_density_b: Optional[float] = None
    anchor_support_score: Optional[float] = None
    anchor_support_bonus: Optional[float] = None
    anchor_inlier_ratio: Optional[float] = None
    object_match_ratio: Optional[float] = None
    background_match_ratio: Optional[float] = None
    window_match_ratio: Optional[float] = None
    same_anchor_label_ratio: Optional[float] = None
    cross_label_object_ratio: Optional[float] = None
    same_object_match_ratio: Optional[float] = None
    same_object_inlier_ratio: Optional[float] = None
    anchor_object_match_ratio: Optional[float] = None
    cross_object_ratio: Optional[float] = None
    semantic_accept_ratio: Optional[float] = None
    low_information_match_ratio: Optional[float] = None
    low_information_penalty: Optional[float] = None
    window_view_leak_penalty: Optional[float] = None
    object_support_signal: Optional[float] = None
    object_set_similarity: Optional[float] = None
    object_association_confidence: Optional[float] = None
    mean_association_score: Optional[float] = None
    matched_region_count: Optional[int] = None
    unmatched_anchor_count: Optional[int] = None
    unmatched_anchor_penalty: Optional[float] = None
    cross_object_penalty: Optional[float] = None
    anchor_mismatch_penalty: Optional[float] = None
    window_match_density_a: Optional[float] = None
    window_match_density_b: Optional[float] = None
    window_support_score: Optional[float] = None
    window_dominance_penalty: Optional[float] = None
    repeated_room_instance_penalty: Optional[float] = None
    semantic_match_counts: Optional[Dict[str, int]] = None
    associated_object_pairs: Optional[List[Dict[str, Any]]] = None
    best_anchor_pair: Optional[Dict[str, Any]] = None
    order_proximity: Optional[float] = None
    continuity_bonus: Optional[float] = None
    crossing_weight: Optional[float] = None
    pair_rank: Optional[float] = None
    certification_status: Optional[str] = None
    rejection_reason: Optional[str] = None
    direction_dx: Optional[float] = None
    direction_dy: Optional[float] = None
    is_connected: Optional[bool] = None


class PairDebugLiveMetrics(BaseModel):
    matcher: Optional[str] = None
    checkpoint: Optional[str] = None
    match_engine: Optional[str] = None
    semantic_backend: Optional[str] = None
    semantic_regions_available: Optional[bool] = None
    retrieval_score: Optional[float] = None
    reciprocal_match_count: Optional[int] = None
    pointmap_consistency: Optional[float] = None
    alignment_residual: Optional[float] = None
    reprojection_error: Optional[float] = None
    parallax_score: Optional[float] = None
    graph_component_id: Optional[int] = None
    graph_edge_score: Optional[float] = None
    confidence_threshold: Optional[float] = None
    geometry_model: Optional[str] = None
    raw_correspondence_count: Optional[int] = None
    raw_matches: List[PairDebugPoint] = []
    threshold_trials: List[Dict[str, Any]] = []
    ransac_reproj_threshold: Optional[float] = None
    num_matches: int = 0
    threshold_match_count: int = 0
    active_match_count: int = 0
    num_inliers: int = 0
    geometric_score: float = 0.0
    pair_rank: float = 0.0
    certification_status: Optional[str] = None
    rejection_reason: Optional[str] = None
    overlap_ratio: Optional[float] = None
    combined_geometry_score: Optional[float] = None
    geometry_soft_penalty: Optional[float] = None
    effective_overlap: Optional[float] = None
    repeated_room_scope: Optional[str] = None
    repeated_room_penalty_weight: Optional[float] = None
    fixture_disagreement_count: Optional[int] = None
    fixture_instance_penalty: Optional[float] = None
    mirror_similarity: Optional[float] = None
    mirror_penalty: Optional[float] = None
    layout_similarity: Optional[float] = None
    layout_penalty: Optional[float] = None
    center_similarity: Optional[float] = None
    center_support_bonus: Optional[float] = None
    center_mismatch_penalty: Optional[float] = None
    anchor_region_name: Optional[str] = None
    anchor_label_a: Optional[str] = None
    anchor_label_b: Optional[str] = None
    anchor_similarity: Optional[float] = None
    object_similarity: Optional[float] = None
    anchor_match_density_a: Optional[float] = None
    anchor_match_density_b: Optional[float] = None
    anchor_support_score: Optional[float] = None
    anchor_support_bonus: Optional[float] = None
    anchor_inlier_ratio: Optional[float] = None
    object_match_ratio: Optional[float] = None
    background_match_ratio: Optional[float] = None
    window_match_ratio: Optional[float] = None
    same_anchor_label_ratio: Optional[float] = None
    cross_label_object_ratio: Optional[float] = None
    same_object_match_ratio: Optional[float] = None
    same_object_inlier_ratio: Optional[float] = None
    anchor_object_match_ratio: Optional[float] = None
    cross_object_ratio: Optional[float] = None
    semantic_accept_ratio: Optional[float] = None
    low_information_match_ratio: Optional[float] = None
    low_information_penalty: Optional[float] = None
    window_view_leak_penalty: Optional[float] = None
    object_support_signal: Optional[float] = None
    object_set_similarity: Optional[float] = None
    object_association_confidence: Optional[float] = None
    mean_association_score: Optional[float] = None
    matched_region_count: Optional[int] = None
    unmatched_anchor_count: Optional[int] = None
    unmatched_anchor_penalty: Optional[float] = None
    cross_object_penalty: Optional[float] = None
    anchor_mismatch_penalty: Optional[float] = None
    window_match_density_a: Optional[float] = None
    window_match_density_b: Optional[float] = None
    window_support_score: Optional[float] = None
    window_dominance_penalty: Optional[float] = None
    repeated_room_instance_penalty: Optional[float] = None
    semantic_match_counts: Dict[str, int] = {}
    associated_object_pairs: List[Dict[str, Any]] = []
    best_anchor_pair: Optional[Dict[str, Any]] = None
    semantic_regions_left: List[SemanticRegionInfo] = []
    semantic_regions_right: List[SemanticRegionInfo] = []
    homography_penalty: Optional[float] = None
    low_flow_penalty: Optional[float] = None
    order_proximity: Optional[float] = None
    continuity_bonus: Optional[float] = None
    crossing_weight: Optional[float] = None
    motion_label: Optional[str] = None
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
    strict_gate: Dict[str, Any] = {}
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
