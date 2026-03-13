"""PhotoSimilarity model - stores certified pair metrics between photos."""
from sqlalchemy import Column, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db.base import Base


class PhotoSimilarity(Base):
    """Stores precision-first certified pair metrics for debugging and planning."""

    __tablename__ = "photo_similarities"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)

    # Photo pair (always stored with photo_a_id < photo_b_id for uniqueness)
    photo_a_id = Column(Integer, ForeignKey("job_photos.id", ondelete="CASCADE"), nullable=False, index=True)
    photo_b_id = Column(Integer, ForeignKey("job_photos.id", ondelete="CASCADE"), nullable=False, index=True)

    # How the pair was proposed
    pair_source = Column(String(50))  # "dinov2_topk", "temporal_window", "both"

    # DINOv2 semantic similarity (0.0 to 1.0, higher = more similar)
    dinov2_similarity = Column(Float)

    # Precision-first pair verification metrics
    raw_matches = Column(Integer)
    f_inliers = Column(Integer)
    f_inlier_ratio = Column(Float)
    coverage_4x4 = Column(Float)
    grid_entropy = Column(Float)
    overlap_ratio = Column(Float)
    homography_ratio = Column(Float)
    median_epipolar_error = Column(Float)
    median_flow_magnitude = Column(Float)
    combined_geometry_score = Column(Float)
    near_positive_ratio = Column(Float)
    near_negative_ratio = Column(Float)
    split_score = Column(Float)
    depth_monotonicity_score = Column(Float)
    dominant_foreground_side_a = Column(Integer)
    dominant_foreground_side_b = Column(Integer)
    foreground_support_persistence_penalty = Column(Float)
    crossing_penalty = Column(Float)
    order_proximity = Column(Float)
    pair_rank = Column(Float)
    certification_status = Column(String(20))
    rejection_reason = Column(String(100))
    direction_dx = Column(Float)
    direction_dy = Column(Float)

    # Whether this pair was used in final clustering
    is_connected = Column(Integer, default=0)  # 0=no, 1=yes

    # Relationships
    photo_a = relationship("JobPhoto", foreign_keys=[photo_a_id])
    photo_b = relationship("JobPhoto", foreign_keys=[photo_b_id])

    __table_args__ = (
        UniqueConstraint('job_id', 'photo_a_id', 'photo_b_id', name='uq_photo_pair'),
    )

    def __repr__(self) -> str:
        return f"<PhotoSimilarity {self.photo_a_id}<->{self.photo_b_id} dinov2={self.dinov2_similarity:.3f}>"

    # Compatibility accessors for old call sites that still expect legacy names.
    @property
    def geometric_matches(self) -> int | None:
        return self.raw_matches

    @property
    def geometric_inliers(self) -> int | None:
        return self.f_inliers

    @property
    def geometric_score(self) -> float | None:
        return self.pair_rank

    @property
    def from_left_25_50(self) -> None:
        return None

    @property
    def from_right_50_75(self) -> None:
        return None

    @property
    def to_left_25_50(self) -> None:
        return None

    @property
    def to_right_50_75(self) -> None:
        return None

    @property
    def cross_left_to_right(self) -> None:
        return None

    @property
    def cross_right_to_left(self) -> None:
        return None

    @property
    def cross_center_to_center(self) -> None:
        return None

    @property
    def kornia_overlap_ratio(self) -> float | None:
        return self.overlap_ratio

    @property
    def kornia_side_overlap(self) -> None:
        return None

    @property
    def kornia_center_overlap(self) -> None:
        return None

    @property
    def kornia_inlier_ratio(self) -> float | None:
        return self.f_inlier_ratio

    @property
    def kornia_transition_overlap_ok(self) -> bool | None:
        return bool(self.is_connected) if self.is_connected is not None else None
