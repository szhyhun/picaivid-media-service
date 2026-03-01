"""PhotoSimilarity model - stores pairwise similarity scores between photos."""
from sqlalchemy import Column, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db.base import Base


class PhotoSimilarity(Base):
    """Stores pairwise similarity scores between photos for debugging.

    Records both DINOv2 semantic similarity and geometric verification results.
    Useful for understanding why photos are/aren't clustered together.
    """

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

    # Geometric verification results
    geometric_matches = Column(Integer)  # Total feature matches
    geometric_inliers = Column(Integer)  # Matches passing RANSAC
    geometric_score = Column(Float)  # Overlap score (0.0 to 1.0)
    direction_dx = Column(Float)  # Content shift X from photo_a -> photo_b (normalized)
    direction_dy = Column(Float)  # Content shift Y from photo_a -> photo_b (normalized)
    from_left_25_50 = Column(Float)
    from_right_50_75 = Column(Float)
    to_left_25_50 = Column(Float)
    to_right_50_75 = Column(Float)
    cross_left_to_right = Column(Float)
    cross_right_to_left = Column(Float)
    cross_center_to_center = Column(Float)
    kornia_overlap_ratio = Column(Float)
    kornia_side_overlap = Column(Float)
    kornia_center_overlap = Column(Float)
    kornia_inlier_ratio = Column(Float)
    kornia_transition_overlap_ok = Column(Integer)

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
