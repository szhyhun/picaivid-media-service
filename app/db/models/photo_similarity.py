"""PhotoSimilarity model - stores MASt3R graph edge metrics between photos."""
from sqlalchemy import Column, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db.base import Base


class PhotoSimilarity(Base):
    """Stores MASt3R edge metrics for debugging and planning."""

    __tablename__ = "photo_similarities"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)

    photo_a_id = Column(Integer, ForeignKey("job_photos.id", ondelete="CASCADE"), nullable=False, index=True)
    photo_b_id = Column(Integer, ForeignKey("job_photos.id", ondelete="CASCADE"), nullable=False, index=True)

    pair_source = Column(String(50))
    match_engine = Column(String(50))
    retrieval_score = Column(Float)
    reciprocal_match_count = Column(Integer)
    pointmap_consistency = Column(Float)
    alignment_residual = Column(Float)
    reprojection_error = Column(Float)
    parallax_score = Column(Float)
    graph_component_id = Column(Integer, index=True)
    graph_edge_score = Column(Float)
    overlap_ratio = Column(Float)
    combined_geometry_score = Column(Float)
    order_proximity = Column(Float)
    pair_rank = Column(Float)
    certification_status = Column(String(20))
    rejection_reason = Column(String(100))
    direction_dx = Column(Float)
    direction_dy = Column(Float)
    is_connected = Column(Integer, default=0)

    photo_a = relationship("JobPhoto", foreign_keys=[photo_a_id])
    photo_b = relationship("JobPhoto", foreign_keys=[photo_b_id])

    __table_args__ = (
        UniqueConstraint("job_id", "photo_a_id", "photo_b_id", name="uq_photo_pair"),
    )

    def __repr__(self) -> str:
        return (
            f"<PhotoSimilarity {self.photo_a_id}<->{self.photo_b_id} "
            f"engine={self.match_engine} score={self.graph_edge_score}>"
        )
