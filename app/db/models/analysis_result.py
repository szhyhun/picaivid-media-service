"""AnalysisResult model for Phase 1 analysis outputs."""
from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base


class AnalysisResult(Base):
    """Analysis result table - stores Phase 1 analysis outputs per room cluster."""

    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    room_cluster_id = Column(Integer, ForeignKey("room_clusters.id", ondelete="CASCADE"), nullable=True)

    # Motion recommendations
    recommended_motion = Column(String(50))
    allowed_motion_types = Column(JSONB)  # Array of allowed motions
    recommended_duration = Column(Float)
    tier = Column(String(20))  # low, medium, high
    model_recommendation = Column(String(100))

    # Debug metrics
    debug_metrics = Column(JSONB, default={})

    # Relationships
    room_cluster = relationship("RoomCluster", back_populates="analysis_results")

    def __repr__(self) -> str:
        return f"<AnalysisResult id={self.id} tier={self.tier} motion={self.recommended_motion}>"
