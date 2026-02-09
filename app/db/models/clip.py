"""Clip model for rendered video clips."""
from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base


class Clip(Base):
    """Clip table - stores rendered video clips from Phase 2."""

    __tablename__ = "clips"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    room_cluster_id = Column(Integer, ForeignKey("room_clusters.id"), nullable=True)
    source_photo_ids = Column(JSONB)  # Array of photo IDs
    motion_type = Column(String(50))
    model_used = Column(String(100))
    is_3d = Column(Boolean, default=False)
    duration = Column(Float)
    s3_uri = Column(String(500))
    validation_score = Column(Float)
    status = Column(String(50), default="pending")

    # Relationships
    job = relationship("Job", back_populates="clips")
    room_cluster = relationship("RoomCluster", back_populates="clips")
    timeline_clips = relationship("TimelineClip", back_populates="clip")

    def __repr__(self) -> str:
        return f"<Clip id={self.id} motion={self.motion_type} status={self.status}>"
