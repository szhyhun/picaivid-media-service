"""JobPhoto model - derived photo data owned by Python media service."""
from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base


class JobPhoto(Base):
    """JobPhoto table - stores derived photo analysis results.

    This is Python's copy of photo metadata plus computed fields.
    Original photo data lives in Rails' photos table (read-only).
    """

    __tablename__ = "job_photos"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    rails_photo_id = Column(String(36), nullable=False, index=True)  # UUID from Rails photos table
    s3_uri = Column(String(500), nullable=False)  # Copied from Rails for convenience
    filename = Column(String(255))
    width = Column(Integer)
    height = Column(Integer)
    position = Column(Integer, default=0)

    # Room classification
    room_label = Column(String(100))  # AI-detected room type
    room_override = Column(String(100))  # Copied from Rails (manual override)
    room_cluster_id = Column(Integer, ForeignKey("room_clusters.id"), nullable=True)

    # Flags
    exclude = Column(Boolean, default=False)

    # Manual metadata copied from Rails
    manual_metadata = Column(JSONB, default={})

    # Quality scores (computed in Phase 1)
    sharpness = Column(Float)
    exposure_score = Column(Float)
    composition_score = Column(Float)
    base_score = Column(Float)
    final_score = Column(Float)

    # Depth analysis (computed in Phase 1)
    depth_variance = Column(Float)
    depth_layers = Column(Integer)

    # Embedding (computed by OpenCLIP in Phase 1)
    embedding = Column(JSONB, nullable=True)

    # Relationships
    job = relationship("Job", back_populates="job_photos")
    room_cluster = relationship("RoomCluster", back_populates="photos")

    def __repr__(self) -> str:
        return f"<JobPhoto id={self.id} rails_id={self.rails_photo_id} room={self.room_label}>"
