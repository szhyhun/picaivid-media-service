"""Clip model for rendered video clips."""
from sqlalchemy import BigInteger, Boolean, Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base


class Clip(Base):
    """Clip table - stores rendered video clips from Phase 2.

    Model types:
    - LTX-2 single: Single image Ken Burns
    - LTX-2 interpolation: Two image interpolation
    - LTX-2 parallax: Multi-frame parallax
    - LTX-2 multi-view: Full multi-view synthesis
    - SEVA: Geometry-driven multi-view

    Status values:
    - pending: Waiting to be rendered
    - rendering: Currently being rendered
    - rendered: Successfully rendered at 720p
    - upscaling: Being upscaled to 4K
    - complete: Fully processed
    - failed: Rendering failed
    - downgraded: Re-rendered with safer motion
    """

    __tablename__ = "clips"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    room_cluster_id = Column(Integer, ForeignKey("room_clusters.id"), nullable=True)
    source_photo_ids = Column(JSONB)  # Array of photo IDs
    motion_type = Column(String(50))
    model_used = Column(String(100))  # LTX-2 single/interp/parallax/multi-view, SEVA
    is_3d = Column(Boolean, default=False)
    duration = Column(Float)

    # LTX-2 prompt and parameters used
    prompt_used = Column(Text)  # Actual prompt sent to LTX-2
    cfg_scale = Column(Float)
    inference_steps = Column(Integer)
    seed = Column(BigInteger)  # Fixed seed for reproducibility

    # S3 URIs
    s3_uri = Column(String(500))  # Original 720p render
    s3_uri_upscaled = Column(String(500))  # Upscaled 4K version

    # Validation and status
    validation_score = Column(Float)
    upscale_status = Column(String(50))  # pending, processing, complete, failed
    status = Column(String(50), default="pending")

    # Render metadata (timing, retries, etc.)
    render_metadata = Column(JSONB, default={})

    # Relationships
    job = relationship("Job", back_populates="clips")
    room_cluster = relationship("RoomCluster", back_populates="clips")
    timeline_clips = relationship("TimelineClip", back_populates="clip")

    def __repr__(self) -> str:
        return f"<Clip id={self.id} motion={self.motion_type} model={self.model_used} status={self.status}>"
