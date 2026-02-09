"""Timeline model for video timeline data."""
from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base


class Timeline(Base):
    """Timeline table - stores timeline data from Phase 3."""

    __tablename__ = "timelines"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    version = Column(Integer, default=1)
    status = Column(String(50), default="draft")
    beat_grid = Column(JSONB)
    total_duration = Column(Float)

    # Relationships
    job = relationship("Job", back_populates="timelines")
    timeline_clips = relationship("TimelineClip", back_populates="timeline", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Timeline id={self.id} version={self.version} status={self.status}>"


class TimelineClip(Base):
    """TimelineClip table - links clips to timeline with ordering and timing."""

    __tablename__ = "timeline_clips"

    id = Column(Integer, primary_key=True)
    timeline_id = Column(Integer, ForeignKey("timelines.id", ondelete="CASCADE"), nullable=False, index=True)
    clip_id = Column(Integer, ForeignKey("clips.id"), nullable=False)
    order_index = Column(Integer, nullable=False)
    in_time = Column(Float)
    out_time = Column(Float)
    transition_type = Column(String(50))
    audio_policy = Column(String(50))

    # Relationships
    timeline = relationship("Timeline", back_populates="timeline_clips")
    clip = relationship("Clip", back_populates="timeline_clips")

    def __repr__(self) -> str:
        return f"<TimelineClip id={self.id} order={self.order_index}>"
