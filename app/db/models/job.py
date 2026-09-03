"""Job model for video generation jobs."""
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base import Base


class Job(Base):
    """Job table - tracks video generation jobs from Rails."""

    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True)
    project_id = Column(String(36), nullable=False, index=True)  # UUID from Rails
    status = Column(String(50), default="pending", index=True)
    current_phase = Column(Integer, default=0)
    template_type = Column(String(50))
    target_length = Column(Float)
    music_uri = Column(String(500))
    # bpm and beat_offset are detected in Phase 3
    bpm = Column(Integer, nullable=True)
    beat_offset = Column(Float, nullable=True)
    enable_beat_sync = Column(Boolean, default=True)
    final_video_uri = Column(String(500), nullable=True)
    error_message = Column(String(1000), nullable=True)

    # Progress reporting. The pipeline already knows how far along it is (it logs
    # VGGT_V2_PAIR_PROGRESS every 25 pairs) but nothing persisted it, so the UI had
    # no way to show anything but a spinner.
    progress_current = Column(Integer, nullable=True)
    progress_total = Column(Integer, nullable=True)
    progress_label = Column(String(120), nullable=True)
    phase_started_at = Column(DateTime, nullable=True)
    # Refreshed on the same cadence as progress. A heartbeat older than
    # STALE_AFTER_SECONDS means the worker died mid-job; without this a crashed
    # run is indistinguishable from a slow one, which is how three separate
    # outages showed up as an endless spinner.
    heartbeat_at = Column(DateTime, nullable=True)

    # Cooperative cancellation: the API sets the flag, the worker checks it at
    # loop boundaries and unwinds cleanly. Never kill mid-write, or the partial
    # scene graph is exactly what gets left behind.
    cancel_requested = Column(Boolean, nullable=False, server_default="false")
    canceled_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    job_photos = relationship("JobPhoto", back_populates="job", cascade="all, delete-orphan")
    scene_components = relationship("SceneComponent", back_populates="job", cascade="all, delete-orphan")
    room_clusters = relationship("RoomCluster", back_populates="job", cascade="all, delete-orphan")
    clips = relationship("Clip", back_populates="job", cascade="all, delete-orphan")
    timelines = relationship("Timeline", back_populates="job", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Job id={self.id} project_id={self.project_id} status={self.status}>"
