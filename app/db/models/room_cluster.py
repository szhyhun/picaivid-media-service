"""RoomCluster model for grouping photos by room."""
from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.db.base import Base


class RoomCluster(Base):
    """Room cluster table - groups photos by room type."""

    __tablename__ = "room_clusters"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    room_type = Column(String(100))
    confidence_tier = Column(String(20))  # low, medium, high
    sfm_eligible = Column(Boolean, default=False)
    image_count = Column(Integer, default=0)
    overlap_score = Column(Float)
    depth_variance = Column(Float)

    # Hero photo selection (references job_photos table, not Rails photos)
    hero_photo_id = Column(Integer, ForeignKey("job_photos.id"), nullable=True)

    # Motion recommendations
    recommended_motion = Column(String(50))
    allowed_motion_types = Column(String(200))  # comma-separated list
    recommended_duration = Column(Float)

    # Relationships
    job = relationship("Job", back_populates="room_clusters")
    photos = relationship("JobPhoto", back_populates="room_cluster", foreign_keys="JobPhoto.room_cluster_id")
    hero_photo = relationship("JobPhoto", foreign_keys=[hero_photo_id], post_update=True)
    analysis_results = relationship("AnalysisResult", back_populates="room_cluster", cascade="all, delete-orphan")
    clips = relationship("Clip", back_populates="room_cluster")

    def __repr__(self) -> str:
        return f"<RoomCluster id={self.id} room_type={self.room_type} tier={self.confidence_tier}>"
