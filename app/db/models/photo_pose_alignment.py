"""Per-photo MASt3R alignment state for debug and cluster suggestions."""
from sqlalchemy import Column, Float, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base


class PhotoPoseAlignment(Base):
    __tablename__ = "photo_pose_alignments"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    photo_id = Column(Integer, ForeignKey("job_photos.id", ondelete="CASCADE"), nullable=False, index=True)
    graph_component_id = Column(Integer, nullable=False, index=True)
    pose_confidence = Column(Float)
    reprojection_error = Column(Float)
    focal_length = Column(Float)
    principal_point = Column(JSONB)
    camera_center = Column(JSONB)
    camera_pose = Column(JSONB)

    job = relationship("Job")
    photo = relationship("JobPhoto")

    __table_args__ = (
        UniqueConstraint("job_id", "photo_id", name="uq_photo_pose_alignment_job_photo"),
    )
