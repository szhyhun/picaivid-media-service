"""Scene geometry models produced by the VGGT phase-1 pipeline."""
from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base


class SceneComponent(Base):
    __tablename__ = "scene_components"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    scene_type = Column(String(32), nullable=False, default="interior")
    component_key = Column(String(64), nullable=False)
    photo_count = Column(Integer, default=0)
    geometry_confidence = Column(Float)
    connectivity_confidence = Column(Float)
    track_coverage = Column(Float)
    avg_reprojection_error = Column(Float)
    hero_photo_id = Column(Integer, ForeignKey("job_photos.id"), nullable=True)
    depth_range = Column(Float)
    motion_affordance = Column(String(32), nullable=True)
    sparse_scene_uri = Column(String(500), nullable=True)
    point_cloud_uri = Column(String(500), nullable=True)
    track_bundle_uri = Column(String(500), nullable=True)
    debug_metrics = Column(JSONB, default={})

    job = relationship("Job", back_populates="scene_components")
    hero_photo = relationship("JobPhoto", foreign_keys=[hero_photo_id], post_update=True)
    memberships = relationship(
        "SceneComponentMembership",
        back_populates="scene_component",
        cascade="all, delete-orphan",
    )
    relations = relationship(
        "PhotoRelation",
        back_populates="scene_component",
        cascade="all, delete-orphan",
    )
    geometries = relationship(
        "PhotoSceneGeometry",
        back_populates="scene_component",
        cascade="all, delete-orphan",
    )
    room_clusters = relationship("RoomCluster", back_populates="scene_component")

    __table_args__ = (
        UniqueConstraint("job_id", "component_key", name="uq_scene_component_job_key"),
    )


class SceneComponentMembership(Base):
    __tablename__ = "scene_component_memberships"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    photo_id = Column(Integer, ForeignKey("job_photos.id", ondelete="CASCADE"), nullable=False, index=True)
    scene_component_id = Column(
        Integer,
        ForeignKey("scene_components.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    order_index = Column(Integer, nullable=True)
    photo_role = Column(String(32), nullable=False, default="support")
    is_primary = Column(Boolean, nullable=False, default=True)

    scene_component = relationship("SceneComponent", back_populates="memberships")
    photo = relationship("JobPhoto")

    __table_args__ = (
        UniqueConstraint("job_id", "photo_id", name="uq_scene_component_membership_job_photo"),
    )


class PhotoSceneGeometry(Base):
    __tablename__ = "photo_scene_geometry"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    photo_id = Column(Integer, ForeignKey("job_photos.id", ondelete="CASCADE"), nullable=False, index=True)
    scene_component_id = Column(
        Integer,
        ForeignKey("scene_components.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    pose_confidence = Column(Float)
    depth_confidence = Column(Float)
    point_confidence = Column(Float)
    visibility_score = Column(Float)
    reprojection_error = Column(Float)
    camera_extrinsic = Column(JSONB)
    camera_intrinsic = Column(JSONB)
    camera_center = Column(JSONB)
    view_direction = Column(JSONB)
    depth_artifact_uri = Column(String(500), nullable=True)
    point_map_artifact_uri = Column(String(500), nullable=True)
    sparse_visibility = Column(JSONB, default={})
    local_metrics = Column(JSONB, default={})

    photo = relationship("JobPhoto")
    scene_component = relationship("SceneComponent", back_populates="geometries")

    __table_args__ = (
        UniqueConstraint("job_id", "photo_id", name="uq_photo_scene_geometry_job_photo"),
    )


class PhotoRelation(Base):
    __tablename__ = "photo_relations"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    photo_a_id = Column(Integer, ForeignKey("job_photos.id", ondelete="CASCADE"), nullable=False, index=True)
    photo_b_id = Column(Integer, ForeignKey("job_photos.id", ondelete="CASCADE"), nullable=False, index=True)
    scene_component_id = Column(
        Integer,
        ForeignKey("scene_components.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    overlap_score = Column(Float)
    track_support = Column(Float)
    reprojection_score = Column(Float)
    relation_confidence = Column(Float)
    baseline_distance = Column(Float)
    relative_transform = Column(JSONB)
    direction_dx = Column(Float)
    direction_dy = Column(Float)
    continuity_type = Column(String(32), nullable=False, default="weak")
    is_bridge_edge = Column(Boolean, nullable=False, default=False)
    is_connected = Column(Boolean, nullable=False, default=False)
    debug_metrics = Column(JSONB, default={})

    photo_a = relationship("JobPhoto", foreign_keys=[photo_a_id])
    photo_b = relationship("JobPhoto", foreign_keys=[photo_b_id])
    scene_component = relationship("SceneComponent", back_populates="relations")

    __table_args__ = (
        UniqueConstraint("job_id", "photo_a_id", "photo_b_id", name="uq_photo_relation_job_pair"),
    )
