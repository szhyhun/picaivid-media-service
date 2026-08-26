"""Human-labeled ground truth for scene-graph evaluation.

One row per project (a project == a listing). Photo membership is stored as
`job_photos.rails_photo_id` (the stable Rails photo UUID), never as
`job_photos.id`: analysis re-runs create a new job whose job_photo ids differ
for the very same photos, and labeling is expensive human work that must
survive those re-runs.
"""
from sqlalchemy import Column, DateTime, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB

from app.db.base import Base

TRUTH_SPLITS = ("calibration", "holdout")
TRUTH_STATUSES = ("draft", "complete")


class SceneTruthSet(Base):
    __tablename__ = "scene_truth_sets"
    __table_args__ = (UniqueConstraint("project_id", name="uq_scene_truth_project"),)

    id = Column(Integer, primary_key=True)
    project_id = Column(String(36), nullable=False, index=True)
    last_job_id = Column(Integer, nullable=True)
    listing_slug = Column(String(120), nullable=False, default="", index=True)
    split = Column(String(16), nullable=False, default="calibration")
    status = Column(String(16), nullable=False, default="draft")

    # [{"instance": "bedroom-a", "photo_keys": ["<rails_photo_id>", ...]}]
    room_instances = Column(JSONB, nullable=False, default=list)
    # [["living-main", "kitchen-main"]] - instance names forming one open space
    open_plan_groups = Column(JSONB, nullable=False, default=list)
    # pair lists of rails_photo_id strings
    duplicates = Column(JSONB, nullable=False, default=list)
    must_not_group = Column(JSONB, nullable=False, default=list)
    preferred_pairs = Column(JSONB, nullable=False, default=list)
    # pairs in DIFFERENT rooms that still make a great transition (doorways,
    # stairs, sightlines). Feeds the story graph; never merges components.
    story_bridges = Column(JSONB, nullable=False, default=list)

    notes = Column(Text, nullable=False, default="")
    labeled_by = Column(String(120), nullable=False, default="")
    photo_count = Column(Integer, nullable=False, default=0)
    labeled_count = Column(Integer, nullable=False, default=0)
    revision = Column(Integer, nullable=False, default=1)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
