"""key scene truth by project + stable rails_photo_id

Ground truth must survive re-analysis: job_photos.id is reassigned on every
ingest (the same photo is 1207 in job 28 and 1263 in job 29), so labels are
stored against jobs.project_id + job_photos.rails_photo_id instead.

Revision ID: 5b6c7d8e9f01
Revises: 4a5b6c7d8e9f
Create Date: 2026-08-24

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "5b6c7d8e9f01"
down_revision: Union[str, None] = "4a5b6c7d8e9f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table_name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(table_name)


def upgrade() -> None:
    # The previous table only ever held smoke-test rows keyed by job_photos.id,
    # which cannot be migrated forward reliably; recreate it.
    if _table_exists("scene_truth_sets"):
        op.drop_table("scene_truth_sets")
    op.create_table(
        "scene_truth_sets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("last_job_id", sa.Integer(), nullable=True),
        sa.Column("listing_slug", sa.String(length=120), nullable=False, server_default=""),
        sa.Column("split", sa.String(length=16), nullable=False, server_default="calibration"),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="draft"),
        # room_instances: [{"instance": "bedroom-a", "photo_keys": ["<rails_photo_id>", ...]}]
        sa.Column("room_instances", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("open_plan_groups", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("duplicates", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("must_not_group", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("preferred_pairs", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("labeled_by", sa.String(length=120), nullable=False, server_default=""),
        sa.Column("photo_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("labeled_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", name="uq_scene_truth_project"),
    )
    op.create_index("ix_scene_truth_sets_project_id", "scene_truth_sets", ["project_id"])
    op.create_index("ix_scene_truth_sets_listing_slug", "scene_truth_sets", ["listing_slug"])


def downgrade() -> None:
    if _table_exists("scene_truth_sets"):
        op.drop_index("ix_scene_truth_sets_listing_slug", table_name="scene_truth_sets")
        op.drop_index("ix_scene_truth_sets_project_id", table_name="scene_truth_sets")
        op.drop_table("scene_truth_sets")
