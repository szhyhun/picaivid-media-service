"""add scene_truth_sets for human ground-truth labeling

Revision ID: 4a5b6c7d8e9f
Revises: 3f4e5d6c7b8a
Create Date: 2026-08-24

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "4a5b6c7d8e9f"
down_revision: Union[str, None] = "3f4e5d6c7b8a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table_name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(table_name)


def upgrade() -> None:
    if _table_exists("scene_truth_sets"):
        return
    op.create_table(
        "scene_truth_sets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("listing_slug", sa.String(length=120), nullable=False, server_default=""),
        sa.Column("split", sa.String(length=16), nullable=False, server_default="calibration"),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="draft"),
        sa.Column("room_instances", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("open_plan_groups", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("duplicates", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("must_not_group", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("preferred_pairs", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("labeled_by", sa.String(length=120), nullable=False, server_default=""),
        sa.Column("photo_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("labeled_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id", name="uq_scene_truth_job"),
    )
    op.create_index("ix_scene_truth_sets_job_id", "scene_truth_sets", ["job_id"])
    op.create_index("ix_scene_truth_sets_project_id", "scene_truth_sets", ["project_id"])


def downgrade() -> None:
    if _table_exists("scene_truth_sets"):
        op.drop_index("ix_scene_truth_sets_project_id", table_name="scene_truth_sets")
        op.drop_index("ix_scene_truth_sets_job_id", table_name="scene_truth_sets")
        op.drop_table("scene_truth_sets")
