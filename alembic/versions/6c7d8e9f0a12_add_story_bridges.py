"""add story_bridges to scene truth

A great cinematic pair can span two *different* rooms that are physically well
connected (patio -> stairs to patio). That is editorial continuity, not room
membership, and the Scene-Graph V2 story graph needs it as ground truth.

Revision ID: 6c7d8e9f0a12
Revises: 5b6c7d8e9f01
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "6c7d8e9f0a12"
down_revision: Union[str, None] = "5b6c7d8e9f01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _columns() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("scene_truth_sets"):
        return set()
    return {column["name"] for column in inspector.get_columns("scene_truth_sets")}


def upgrade() -> None:
    if "story_bridges" not in _columns():
        op.add_column(
            "scene_truth_sets",
            sa.Column("story_bridges", postgresql.JSONB(), nullable=False, server_default="[]"),
        )


def downgrade() -> None:
    if "story_bridges" in _columns():
        op.drop_column("scene_truth_sets", "story_bridges")
