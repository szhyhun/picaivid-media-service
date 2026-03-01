"""add_overlap_metrics_to_photo_similarities

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-02-26

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("photo_similarities", sa.Column("from_left_25_50", sa.Float(), nullable=True))
    op.add_column("photo_similarities", sa.Column("from_right_50_75", sa.Float(), nullable=True))
    op.add_column("photo_similarities", sa.Column("to_left_25_50", sa.Float(), nullable=True))
    op.add_column("photo_similarities", sa.Column("to_right_50_75", sa.Float(), nullable=True))
    op.add_column("photo_similarities", sa.Column("cross_left_to_right", sa.Float(), nullable=True))
    op.add_column("photo_similarities", sa.Column("cross_right_to_left", sa.Float(), nullable=True))
    op.add_column("photo_similarities", sa.Column("cross_center_to_center", sa.Float(), nullable=True))
    op.add_column("photo_similarities", sa.Column("kornia_overlap_ratio", sa.Float(), nullable=True))
    op.add_column("photo_similarities", sa.Column("kornia_side_overlap", sa.Float(), nullable=True))
    op.add_column("photo_similarities", sa.Column("kornia_center_overlap", sa.Float(), nullable=True))
    op.add_column("photo_similarities", sa.Column("kornia_inlier_ratio", sa.Float(), nullable=True))
    op.add_column("photo_similarities", sa.Column("kornia_transition_overlap_ok", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("photo_similarities", "kornia_transition_overlap_ok")
    op.drop_column("photo_similarities", "kornia_inlier_ratio")
    op.drop_column("photo_similarities", "kornia_center_overlap")
    op.drop_column("photo_similarities", "kornia_side_overlap")
    op.drop_column("photo_similarities", "kornia_overlap_ratio")
    op.drop_column("photo_similarities", "cross_center_to_center")
    op.drop_column("photo_similarities", "cross_right_to_left")
    op.drop_column("photo_similarities", "cross_left_to_right")
    op.drop_column("photo_similarities", "to_right_50_75")
    op.drop_column("photo_similarities", "to_left_25_50")
    op.drop_column("photo_similarities", "from_right_50_75")
    op.drop_column("photo_similarities", "from_left_25_50")
