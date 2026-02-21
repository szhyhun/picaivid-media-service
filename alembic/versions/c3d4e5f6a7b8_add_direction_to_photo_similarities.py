"""add_direction_to_photo_similarities

Revision ID: c3d4e5f6a7b8
Revises: b8e5f03d9c11
Create Date: 2026-02-20

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, None] = "b8e5f03d9c11"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("photo_similarities", sa.Column("direction_dx", sa.Float(), nullable=True))
    op.add_column("photo_similarities", sa.Column("direction_dy", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("photo_similarities", "direction_dy")
    op.drop_column("photo_similarities", "direction_dx")
