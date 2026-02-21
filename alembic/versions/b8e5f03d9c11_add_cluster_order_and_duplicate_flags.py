"""add_cluster_order_and_duplicate_flags

Revision ID: b8e5f03d9c11
Revises: a1b2c3d4e5f6
Create Date: 2026-02-20

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b8e5f03d9c11"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("room_clusters", sa.Column("sequence_order", sa.Integer(), nullable=True))

    op.add_column("job_photos", sa.Column("cluster_order", sa.Integer(), nullable=True))
    op.add_column(
        "job_photos",
        sa.Column("is_duplicate", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.add_column("job_photos", sa.Column("duplicate_of_photo_id", sa.Integer(), nullable=True))

    op.create_foreign_key(
        "fk_job_photos_duplicate_of_photo_id",
        "job_photos",
        "job_photos",
        ["duplicate_of_photo_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.create_index(
        "ix_room_clusters_job_id_sequence_order",
        "room_clusters",
        ["job_id", "sequence_order"],
    )
    op.create_index(
        "ix_job_photos_job_id_is_duplicate",
        "job_photos",
        ["job_id", "is_duplicate"],
    )


def downgrade() -> None:
    op.drop_index("ix_job_photos_job_id_is_duplicate", table_name="job_photos")
    op.drop_index("ix_room_clusters_job_id_sequence_order", table_name="room_clusters")

    op.drop_constraint("fk_job_photos_duplicate_of_photo_id", "job_photos", type_="foreignkey")

    op.drop_column("job_photos", "duplicate_of_photo_id")
    op.drop_column("job_photos", "is_duplicate")
    op.drop_column("job_photos", "cluster_order")

    op.drop_column("room_clusters", "sequence_order")
