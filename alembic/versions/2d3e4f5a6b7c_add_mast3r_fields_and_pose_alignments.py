"""historical geometry fields and pose alignments

Revision ID: 2d3e4f5a6b7c
Revises: 1c2d3e4f5a6b
Create Date: 2026-03-30 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "2d3e4f5a6b7c"
down_revision = "1c2d3e4f5a6b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("photo_similarities", sa.Column("match_engine", sa.String(length=50), nullable=True))
    op.add_column("photo_similarities", sa.Column("retrieval_score", sa.Float(), nullable=True))
    op.add_column("photo_similarities", sa.Column("reciprocal_match_count", sa.Integer(), nullable=True))
    op.add_column("photo_similarities", sa.Column("pointmap_consistency", sa.Float(), nullable=True))
    op.add_column("photo_similarities", sa.Column("alignment_residual", sa.Float(), nullable=True))
    op.add_column("photo_similarities", sa.Column("reprojection_error", sa.Float(), nullable=True))
    op.add_column("photo_similarities", sa.Column("parallax_score", sa.Float(), nullable=True))
    op.add_column("photo_similarities", sa.Column("graph_component_id", sa.Integer(), nullable=True))
    op.add_column("photo_similarities", sa.Column("graph_edge_score", sa.Float(), nullable=True))
    op.create_index(op.f("ix_photo_similarities_graph_component_id"), "photo_similarities", ["graph_component_id"], unique=False)

    op.create_table(
        "photo_pose_alignments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("photo_id", sa.Integer(), nullable=False),
        sa.Column("graph_component_id", sa.Integer(), nullable=False),
        sa.Column("pose_confidence", sa.Float(), nullable=True),
        sa.Column("reprojection_error", sa.Float(), nullable=True),
        sa.Column("focal_length", sa.Float(), nullable=True),
        sa.Column("principal_point", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("camera_center", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("camera_pose", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["photo_id"], ["job_photos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id", "photo_id", name="uq_photo_pose_alignment_job_photo"),
    )
    op.create_index(op.f("ix_photo_pose_alignments_job_id"), "photo_pose_alignments", ["job_id"], unique=False)
    op.create_index(op.f("ix_photo_pose_alignments_photo_id"), "photo_pose_alignments", ["photo_id"], unique=False)
    op.create_index(op.f("ix_photo_pose_alignments_graph_component_id"), "photo_pose_alignments", ["graph_component_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_photo_pose_alignments_graph_component_id"), table_name="photo_pose_alignments")
    op.drop_index(op.f("ix_photo_pose_alignments_photo_id"), table_name="photo_pose_alignments")
    op.drop_index(op.f("ix_photo_pose_alignments_job_id"), table_name="photo_pose_alignments")
    op.drop_table("photo_pose_alignments")

    op.drop_index(op.f("ix_photo_similarities_graph_component_id"), table_name="photo_similarities")
    op.drop_column("photo_similarities", "graph_edge_score")
    op.drop_column("photo_similarities", "graph_component_id")
    op.drop_column("photo_similarities", "parallax_score")
    op.drop_column("photo_similarities", "reprojection_error")
    op.drop_column("photo_similarities", "alignment_residual")
    op.drop_column("photo_similarities", "pointmap_consistency")
    op.drop_column("photo_similarities", "reciprocal_match_count")
    op.drop_column("photo_similarities", "retrieval_score")
    op.drop_column("photo_similarities", "match_engine")
