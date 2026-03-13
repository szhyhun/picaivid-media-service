"""precision_first_transition_pipeline

Revision ID: f1a2b3c4d5e6
Revises: d4e5f6a7b8c9
Create Date: 2026-03-12

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return {column["name"] for column in inspector.get_columns(table_name)}


def _table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return inspector.has_table(table_name)


def _add_column_if_missing(table_name: str, column: sa.Column) -> None:
    if column.name not in _column_names(table_name):
        op.add_column(table_name, column)


def _drop_column_if_exists(table_name: str, column_name: str) -> None:
    if column_name in _column_names(table_name):
        op.drop_column(table_name, column_name)


def upgrade() -> None:
    existing_columns = _column_names("photo_similarities")

    _add_column_if_missing("photo_similarities", sa.Column("raw_matches", sa.Integer(), nullable=True))
    _add_column_if_missing("photo_similarities", sa.Column("f_inliers", sa.Integer(), nullable=True))
    _add_column_if_missing("photo_similarities", sa.Column("f_inlier_ratio", sa.Float(), nullable=True))
    _add_column_if_missing("photo_similarities", sa.Column("coverage_4x4", sa.Float(), nullable=True))
    _add_column_if_missing("photo_similarities", sa.Column("grid_entropy", sa.Float(), nullable=True))
    _add_column_if_missing("photo_similarities", sa.Column("overlap_ratio", sa.Float(), nullable=True))
    _add_column_if_missing("photo_similarities", sa.Column("homography_ratio", sa.Float(), nullable=True))
    _add_column_if_missing("photo_similarities", sa.Column("median_epipolar_error", sa.Float(), nullable=True))
    _add_column_if_missing("photo_similarities", sa.Column("median_flow_magnitude", sa.Float(), nullable=True))
    _add_column_if_missing("photo_similarities", sa.Column("combined_geometry_score", sa.Float(), nullable=True))
    _add_column_if_missing("photo_similarities", sa.Column("near_positive_ratio", sa.Float(), nullable=True))
    _add_column_if_missing("photo_similarities", sa.Column("near_negative_ratio", sa.Float(), nullable=True))
    _add_column_if_missing("photo_similarities", sa.Column("split_score", sa.Float(), nullable=True))
    _add_column_if_missing("photo_similarities", sa.Column("depth_monotonicity_score", sa.Float(), nullable=True))
    _add_column_if_missing("photo_similarities", sa.Column("dominant_foreground_side_a", sa.Integer(), nullable=True))
    _add_column_if_missing("photo_similarities", sa.Column("dominant_foreground_side_b", sa.Integer(), nullable=True))
    _add_column_if_missing("photo_similarities", sa.Column("foreground_support_persistence_penalty", sa.Float(), nullable=True))
    _add_column_if_missing("photo_similarities", sa.Column("crossing_penalty", sa.Float(), nullable=True))
    _add_column_if_missing("photo_similarities", sa.Column("order_proximity", sa.Float(), nullable=True))
    _add_column_if_missing("photo_similarities", sa.Column("pair_rank", sa.Float(), nullable=True))
    _add_column_if_missing("photo_similarities", sa.Column("certification_status", sa.String(length=20), nullable=True))
    _add_column_if_missing("photo_similarities", sa.Column("rejection_reason", sa.String(length=100), nullable=True))

    if not _table_exists("transition_sequences"):
        op.create_table(
            "transition_sequences",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("job_id", sa.Integer(), sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
            sa.Column("sequence_rank", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("sequence_score", sa.Float(), nullable=False, server_default="0"),
            sa.Column("certification_status", sa.String(length=20), nullable=False, server_default="usable"),
            sa.Column("room_type_hint", sa.String(length=100), nullable=True),
            sa.Column("source_cluster_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("motion_hint", sa.String(length=100), nullable=True),
        )
        op.create_index("ix_transition_sequences_job_id", "transition_sequences", ["job_id"])

    if not _table_exists("transition_sequence_steps"):
        op.create_table(
            "transition_sequence_steps",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "transition_sequence_id",
                sa.Integer(),
                sa.ForeignKey("transition_sequences.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("step_index", sa.Integer(), nullable=False),
            sa.Column("photo_id", sa.Integer(), sa.ForeignKey("job_photos.id", ondelete="CASCADE"), nullable=False),
            sa.Column(
                "photo_similarity_id",
                sa.Integer(),
                sa.ForeignKey("photo_similarities.id", ondelete="SET NULL"),
                nullable=True,
            ),
        )
        op.create_index("ix_transition_sequence_steps_transition_sequence_id", "transition_sequence_steps", ["transition_sequence_id"])

    if "geometric_matches" in existing_columns:
        op.execute("UPDATE photo_similarities SET raw_matches = geometric_matches WHERE raw_matches IS NULL")
    if "geometric_inliers" in existing_columns:
        op.execute("UPDATE photo_similarities SET f_inliers = geometric_inliers WHERE f_inliers IS NULL")
    if "geometric_score" in existing_columns:
        op.execute("UPDATE photo_similarities SET pair_rank = geometric_score WHERE pair_rank IS NULL")
        op.execute("UPDATE photo_similarities SET combined_geometry_score = geometric_score WHERE combined_geometry_score IS NULL")
    if "kornia_overlap_ratio" in existing_columns:
        op.execute("UPDATE photo_similarities SET overlap_ratio = kornia_overlap_ratio WHERE overlap_ratio IS NULL")
    if "kornia_inlier_ratio" in existing_columns:
        op.execute("UPDATE photo_similarities SET f_inlier_ratio = kornia_inlier_ratio WHERE f_inlier_ratio IS NULL")

    _drop_column_if_exists("photo_similarities", "from_left_25_50")
    _drop_column_if_exists("photo_similarities", "from_right_50_75")
    _drop_column_if_exists("photo_similarities", "to_left_25_50")
    _drop_column_if_exists("photo_similarities", "to_right_50_75")
    _drop_column_if_exists("photo_similarities", "cross_left_to_right")
    _drop_column_if_exists("photo_similarities", "cross_right_to_left")
    _drop_column_if_exists("photo_similarities", "cross_center_to_center")
    _drop_column_if_exists("photo_similarities", "kornia_overlap_ratio")
    _drop_column_if_exists("photo_similarities", "kornia_side_overlap")
    _drop_column_if_exists("photo_similarities", "kornia_center_overlap")
    _drop_column_if_exists("photo_similarities", "kornia_inlier_ratio")
    _drop_column_if_exists("photo_similarities", "kornia_transition_overlap_ok")
    _drop_column_if_exists("photo_similarities", "geometric_matches")
    _drop_column_if_exists("photo_similarities", "geometric_inliers")
    _drop_column_if_exists("photo_similarities", "geometric_score")


def downgrade() -> None:
    op.add_column("photo_similarities", sa.Column("geometric_matches", sa.Integer(), nullable=True))
    op.add_column("photo_similarities", sa.Column("geometric_inliers", sa.Integer(), nullable=True))
    op.add_column("photo_similarities", sa.Column("geometric_score", sa.Float(), nullable=True))
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

    op.execute("UPDATE photo_similarities SET geometric_matches = raw_matches WHERE geometric_matches IS NULL")
    op.execute("UPDATE photo_similarities SET geometric_inliers = f_inliers WHERE geometric_inliers IS NULL")
    op.execute("UPDATE photo_similarities SET geometric_score = pair_rank WHERE geometric_score IS NULL")
    op.execute("UPDATE photo_similarities SET kornia_overlap_ratio = overlap_ratio WHERE kornia_overlap_ratio IS NULL")
    op.execute("UPDATE photo_similarities SET kornia_inlier_ratio = f_inlier_ratio WHERE kornia_inlier_ratio IS NULL")

    op.drop_index("ix_transition_sequence_steps_transition_sequence_id", table_name="transition_sequence_steps")
    op.drop_table("transition_sequence_steps")
    op.drop_index("ix_transition_sequences_job_id", table_name="transition_sequences")
    op.drop_table("transition_sequences")

    op.drop_column("photo_similarities", "rejection_reason")
    op.drop_column("photo_similarities", "certification_status")
    op.drop_column("photo_similarities", "pair_rank")
    op.drop_column("photo_similarities", "order_proximity")
    op.drop_column("photo_similarities", "crossing_penalty")
    op.drop_column("photo_similarities", "foreground_support_persistence_penalty")
    op.drop_column("photo_similarities", "dominant_foreground_side_b")
    op.drop_column("photo_similarities", "dominant_foreground_side_a")
    op.drop_column("photo_similarities", "depth_monotonicity_score")
    op.drop_column("photo_similarities", "split_score")
    op.drop_column("photo_similarities", "near_negative_ratio")
    op.drop_column("photo_similarities", "near_positive_ratio")
    op.drop_column("photo_similarities", "combined_geometry_score")
    op.drop_column("photo_similarities", "median_flow_magnitude")
    op.drop_column("photo_similarities", "median_epipolar_error")
    op.drop_column("photo_similarities", "homography_ratio")
    op.drop_column("photo_similarities", "overlap_ratio")
    op.drop_column("photo_similarities", "grid_entropy")
    op.drop_column("photo_similarities", "coverage_4x4")
    op.drop_column("photo_similarities", "f_inlier_ratio")
    op.drop_column("photo_similarities", "f_inliers")
    op.drop_column("photo_similarities", "raw_matches")
