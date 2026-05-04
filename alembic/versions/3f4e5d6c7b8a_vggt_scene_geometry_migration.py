"""vggt_scene_geometry_migration

Revision ID: 3f4e5d6c7b8a
Revises: 2d3e4f5a6b7c
Create Date: 2026-05-04

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "3f4e5d6c7b8a"
down_revision: Union[str, None] = "2d3e4f5a6b7c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return inspector.has_table(table_name)


def _column_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table(table_name):
        return set()
    return {column["name"] for column in inspector.get_columns(table_name)}


def _drop_table_if_exists(table_name: str) -> None:
    if _table_exists(table_name):
        op.drop_table(table_name)


def _drop_index_if_exists(index_name: str, table_name: str) -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    indexes = {index["name"] for index in inspector.get_indexes(table_name)} if inspector.has_table(table_name) else set()
    if index_name in indexes:
        op.drop_index(index_name, table_name=table_name)


def upgrade() -> None:
    if not _table_exists("scene_components"):
        op.create_table(
            "scene_components",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("job_id", sa.Integer(), sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
            sa.Column("scene_type", sa.String(length=32), nullable=False, server_default="interior"),
            sa.Column("component_key", sa.String(length=64), nullable=False),
            sa.Column("photo_count", sa.Integer(), nullable=True),
            sa.Column("geometry_confidence", sa.Float(), nullable=True),
            sa.Column("connectivity_confidence", sa.Float(), nullable=True),
            sa.Column("track_coverage", sa.Float(), nullable=True),
            sa.Column("avg_reprojection_error", sa.Float(), nullable=True),
            sa.Column("hero_photo_id", sa.Integer(), sa.ForeignKey("job_photos.id"), nullable=True),
            sa.Column("depth_range", sa.Float(), nullable=True),
            sa.Column("motion_affordance", sa.String(length=32), nullable=True),
            sa.Column("sparse_scene_uri", sa.String(length=500), nullable=True),
            sa.Column("point_cloud_uri", sa.String(length=500), nullable=True),
            sa.Column("track_bundle_uri", sa.String(length=500), nullable=True),
            sa.Column("debug_metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.UniqueConstraint("job_id", "component_key", name="uq_scene_component_job_key"),
        )
        op.create_index("ix_scene_components_job_id", "scene_components", ["job_id"])

    if not _table_exists("scene_component_memberships"):
        op.create_table(
            "scene_component_memberships",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("job_id", sa.Integer(), sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
            sa.Column("photo_id", sa.Integer(), sa.ForeignKey("job_photos.id", ondelete="CASCADE"), nullable=False),
            sa.Column("scene_component_id", sa.Integer(), sa.ForeignKey("scene_components.id", ondelete="CASCADE"), nullable=False),
            sa.Column("order_index", sa.Integer(), nullable=True),
            sa.Column("photo_role", sa.String(length=32), nullable=False, server_default="support"),
            sa.Column("is_primary", sa.Boolean(), nullable=False, server_default=sa.text("true")),
            sa.UniqueConstraint("job_id", "photo_id", name="uq_scene_component_membership_job_photo"),
        )
        op.create_index("ix_scene_component_memberships_job_id", "scene_component_memberships", ["job_id"])
        op.create_index("ix_scene_component_memberships_photo_id", "scene_component_memberships", ["photo_id"])
        op.create_index("ix_scene_component_memberships_scene_component_id", "scene_component_memberships", ["scene_component_id"])

    if not _table_exists("photo_scene_geometry"):
        op.create_table(
            "photo_scene_geometry",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("job_id", sa.Integer(), sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
            sa.Column("photo_id", sa.Integer(), sa.ForeignKey("job_photos.id", ondelete="CASCADE"), nullable=False),
            sa.Column("scene_component_id", sa.Integer(), sa.ForeignKey("scene_components.id", ondelete="CASCADE"), nullable=True),
            sa.Column("pose_confidence", sa.Float(), nullable=True),
            sa.Column("depth_confidence", sa.Float(), nullable=True),
            sa.Column("point_confidence", sa.Float(), nullable=True),
            sa.Column("visibility_score", sa.Float(), nullable=True),
            sa.Column("reprojection_error", sa.Float(), nullable=True),
            sa.Column("camera_extrinsic", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("camera_intrinsic", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("camera_center", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("view_direction", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("depth_artifact_uri", sa.String(length=500), nullable=True),
            sa.Column("point_map_artifact_uri", sa.String(length=500), nullable=True),
            sa.Column("sparse_visibility", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("local_metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.UniqueConstraint("job_id", "photo_id", name="uq_photo_scene_geometry_job_photo"),
        )
        op.create_index("ix_photo_scene_geometry_job_id", "photo_scene_geometry", ["job_id"])
        op.create_index("ix_photo_scene_geometry_photo_id", "photo_scene_geometry", ["photo_id"])
        op.create_index("ix_photo_scene_geometry_scene_component_id", "photo_scene_geometry", ["scene_component_id"])

    if not _table_exists("photo_relations"):
        op.create_table(
            "photo_relations",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("job_id", sa.Integer(), sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
            sa.Column("photo_a_id", sa.Integer(), sa.ForeignKey("job_photos.id", ondelete="CASCADE"), nullable=False),
            sa.Column("photo_b_id", sa.Integer(), sa.ForeignKey("job_photos.id", ondelete="CASCADE"), nullable=False),
            sa.Column("scene_component_id", sa.Integer(), sa.ForeignKey("scene_components.id", ondelete="CASCADE"), nullable=True),
            sa.Column("overlap_score", sa.Float(), nullable=True),
            sa.Column("track_support", sa.Float(), nullable=True),
            sa.Column("reprojection_score", sa.Float(), nullable=True),
            sa.Column("relation_confidence", sa.Float(), nullable=True),
            sa.Column("baseline_distance", sa.Float(), nullable=True),
            sa.Column("relative_transform", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("direction_dx", sa.Float(), nullable=True),
            sa.Column("direction_dy", sa.Float(), nullable=True),
            sa.Column("continuity_type", sa.String(length=32), nullable=False, server_default="weak"),
            sa.Column("is_bridge_edge", sa.Boolean(), nullable=False, server_default=sa.text("false")),
            sa.Column("is_connected", sa.Boolean(), nullable=False, server_default=sa.text("false")),
            sa.Column("debug_metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.UniqueConstraint("job_id", "photo_a_id", "photo_b_id", name="uq_photo_relation_job_pair"),
        )
        op.create_index("ix_photo_relations_job_id", "photo_relations", ["job_id"])
        op.create_index("ix_photo_relations_photo_a_id", "photo_relations", ["photo_a_id"])
        op.create_index("ix_photo_relations_photo_b_id", "photo_relations", ["photo_b_id"])
        op.create_index("ix_photo_relations_scene_component_id", "photo_relations", ["scene_component_id"])

    room_cluster_columns = _column_names("room_clusters")
    if "scene_component_id" not in room_cluster_columns:
        op.add_column("room_clusters", sa.Column("scene_component_id", sa.Integer(), nullable=True))
        op.create_foreign_key(
            "fk_room_clusters_scene_component_id",
            "room_clusters",
            "scene_components",
            ["scene_component_id"],
            ["id"],
            ondelete="SET NULL",
        )
        op.create_index("ix_room_clusters_scene_component_id", "room_clusters", ["scene_component_id"])
    if "geometry_confidence" not in room_cluster_columns:
        op.add_column("room_clusters", sa.Column("geometry_confidence", sa.Float(), nullable=True))

    _drop_table_if_exists("transition_sequence_steps")
    _drop_table_if_exists("transition_sequences")
    _drop_table_if_exists("photo_pose_alignments")
    _drop_table_if_exists("photo_similarities")


def downgrade() -> None:
    _drop_table_if_exists("photo_relations")
    _drop_table_if_exists("photo_scene_geometry")
    _drop_table_if_exists("scene_component_memberships")
    _drop_table_if_exists("scene_components")

    room_cluster_columns = _column_names("room_clusters")
    if "geometry_confidence" in room_cluster_columns:
        op.drop_column("room_clusters", "geometry_confidence")
    if "scene_component_id" in room_cluster_columns:
        _drop_index_if_exists("ix_room_clusters_scene_component_id", "room_clusters")
        op.drop_constraint("fk_room_clusters_scene_component_id", "room_clusters", type_="foreignkey")
        op.drop_column("room_clusters", "scene_component_id")
