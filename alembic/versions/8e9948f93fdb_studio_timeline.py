"""Studio timeline: one row per shot, spanning storyboard and edit.

The shot line was ephemeral -- recomputed on every request, identified only by
list position. Nothing could be edited because nothing was stored.

These columns let one `timeline_clips` row serve both stages. In the storyboard a
shot is a *plan*: photos, kind, camera move, no duration. In the timeline it is a
*clip*: three generated seconds you can trim anywhere inside. Keeping them in one
row is what makes "regenerate this one shot" possible and lets the storyboard
show what has already been made.

`clip_id` becomes nullable so a planned shot can exist before its video does --
that is the whole point of the four shot states, which are derived from
(clip_id, Clip.status) rather than duplicated into a column here.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "8e9948f93fdb"
down_revision = "7b41e9c05d3a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A planned shot has no clip yet.
    op.alter_column("timeline_clips", "clip_id",
                    existing_type=sa.Integer(), nullable=True)

    # Stable identity. List position cannot survive a reorder, and every edit
    # operation needs to name a shot that outlives its index.
    op.add_column("timeline_clips", sa.Column("shot_uid", sa.String(36)))
    op.create_index("ix_timeline_clips_shot_uid", "timeline_clips", ["shot_uid"])

    # Storyboard columns: what the shot IS, independent of any generated video.
    for name, type_ in (
        ("room_id", sa.String(120)),
        ("room_type", sa.String(60)),
        ("act", sa.Integer()),
        ("kind", sa.String(30)),
        ("photo_ids", postgresql.JSONB()),
        ("move_label", sa.String(120)),
        ("move_vector", postgresql.JSONB()),
        ("is_close_up", sa.Boolean()),
        ("crop_box", postgresql.JSONB()),
        ("subject_box", postgresql.JSONB()),
        ("source_size", postgresql.JSONB()),
        ("detail_subject", sa.String(120)),
        ("prompt_override", sa.Text()),
    ):
        op.add_column("timeline_clips", sa.Column(name, type_))

    # Edit columns. `trim_start` is the offset into the generated source;
    # `duration_override` NULL means "follow the beat grid", so changing the
    # music retimes every unpinned shot without regenerating anything.
    op.add_column("timeline_clips",
                  sa.Column("trim_start", sa.Float(), server_default="0"))
    op.add_column("timeline_clips", sa.Column("duration_override", sa.Float()))

    op.execute("UPDATE timeline_clips SET is_close_up = false WHERE is_close_up IS NULL")

    # Provider linkage. `provider_task_uuid` is written before we await, so a
    # worker restart resumes an in-flight generation instead of paying twice.
    op.add_column("clips", sa.Column("shot_uid", sa.String(36)))
    op.add_column("clips", sa.Column("provider", sa.String(50)))
    op.add_column("clips", sa.Column("provider_task_uuid", sa.String(64)))
    op.create_index("ix_clips_shot_uid", "clips", ["shot_uid"])
    op.create_index("ix_clips_provider_task_uuid", "clips", ["provider_task_uuid"])


def downgrade() -> None:
    op.drop_index("ix_clips_provider_task_uuid", table_name="clips")
    op.drop_index("ix_clips_shot_uid", table_name="clips")
    for name in ("provider_task_uuid", "provider", "shot_uid"):
        op.drop_column("clips", name)

    for name in ("duration_override", "trim_start", "prompt_override",
                 "detail_subject", "source_size", "subject_box", "crop_box",
                 "is_close_up", "move_vector", "move_label", "photo_ids",
                 "kind", "act", "room_type", "room_id"):
        op.drop_column("timeline_clips", name)
    op.drop_index("ix_timeline_clips_shot_uid", table_name="timeline_clips")
    op.drop_column("timeline_clips", "shot_uid")

    op.alter_column("timeline_clips", "clip_id",
                    existing_type=sa.Integer(), nullable=False)
