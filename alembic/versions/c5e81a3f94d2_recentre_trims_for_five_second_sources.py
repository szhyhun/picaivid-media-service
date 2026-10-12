"""Re-centre every trim inside a five-second source.

Singles were briefly bought at three seconds and their trims were re-centred
for that. Ray 3.2 sells only 5s and 10s for image-to-video, and 10s refuses
the start/end keyframes a pair depends on, so the three-second buy was never
purchasable and the shorter centring now points at the wrong part of a
five-second clip.

Centring matters: input anchoring and generation artefacts are worst at both
ends of a generated clip, so the window we actually show is taken from the
stable middle.

Revision ID: c5e81a3f94d2
Revises: 9a4f1c6d20b7
"""
from alembic import op

revision = "c5e81a3f94d2"
down_revision = "9a4f1c6d20b7"
branch_labels = None
depends_on = None

SOURCE_SECONDS = 5.0


def upgrade() -> None:
    # Only shots still sitting at the default centre are moved. A trim someone
    # set by hand is an editorial decision and is left exactly where it is.
    op.execute(f"""
        UPDATE timeline_clips AS tc
        SET trim_start = GREATEST(
            0.0,
            ({SOURCE_SECONDS} - LEAST({SOURCE_SECONDS}, COALESCE(
                tc.duration_override,
                COALESCE((t.beat_grid->>'beats_per_shot')::float, 2.0) * 60.0
                    / NULLIF(COALESCE((t.beat_grid->>'bpm')::float, 100.0), 0),
                1.2
            ))) / 2.0
        )
        FROM timelines AS t
        WHERE tc.timeline_id = t.id
          AND tc.kind = 'single_image'
    """)


def downgrade() -> None:
    pass
