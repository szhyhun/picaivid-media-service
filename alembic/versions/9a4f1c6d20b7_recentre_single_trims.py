"""Re-centre single-image trims inside the shorter source we now buy.

Singles used to be generated at five seconds and their trim was centred in
that: a 1.2s window sat at trim_start 1.9. A single is now bought at three
seconds, so 1.9 leaves only 1.1s of video behind it -- the shot silently
clamps short and the beat grid it was cut to stops holding.

Recentring uses each timeline's own grid (beats_per_shot x 60 / bpm) rather
than a constant, because that is what the window length actually is.

Revision ID: 9a4f1c6d20b7
Revises: 7d2e4c8b1f36
"""
from alembic import op

revision = "9a4f1c6d20b7"
down_revision = "7d2e4c8b1f36"
branch_labels = None
depends_on = None

SINGLE_SOURCE_SECONDS = 3.0


def upgrade() -> None:
    op.execute(f"""
        UPDATE timeline_clips AS tc
        SET trim_start = GREATEST(
            0.0,
            ({SINGLE_SOURCE_SECONDS} - LEAST(
                {SINGLE_SOURCE_SECONDS},
                COALESCE(
                    tc.duration_override,
                    COALESCE((t.beat_grid->>'beats_per_shot')::float, 2.0) * 60.0
                        / NULLIF(COALESCE((t.beat_grid->>'bpm')::float, 100.0), 0),
                    1.2
                )
            )) / 2.0
        )
        FROM timelines AS t
        WHERE tc.timeline_id = t.id
          AND tc.kind = 'single_image'
          AND COALESCE(tc.trim_start, 0) + COALESCE(
                tc.duration_override,
                COALESCE((t.beat_grid->>'beats_per_shot')::float, 2.0) * 60.0
                    / NULLIF(COALESCE((t.beat_grid->>'bpm')::float, 100.0), 0),
                1.2
              ) > {SINGLE_SOURCE_SECONDS}
    """)


def downgrade() -> None:
    # The old centring is recoverable from the grid, but restoring a trim that
    # points past the end of a three-second clip would only reintroduce the
    # bug. Leave the corrected values in place.
    pass
