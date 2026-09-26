"""Record the instant a clip is handed to the provider.

Billing starts at submission, so "have we submitted this clip?" has to be a
durable fact rather than something inferred from status -- a worker that dies
mid-generation leaves status untouched, and the next worker cannot otherwise
tell an unsent clip from one that is already being paid for.

Revision ID: 7d2e4c8b1f36
Revises: 3b1c7f2ae904
"""
from alembic import op
import sqlalchemy as sa

revision = "7d2e4c8b1f36"
down_revision = "3b1c7f2ae904"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("clips",
                  sa.Column("provider_submitted_at", sa.DateTime(), nullable=True))
    # Anything already past submission was, by definition, submitted. Without
    # this backfill the first run after deploy would re-submit in-flight work
    # and bill it a second time -- the exact failure this column prevents.
    op.execute(
        "UPDATE clips SET provider_submitted_at = NOW() "
        "WHERE provider_task_uuid IS NOT NULL "
        "AND LOWER(COALESCE(status, '')) NOT IN ('', 'pending')"
    )


def downgrade() -> None:
    op.drop_column("clips", "provider_submitted_at")
