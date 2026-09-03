"""Add job progress reporting, heartbeat, and cooperative cancellation.

Purely additive: every column is nullable or has a server default, so a worker
running the previous code keeps working against this schema unchanged.

Motivation is recorded in JOB_STATUS_PLAN.md. Briefly: the pipeline already knew
its progress (it logs VGGT_V2_PAIR_PROGRESS every 25 pairs) but nothing persisted
it, so the UI could only ever show a spinner -- and three separate outages
(LocalStack OOM, no worker running, wrong WORKER_TYPE) were indistinguishable
from "still working" for as long as anyone cared to wait.

Revision ID: 7b41e9c05d3a
Revises: 6c7d8e9f0a12
"""
import sqlalchemy as sa
from alembic import op

revision = "7b41e9c05d3a"
down_revision = "6c7d8e9f0a12"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("progress_current", sa.Integer(), nullable=True))
    op.add_column("jobs", sa.Column("progress_total", sa.Integer(), nullable=True))
    op.add_column("jobs", sa.Column("progress_label", sa.String(length=120), nullable=True))
    op.add_column("jobs", sa.Column("phase_started_at", sa.DateTime(), nullable=True))
    op.add_column("jobs", sa.Column("heartbeat_at", sa.DateTime(), nullable=True))
    op.add_column(
        "jobs",
        sa.Column(
            "cancel_requested",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column("jobs", sa.Column("canceled_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "canceled_at")
    op.drop_column("jobs", "cancel_requested")
    op.drop_column("jobs", "heartbeat_at")
    op.drop_column("jobs", "phase_started_at")
    op.drop_column("jobs", "progress_label")
    op.drop_column("jobs", "progress_total")
    op.drop_column("jobs", "progress_current")
