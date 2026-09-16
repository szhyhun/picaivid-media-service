"""Keep the measured move when a customer composes their own.

`move_label` used to be overwritten by a customer's choice, which made the
planner's own decision unrecoverable -- there was nothing left to reset to.
The measured move stays in `move_label`; the composed one lives here.

Revision ID: 3b1c7f2ae904
Revises: 164ae810dcce
"""
from alembic import op
import sqlalchemy as sa

revision = "3b1c7f2ae904"
down_revision = "164ae810dcce"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("timeline_clips",
                  sa.Column("move_chosen", sa.String(length=120), nullable=True))
    # Rows locked before this column existed had their measured move
    # overwritten in place; the composed value is all that survives, so carry
    # it across rather than pretending a reset target exists.
    op.execute("UPDATE timeline_clips SET move_chosen = move_label "
               "WHERE move_locked = true")


def downgrade() -> None:
    op.drop_column("timeline_clips", "move_chosen")
