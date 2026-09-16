"""Mark a move the customer chose, so position stops overruling it.

`_single_motion_label` forces the move for the shots whose grammar is
measured: a detail pushes in, an aerial or a closing shot pulls back. That is
right as a default and wrong as a rule once a customer picks something, because
their pick would be discarded in silence.
"""
from alembic import op
import sqlalchemy as sa

revision = "164ae810dcce"
down_revision = "9ca53ab4712a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("timeline_clips",
                  sa.Column("move_locked", sa.Boolean(), server_default=sa.false()))
    op.execute("UPDATE timeline_clips SET move_locked = false WHERE move_locked IS NULL")


def downgrade() -> None:
    op.drop_column("timeline_clips", "move_locked")
