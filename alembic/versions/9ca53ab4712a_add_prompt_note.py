"""A customer's own words, appended to the prompt we build.

`prompt_override` replaces the generated prompt outright, which is the right
tool for development and the wrong one for a customer: it throws away every
measured decision the pipeline made. A note is additive -- our direction still
goes to the model, theirs follows it.
"""
from alembic import op
import sqlalchemy as sa

revision = "9ca53ab4712a"
down_revision = "8e9948f93fdb"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("timeline_clips", sa.Column("prompt_note", sa.Text()))


def downgrade() -> None:
    op.drop_column("timeline_clips", "prompt_note")
