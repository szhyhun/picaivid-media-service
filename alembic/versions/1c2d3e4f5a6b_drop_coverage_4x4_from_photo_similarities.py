"""drop coverage_4x4 from photo_similarities

Revision ID: 1c2d3e4f5a6b
Revises: f1a2b3c4d5e6
Create Date: 2026-03-18 19:45:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "1c2d3e4f5a6b"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None


def _column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = inspector.get_columns(table_name)
    return any(col["name"] == column_name for col in columns)


def upgrade() -> None:
    if _column_exists("photo_similarities", "coverage_4x4"):
        op.drop_column("photo_similarities", "coverage_4x4")


def downgrade() -> None:
    if not _column_exists("photo_similarities", "coverage_4x4"):
        op.add_column("photo_similarities", sa.Column("coverage_4x4", sa.Float(), nullable=True))
