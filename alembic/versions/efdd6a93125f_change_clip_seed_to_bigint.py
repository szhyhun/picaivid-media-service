"""change_clip_seed_to_bigint

Revision ID: efdd6a93125f
Revises: add_ltx2_prompt_fields
Create Date: 2026-02-13 02:01:17.739019

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'efdd6a93125f'
down_revision: Union[str, None] = 'add_ltx2_prompt_fields'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        'clips',
        'seed',
        type_=sa.BigInteger(),
        existing_type=sa.Integer(),
    )


def downgrade() -> None:
    op.alter_column(
        'clips',
        'seed',
        type_=sa.Integer(),
        existing_type=sa.BigInteger(),
    )
