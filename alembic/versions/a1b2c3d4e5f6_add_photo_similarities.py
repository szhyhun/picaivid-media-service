"""add_photo_similarities

Revision ID: a1b2c3d4e5f6
Revises: efdd6a93125f
Create Date: 2026-02-16

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'efdd6a93125f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'photo_similarities',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('job_id', sa.Integer(), nullable=False),
        sa.Column('photo_a_id', sa.Integer(), nullable=False),
        sa.Column('photo_b_id', sa.Integer(), nullable=False),
        sa.Column('pair_source', sa.String(50), nullable=True),
        sa.Column('dinov2_similarity', sa.Float(), nullable=True),
        sa.Column('geometric_matches', sa.Integer(), nullable=True),
        sa.Column('geometric_inliers', sa.Integer(), nullable=True),
        sa.Column('geometric_score', sa.Float(), nullable=True),
        sa.Column('is_connected', sa.Integer(), default=0),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['photo_a_id'], ['job_photos.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['photo_b_id'], ['job_photos.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('job_id', 'photo_a_id', 'photo_b_id', name='uq_photo_pair'),
    )
    op.create_index('ix_photo_similarities_job_id', 'photo_similarities', ['job_id'])
    op.create_index('ix_photo_similarities_photo_a_id', 'photo_similarities', ['photo_a_id'])
    op.create_index('ix_photo_similarities_photo_b_id', 'photo_similarities', ['photo_b_id'])


def downgrade() -> None:
    op.drop_index('ix_photo_similarities_photo_b_id', 'photo_similarities')
    op.drop_index('ix_photo_similarities_photo_a_id', 'photo_similarities')
    op.drop_index('ix_photo_similarities_job_id', 'photo_similarities')
    op.drop_table('photo_similarities')
