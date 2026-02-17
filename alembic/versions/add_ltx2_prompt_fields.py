"""Add LTX-2 prompt and parameter fields

Revision ID: add_ltx2_prompt_fields
Revises: 9876fb89dde8
Create Date: 2026-02-12

Adds fields for LTX-2 prompting strategy:
- analysis_results: prompt_template, cfg_scale, inference_steps
- clips: prompt_used, cfg_scale, inference_steps, seed, s3_uri_upscaled, upscale_status, render_metadata
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'add_ltx2_prompt_fields'
down_revision: Union[str, None] = '9876fb89dde8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add LTX-2 prompt and parameter fields to analysis_results
    op.add_column('analysis_results', sa.Column('prompt_template', sa.Text(), nullable=True))
    op.add_column('analysis_results', sa.Column('cfg_scale', sa.Float(), nullable=True))
    op.add_column('analysis_results', sa.Column('inference_steps', sa.Integer(), nullable=True))

    # Add LTX-2 prompt and parameter fields to clips
    op.add_column('clips', sa.Column('prompt_used', sa.Text(), nullable=True))
    op.add_column('clips', sa.Column('cfg_scale', sa.Float(), nullable=True))
    op.add_column('clips', sa.Column('inference_steps', sa.Integer(), nullable=True))
    op.add_column('clips', sa.Column('seed', sa.Integer(), nullable=True))
    op.add_column('clips', sa.Column('s3_uri_upscaled', sa.String(length=500), nullable=True))
    op.add_column('clips', sa.Column('upscale_status', sa.String(length=50), nullable=True))
    op.add_column('clips', sa.Column('render_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    # Remove fields from clips
    op.drop_column('clips', 'render_metadata')
    op.drop_column('clips', 'upscale_status')
    op.drop_column('clips', 's3_uri_upscaled')
    op.drop_column('clips', 'seed')
    op.drop_column('clips', 'inference_steps')
    op.drop_column('clips', 'cfg_scale')
    op.drop_column('clips', 'prompt_used')

    # Remove fields from analysis_results
    op.drop_column('analysis_results', 'inference_steps')
    op.drop_column('analysis_results', 'cfg_scale')
    op.drop_column('analysis_results', 'prompt_template')
