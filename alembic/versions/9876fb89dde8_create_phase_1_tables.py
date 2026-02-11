"""Create Phase 1 tables

Revision ID: 9876fb89dde8
Revises:
Create Date: 2026-02-10 21:25:18.968596

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '9876fb89dde8'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create jobs table first (no dependencies)
    op.create_table('jobs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('project_id', sa.String(length=36), nullable=False),
    sa.Column('status', sa.String(length=50), nullable=True),
    sa.Column('current_phase', sa.Integer(), nullable=True),
    sa.Column('template_type', sa.String(length=50), nullable=True),
    sa.Column('target_length', sa.Float(), nullable=True),
    sa.Column('music_uri', sa.String(length=500), nullable=True),
    sa.Column('bpm', sa.Integer(), nullable=True),
    sa.Column('beat_offset', sa.Float(), nullable=True),
    sa.Column('enable_beat_sync', sa.Boolean(), nullable=True),
    sa.Column('final_video_uri', sa.String(length=500), nullable=True),
    sa.Column('error_message', sa.String(length=1000), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_jobs_project_id'), 'jobs', ['project_id'], unique=False)
    op.create_index(op.f('ix_jobs_status'), 'jobs', ['status'], unique=False)

    # 2. Create job_photos without room_cluster_id FK (will add later)
    op.create_table('job_photos',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('job_id', sa.Integer(), nullable=False),
    sa.Column('rails_photo_id', sa.String(length=36), nullable=False),
    sa.Column('s3_uri', sa.String(length=500), nullable=False),
    sa.Column('filename', sa.String(length=255), nullable=True),
    sa.Column('width', sa.Integer(), nullable=True),
    sa.Column('height', sa.Integer(), nullable=True),
    sa.Column('position', sa.Integer(), nullable=True),
    sa.Column('room_label', sa.String(length=100), nullable=True),
    sa.Column('room_override', sa.String(length=100), nullable=True),
    sa.Column('room_cluster_id', sa.Integer(), nullable=True),
    sa.Column('exclude', sa.Boolean(), nullable=True),
    sa.Column('manual_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('sharpness', sa.Float(), nullable=True),
    sa.Column('exposure_score', sa.Float(), nullable=True),
    sa.Column('composition_score', sa.Float(), nullable=True),
    sa.Column('base_score', sa.Float(), nullable=True),
    sa.Column('final_score', sa.Float(), nullable=True),
    sa.Column('depth_variance', sa.Float(), nullable=True),
    sa.Column('depth_layers', sa.Integer(), nullable=True),
    sa.Column('embedding', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_job_photos_job_id'), 'job_photos', ['job_id'], unique=False)
    op.create_index(op.f('ix_job_photos_rails_photo_id'), 'job_photos', ['rails_photo_id'], unique=False)

    # 3. Create room_clusters (references jobs and job_photos)
    op.create_table('room_clusters',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('job_id', sa.Integer(), nullable=False),
    sa.Column('room_type', sa.String(length=100), nullable=True),
    sa.Column('confidence_tier', sa.String(length=20), nullable=True),
    sa.Column('sfm_eligible', sa.Boolean(), nullable=True),
    sa.Column('image_count', sa.Integer(), nullable=True),
    sa.Column('overlap_score', sa.Float(), nullable=True),
    sa.Column('depth_variance', sa.Float(), nullable=True),
    sa.Column('hero_photo_id', sa.Integer(), nullable=True),
    sa.Column('recommended_motion', sa.String(length=50), nullable=True),
    sa.Column('allowed_motion_types', sa.String(length=200), nullable=True),
    sa.Column('recommended_duration', sa.Float(), nullable=True),
    sa.ForeignKeyConstraint(['hero_photo_id'], ['job_photos.id'], ),
    sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_room_clusters_job_id'), 'room_clusters', ['job_id'], unique=False)

    # 4. Add the room_cluster_id FK to job_photos (circular reference)
    op.create_foreign_key(
        'fk_job_photos_room_cluster_id',
        'job_photos', 'room_clusters',
        ['room_cluster_id'], ['id']
    )

    # 5. Create analysis_results
    op.create_table('analysis_results',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('job_id', sa.Integer(), nullable=False),
    sa.Column('room_cluster_id', sa.Integer(), nullable=True),
    sa.Column('recommended_motion', sa.String(length=50), nullable=True),
    sa.Column('allowed_motion_types', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('recommended_duration', sa.Float(), nullable=True),
    sa.Column('tier', sa.String(length=20), nullable=True),
    sa.Column('model_recommendation', sa.String(length=100), nullable=True),
    sa.Column('debug_metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['room_cluster_id'], ['room_clusters.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_analysis_results_job_id'), 'analysis_results', ['job_id'], unique=False)

    # 6. Create clips
    op.create_table('clips',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('job_id', sa.Integer(), nullable=False),
    sa.Column('room_cluster_id', sa.Integer(), nullable=True),
    sa.Column('source_photo_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('motion_type', sa.String(length=50), nullable=True),
    sa.Column('model_used', sa.String(length=100), nullable=True),
    sa.Column('is_3d', sa.Boolean(), nullable=True),
    sa.Column('duration', sa.Float(), nullable=True),
    sa.Column('s3_uri', sa.String(length=500), nullable=True),
    sa.Column('validation_score', sa.Float(), nullable=True),
    sa.Column('status', sa.String(length=50), nullable=True),
    sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['room_cluster_id'], ['room_clusters.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_clips_job_id'), 'clips', ['job_id'], unique=False)

    # 7. Create timelines
    op.create_table('timelines',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('job_id', sa.Integer(), nullable=False),
    sa.Column('version', sa.Integer(), nullable=True),
    sa.Column('status', sa.String(length=50), nullable=True),
    sa.Column('beat_grid', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('total_duration', sa.Float(), nullable=True),
    sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_timelines_job_id'), 'timelines', ['job_id'], unique=False)

    # 8. Create timeline_clips
    op.create_table('timeline_clips',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('timeline_id', sa.Integer(), nullable=False),
    sa.Column('clip_id', sa.Integer(), nullable=False),
    sa.Column('order_index', sa.Integer(), nullable=False),
    sa.Column('in_time', sa.Float(), nullable=True),
    sa.Column('out_time', sa.Float(), nullable=True),
    sa.Column('transition_type', sa.String(length=50), nullable=True),
    sa.Column('audio_policy', sa.String(length=50), nullable=True),
    sa.ForeignKeyConstraint(['clip_id'], ['clips.id'], ),
    sa.ForeignKeyConstraint(['timeline_id'], ['timelines.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_timeline_clips_timeline_id'), 'timeline_clips', ['timeline_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_timeline_clips_timeline_id'), table_name='timeline_clips')
    op.drop_table('timeline_clips')
    op.drop_index(op.f('ix_timelines_job_id'), table_name='timelines')
    op.drop_table('timelines')
    op.drop_index(op.f('ix_clips_job_id'), table_name='clips')
    op.drop_table('clips')
    op.drop_index(op.f('ix_analysis_results_job_id'), table_name='analysis_results')
    op.drop_table('analysis_results')
    # Drop FK first before dropping room_clusters
    op.drop_constraint('fk_job_photos_room_cluster_id', 'job_photos', type_='foreignkey')
    op.drop_index(op.f('ix_room_clusters_job_id'), table_name='room_clusters')
    op.drop_table('room_clusters')
    op.drop_index(op.f('ix_job_photos_rails_photo_id'), table_name='job_photos')
    op.drop_index(op.f('ix_job_photos_job_id'), table_name='job_photos')
    op.drop_table('job_photos')
    op.drop_index(op.f('ix_jobs_status'), table_name='jobs')
    op.drop_index(op.f('ix_jobs_project_id'), table_name='jobs')
    op.drop_table('jobs')
