"""Alembic environment configuration.

Alembic is the database migration tool for SQLAlchemy (like Rails migrations).
This file configures how Alembic connects to the database and which models to track.
"""
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Import the SQLAlchemy Base and all models
# Alembic uses Base.metadata to detect table definitions
from app.db.base import Base
from app.db.models import Job, JobPhoto, RoomCluster, AnalysisResult, Clip, Timeline, TimelineClip

# Alembic Config object - reads from alembic.ini
config = context.config

# Setup Python logging from the config file
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# This is the metadata object that Alembic uses to detect schema changes
# When you run `alembic revision --autogenerate`, it compares this metadata
# against the actual database and generates migration code for the differences
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Generates SQL without connecting to the database.
    Useful for generating SQL scripts to run manually.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Connects to the database and applies migrations directly.
    This is the normal mode for `alembic upgrade head`.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
