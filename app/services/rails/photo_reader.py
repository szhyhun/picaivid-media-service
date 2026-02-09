"""Read-only access to Rails photos table."""
import logging
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RailsPhoto:
    """Dataclass representing a photo from Rails database (read-only)."""
    id: str  # UUID
    project_id: str  # UUID
    s3_object_key: str
    filename: str
    width: Optional[int]
    height: Optional[int]
    position: int
    room_type: Optional[str]  # Manual override if set
    metadata: Optional[dict]


class RailsPhotoReader:
    """Read-only access to Rails photos table.

    This service reads from Rails' PostgreSQL database but never writes.
    All derived data goes to Python's own tables.
    """

    def __init__(self, database_url: Optional[str] = None):
        self._database_url = database_url or settings.DATABASE_URL
        self._engine: Optional[Engine] = None

    @property
    def engine(self) -> Engine:
        """Lazy-load database engine."""
        if self._engine is None:
            self._engine = create_engine(self._database_url)
        return self._engine

    def get_photos_by_project_id(self, project_id: str) -> List[RailsPhoto]:
        """Get all photos for a project from Rails database.

        Args:
            project_id: Rails project UUID

        Returns:
            List of RailsPhoto dataclasses
        """
        query = text("""
            SELECT
                id,
                project_id,
                s3_object_key,
                filename,
                width,
                height,
                position,
                room_type,
                metadata
            FROM photos
            WHERE project_id = :project_id
            AND status = 2  -- ready status in Rails enum
            ORDER BY position ASC
        """)

        photos = []
        with self.engine.connect() as conn:
            result = conn.execute(query, {"project_id": project_id})
            for row in result:
                photo = RailsPhoto(
                    id=str(row.id),
                    project_id=str(row.project_id),
                    s3_object_key=row.s3_object_key or "",
                    filename=row.filename,
                    width=row.width,
                    height=row.height,
                    position=row.position or 0,
                    room_type=row.room_type,
                    metadata=row.metadata or {},
                )
                photos.append(photo)

        logger.info(f"Read {len(photos)} photos from Rails for project {project_id}")
        return photos

    def get_project_exists(self, project_id: str) -> bool:
        """Check if project exists in Rails database.

        Args:
            project_id: Rails project UUID

        Returns:
            True if project exists
        """
        query = text("SELECT 1 FROM projects WHERE id = :project_id LIMIT 1")
        with self.engine.connect() as conn:
            result = conn.execute(query, {"project_id": project_id})
            return result.fetchone() is not None


# Singleton instance
rails_photo_reader = RailsPhotoReader()
