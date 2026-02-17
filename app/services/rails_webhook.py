"""Rails webhook client for notifying project status updates."""
import logging
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


def notify_render_complete(project_id: str) -> bool:
    """Notify Rails that rendering is complete.

    Args:
        project_id: Rails project UUID

    Returns:
        True if notification succeeded
    """
    if not settings.RAILS_WEBHOOK_URL:
        logger.warning("RAILS_WEBHOOK_URL not configured, skipping webhook")
        return False

    url = f"{settings.RAILS_WEBHOOK_URL}/webhooks/media_service/render-complete"

    try:
        response = httpx.post(url, json={"project_id": project_id}, timeout=10.0)
        response.raise_for_status()
        logger.info(f"Notified Rails of render complete for project {project_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to notify Rails: {e}")
        return False


def notify_render_failed(project_id: str, error: str) -> bool:
    """Notify Rails that rendering failed.

    Args:
        project_id: Rails project UUID
        error: Error message

    Returns:
        True if notification succeeded
    """
    if not settings.RAILS_WEBHOOK_URL:
        logger.warning("RAILS_WEBHOOK_URL not configured, skipping webhook")
        return False

    url = f"{settings.RAILS_WEBHOOK_URL}/webhooks/media_service/render-failed"

    try:
        response = httpx.post(
            url,
            json={"project_id": project_id, "error": error},
            timeout=10.0
        )
        response.raise_for_status()
        logger.info(f"Notified Rails of render failure for project {project_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to notify Rails: {e}")
        return False
