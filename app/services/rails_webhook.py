"""Rails webhook client for notifying project status updates."""
import logging
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


def _headers() -> dict[str, str]:
    key = settings.MEDIA_SERVICE_WEBHOOK_KEY or settings.MEDIA_SERVICE_API_KEY or settings.API_KEY
    return {"X-Media-Service-Key": key} if key else {}


def notify_analysis_complete(project_id: str) -> bool:
    """Notify Rails that the renderer-neutral cinematic plan is ready."""
    if not settings.RAILS_WEBHOOK_URL:
        logger.warning("RAILS_WEBHOOK_URL not configured, skipping webhook")
        return False

    url = f"{settings.RAILS_WEBHOOK_URL}/webhooks/media_service/analysis-complete"

    try:
        response = httpx.post(
            url, json={"project_id": project_id}, headers=_headers(), timeout=10.0
        )
        response.raise_for_status()
        logger.info(f"Notified Rails of analysis complete for project {project_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to notify Rails of analysis completion: {e}")
        return False


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
        response = httpx.post(
            url, json={"project_id": project_id}, headers=_headers(), timeout=10.0
        )
        response.raise_for_status()
        logger.info(f"Notified Rails of render complete for project {project_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to notify Rails: {e}")
        return False


def notify_timeline_complete(
    project_id: str, *, requested: int, succeeded: int, failed: int
) -> bool:
    """Tell Rails that a timeline generation command finished."""
    if not settings.RAILS_WEBHOOK_URL:
        logger.warning("RAILS_WEBHOOK_URL not configured, skipping webhook")
        return False

    url = f"{settings.RAILS_WEBHOOK_URL}/webhooks/media_service/timeline-complete"
    payload = {
        "project_id": project_id,
        "requested": requested,
        "succeeded": succeeded,
        "failed": failed,
    }

    try:
        response = httpx.post(url, json=payload, headers=_headers(), timeout=10.0)
        response.raise_for_status()
        logger.info("Notified Rails of timeline generation for project %s", project_id)
        return True
    except Exception as e:
        logger.error("Failed to notify Rails of timeline generation: %s", e)
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
            headers=_headers(),
            timeout=10.0
        )
        response.raise_for_status()
        logger.info(f"Notified Rails of render failure for project {project_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to notify Rails: {e}")
        return False
