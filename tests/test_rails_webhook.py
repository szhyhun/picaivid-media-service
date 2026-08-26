import unittest
from unittest.mock import Mock, patch

from app.services import rails_webhook


class RailsWebhookTests(unittest.TestCase):
    def test_notify_analysis_complete(self) -> None:
        response = Mock()
        response.raise_for_status = Mock()
        post = Mock(return_value=response)

        with patch.object(rails_webhook.httpx, "post", post), patch.object(
            rails_webhook.settings,
            "RAILS_WEBHOOK_URL",
            "http://rails.test",
        ):
            self.assertTrue(rails_webhook.notify_analysis_complete("project-123"))

        post.assert_called_once_with(
            "http://rails.test/webhooks/media_service/analysis-complete",
            json={"project_id": "project-123"},
            timeout=10.0,
        )
        response.raise_for_status.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
