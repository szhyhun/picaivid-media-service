"""The media HTTP API is a Rails-only internal surface."""
import unittest
from unittest.mock import patch

from app.main import _valid_internal_service_key


class InternalApiAuthTests(unittest.TestCase):
    def test_accepts_the_configured_service_key(self):
        with patch("app.main.settings.MEDIA_SERVICE_API_KEY", "rails-to-media"):
            self.assertTrue(_valid_internal_service_key("rails-to-media"))

    def test_rejects_missing_or_wrong_keys(self):
        with patch("app.main.settings.MEDIA_SERVICE_API_KEY", "rails-to-media"):
            self.assertFalse(_valid_internal_service_key(""))
            self.assertFalse(_valid_internal_service_key("browser-request"))

    def test_has_no_valid_key_when_service_auth_is_unconfigured(self):
        with patch("app.main.settings.MEDIA_SERVICE_API_KEY", None), patch(
            "app.main.settings.API_KEY", None
        ):
            self.assertFalse(_valid_internal_service_key("anything"))


if __name__ == "__main__":
    unittest.main()
