"""Input transport rules for Runware timeline generation."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.pipeline.phase3_timeline import generate


class InputTransportTests(unittest.TestCase):
    def test_auto_uses_presigned_urls_for_production_s3(self):
        with patch.object(generate.settings, "RUNWARE_INPUT_MODE", "auto"), \
             patch.object(generate.settings, "S3_ENDPOINT", None):
            self.assertFalse(generate._uses_inline_media())

    def test_auto_sends_bytes_when_minio_is_only_local(self):
        with patch.object(generate.settings, "RUNWARE_INPUT_MODE", "auto"), \
             patch.object(generate.settings, "S3_ENDPOINT", "http://localhost:9000"):
            self.assertTrue(generate._uses_inline_media())


class QueueTests(unittest.TestCase):
    def test_queue_persists_shots_before_background_work_starts(self):
        first = SimpleNamespace(shot_uid="first")
        second = SimpleNamespace(shot_uid="second")
        second_clip = SimpleNamespace(render_metadata={"old": True}, status=None)
        db = MagicMock()

        with patch.object(generate, "pending_shots", return_value=[first, second]), \
             patch.object(generate, "_prepare", return_value=second_clip) as prepare:
            queued = generate.queue_timeline_clips(
                db, object(), shot_uids=["second"], resolution="1080p")

        self.assertEqual(queued, ["second"])
        prepare.assert_called_once()
        self.assertEqual(second_clip.status, "queued")
        self.assertEqual(second_clip.render_metadata,
                         {"old": True, "resolution": "1080p"})
        db.commit.assert_called_once()

    def test_background_runner_only_claims_persisted_queue_entries(self):
        queued = SimpleNamespace(clip=SimpleNamespace(status="queued"))
        active = SimpleNamespace(clip=SimpleNamespace(status="rendering"))
        planned = SimpleNamespace(clip=None)
        timeline = SimpleNamespace(ordered_clips=lambda: [queued, active, planned])
        self.assertEqual(generate.queued_shots(timeline), [queued])

    def test_explicit_mode_overrides_endpoint_detection(self):
        with patch.object(generate.settings, "RUNWARE_INPUT_MODE", "url"), \
             patch.object(generate.settings, "S3_ENDPOINT", "http://localhost:9000"):
            self.assertFalse(generate._uses_inline_media())
        with patch.object(generate.settings, "RUNWARE_INPUT_MODE", "data_uri"), \
             patch.object(generate.settings, "S3_ENDPOINT", None):
            self.assertTrue(generate._uses_inline_media())


if __name__ == "__main__":
    unittest.main()
