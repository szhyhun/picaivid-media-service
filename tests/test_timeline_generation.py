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

    def test_provider_result_download_checks_http_status_and_bytes(self):
        response = MagicMock(content=b"mp4")
        with patch.object(generate.httpx, "get", return_value=response) as get:
            self.assertEqual(
                generate._download_video("https://example.test/v.mp4"), b"mp4")
        get.assert_called_once_with("https://example.test/v.mp4", timeout=120.0)
        response.raise_for_status.assert_called_once_with()

    def test_empty_provider_result_is_rejected(self):
        response = MagicMock(content=b"")
        with patch.object(generate.httpx, "get", return_value=response):
            with self.assertRaises(generate.RunwareError):
                generate._download_video("https://example.test/v.mp4")


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
                         {"old": True, "model": "luma:ray@3.2", "duration": 5,
                          "resolution": "1080p", "hdr": True, "exr_export": False})
        db.commit.assert_called_once()

    def test_background_runner_only_claims_persisted_queue_entries(self):
        queued = SimpleNamespace(clip=SimpleNamespace(status="queued"))
        active = SimpleNamespace(clip=SimpleNamespace(status="rendering"))
        planned = SimpleNamespace(clip=None)
        timeline = SimpleNamespace(ordered_clips=lambda: [queued, active, planned])
        self.assertEqual(generate.queued_shots(timeline), [queued])

    def test_input_failures_are_counted_and_reported(self):
        clips = [
            SimpleNamespace(status="queued", render_metadata={}),
            SimpleNamespace(status="queued", render_metadata={}),
        ]
        shots = [
            SimpleNamespace(shot_uid="a", clip=clips[0], photo_ids=[]),
            SimpleNamespace(shot_uid="b", clip=clips[1], photo_ids=[]),
        ]
        progress = MagicMock()
        with patch.object(generate, "queued_shots", return_value=shots), \
             patch.object(generate, "_prepare", side_effect=generate.RunwareError("bad input")):
            result = generate.generate_timeline_clips(
                MagicMock(), object(), progress=progress)

        self.assertEqual(result, {"requested": 2, "succeeded": 0, "failed": 2})
        self.assertTrue(all(clip.status == "failed" for clip in clips))
        progress.update.assert_called_with(2, 2, "Generating clips")

    def test_explicit_mode_overrides_endpoint_detection(self):
        with patch.object(generate.settings, "RUNWARE_INPUT_MODE", "url"), \
             patch.object(generate.settings, "S3_ENDPOINT", "http://localhost:9000"):
            self.assertFalse(generate._uses_inline_media())
        with patch.object(generate.settings, "RUNWARE_INPUT_MODE", "data_uri"), \
             patch.object(generate.settings, "S3_ENDPOINT", None):
            self.assertTrue(generate._uses_inline_media())


if __name__ == "__main__":
    unittest.main()
