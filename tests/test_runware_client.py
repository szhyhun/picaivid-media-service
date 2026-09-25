"""The request we send Runware, and the money-losing mistakes it must not make."""
import asyncio
import unittest
from unittest.mock import patch

from app.services.runware import client as rw


class FrameImageTests(unittest.TestCase):
    def test_one_photo_becomes_a_first_frame(self):
        self.assertEqual(
            rw.build_frame_images(["a.jpg"]),
            [{"image": "a.jpg", "frame": "first"}],
        )

    def test_two_photos_pin_both_ends_of_the_generated_move(self):
        # This is the whole point of the pairing geometry: with a real
        # photograph at each end the model only invents the middle.
        self.assertEqual(
            rw.build_frame_images(["a.jpg", "b.jpg"]),
            [
                {"image": "a.jpg", "frame": "first"},
                {"image": "b.jpg", "frame": "last"},
            ],
        )

    def test_a_shot_with_no_photograph_is_refused(self):
        with self.assertRaises(rw.RunwareError):
            rw.build_frame_images([])


class VideoTaskTests(unittest.TestCase):
    def task(self, **kw):
        return rw.build_video_task(
            prompt=kw.pop("prompt", "p"),
            photo_urls=kw.pop("photo_urls", ["a.jpg"]),
            **kw,
        )

    def test_it_targets_ray32_with_five_second_hdr_output(self):
        task = self.task()
        self.assertEqual(task["model"], "luma:ray@3.2")
        self.assertEqual(task["duration"], 5)
        self.assertEqual(task["resolution"], "720p")
        self.assertEqual(task["settings"], {"hdr": True, "exrExport": False})
        self.assertEqual(task["numberResults"], 1)
        self.assertEqual(task["outputQuality"], 95)
        self.assertEqual(task["deliveryMethod"], "async")

    def test_it_does_not_request_provider_cost_by_default(self):
        self.assertIs(self.task()["includeCost"], False)

    def test_it_uses_explicit_ray_keyframe_objects(self):
        self.assertEqual(
            self.task(photo_urls=["a.jpg", "b.jpg"])["inputs"]["frameImages"],
            [
                {"image": "a.jpg", "frame": "first"},
                {"image": "b.jpg", "frame": "last"},
            ],
        )

    def test_it_does_not_send_vidu_or_dimensions_to_ray(self):
        task = self.task()
        self.assertNotIn("providerSettings", task)
        self.assertNotIn("width", task)
        self.assertNotIn("height", task)

    def test_resolution_is_a_ray_preset(self):
        self.assertEqual(rw.resolution_size("540p"), (960, 540))
        self.assertEqual(self.task(resolution="1080p")["resolution"], "1080p")

    def test_hdr_rejects_540p_before_submission(self):
        with self.assertRaises(rw.RunwareError):
            self.task(resolution="540p")

    def test_frame_generation_requires_five_seconds(self):
        with self.assertRaises(rw.RunwareError):
            self.task(duration=3)

    def test_wrong_model_is_rejected_before_submission(self):
        with self.assertRaises(rw.RunwareError):
            self.task(model="vidu:4@2")

    def test_wrong_configured_model_is_rejected_before_submission(self):
        with patch.object(rw.settings, "RUNWARE_MODEL", "vidu:4@2"):
            with self.assertRaises(rw.RunwareError):
                self.task()

    def test_an_unknown_resolution_fails_loudly(self):
        with self.assertRaises(rw.RunwareError):
            rw.resolution_size("4k")

    def test_the_task_id_is_ours_so_a_retry_can_reuse_it(self):
        self.assertEqual(self.task(task_uuid="fixed")["taskUUID"], "fixed")


class PollingTests(unittest.TestCase):
    def test_terminal_error_uses_documented_error_status(self):
        async def run():
            async def response(*_args, **_kwargs):
                return {"errors": [{"status": "error", "code": "timeoutProvider",
                                    "message": "provider timed out"}]}

            with patch.object(rw, "_post", side_effect=response):
                return await rw.poll_video(object(), "task", interval=1, max_attempts=1)

        result = asyncio.run(run())
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "provider timed out")

    def test_completed_video_keeps_provider_identifiers_and_cost(self):
        async def run():
            async def response(*_args, **_kwargs):
                return {"data": [{"status": "success", "videoUUID": "video-1",
                                  "videoURL": "https://example.test/clip.mp4",
                                  "seed": 42, "cost": 0.12}]}

            with patch.object(rw, "_post", side_effect=response):
                return await rw.poll_video(object(), "task", interval=1, max_attempts=1)

        result = asyncio.run(run())
        self.assertTrue(result.ok)
        self.assertEqual(result.video_uuid, "video-1")
        self.assertEqual(result.seed, 42)
        self.assertAlmostEqual(result.cost or 0, 0.12)

    def test_polling_has_a_wall_clock_deadline(self):
        async def run():
            async def response(*_args, **_kwargs):
                return {"data": [{"status": "processing"}]}

            with patch.object(rw, "_post", side_effect=response), \
                 patch.object(rw.time, "monotonic", side_effect=[0.0, 0.0, 2.0]):
                return await rw.poll_video(
                    object(), "task", interval=1, max_attempts=99,
                    max_wait_seconds=1,
                )

        result = asyncio.run(run())
        self.assertFalse(result.ok)
        self.assertIn("Timed out after 1s", result.error or "")


class CredentialTests(unittest.TestCase):
    def test_a_missing_api_key_is_refused_before_any_request(self):
        with patch.object(rw.settings, "RUNWARE_API_KEY", None):
            with self.assertRaises(rw.RunwareError):
                rw._headers()


if __name__ == "__main__":
    unittest.main()
