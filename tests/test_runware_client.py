"""The request we send Runware, and the money-losing mistakes it must not make."""
import unittest
from unittest.mock import patch

from app.services.runware import client as rw


class FrameImageTests(unittest.TestCase):
    def test_one_photo_becomes_a_plain_start_frame(self):
        self.assertEqual(rw.build_frame_images(["a.jpg"]), ["a.jpg"])

    def test_two_photos_pin_both_ends_of_the_generated_move(self):
        # This is the whole point of the pairing geometry: with a real
        # photograph at each end the model only invents the middle.
        self.assertEqual(
            rw.build_frame_images(["a.jpg", "b.jpg"]),
            ["a.jpg", {"image": "b.jpg", "frame": "last"}],
        )

    def test_a_shot_with_no_photograph_is_refused(self):
        with self.assertRaises(rw.RunwareError):
            rw.build_frame_images([])


class VideoTaskTests(unittest.TestCase):
    def task(self, **kw):
        return rw.build_video_task(prompt="p", photo_urls=["a.jpg"], **kw)

    def test_it_never_pays_for_audio_it_throws_away(self):
        # Vidu defaults audio ON. We lay our own music on a beat grid and
        # discard the generated track, so leaving this on is pure waste.
        self.assertIs(self.task()["providerSettings"]["vidu"]["audio"], False)

    def test_it_asks_for_more_source_than_the_edit_shows(self):
        # The surplus is what makes trimming and retiming free.
        self.assertGreaterEqual(self.task()["duration"], 3)

    def test_resolution_defaults_to_the_cheapest_tier(self):
        self.assertEqual(rw.resolution_size("540p"), (960, 540))
        self.assertEqual(self.task(resolution="1080p")["width"], 1920)

    def test_an_unknown_resolution_fails_loudly(self):
        with self.assertRaises(rw.RunwareError):
            rw.resolution_size("4k")

    def test_the_task_id_is_ours_so_a_retry_can_reuse_it(self):
        self.assertEqual(self.task(task_uuid="fixed")["taskUUID"], "fixed")


class CredentialTests(unittest.TestCase):
    def test_a_missing_api_key_is_refused_before_any_request(self):
        with patch.object(rw.settings, "RUNWARE_API_KEY", None):
            with self.assertRaises(rw.RunwareError):
                rw._headers()


if __name__ == "__main__":
    unittest.main()
