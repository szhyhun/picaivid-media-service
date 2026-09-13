"""Cutting generated clips into beat slots."""
import unittest

from app.pipeline.phase4_assembly.assemble import BLEND_FRAMES, shot_filter


class ShotFilterTests(unittest.TestCase):
    def test_it_uses_the_stable_middle_at_native_speed(self):
        chain = shot_filter(source_seconds=3.0, trim_start=0.9, duration=1.2)
        self.assertIn("trim=start=0.900:duration=1.200", chain)
        self.assertNotIn("setpts=0.", chain)

    def test_the_middle_window_follows_the_music_grid(self):
        chain = shot_filter(source_seconds=3.0, trim_start=0.83335, duration=1.3333)
        self.assertIn("trim=start=0.833:duration=1.333", chain)

    def test_it_blends_real_frames_into_a_shutter(self):
        # Generated frames are individually sharp, and a run of sharp frames
        # reads as a slideshow however many there are. Blending pairs of real
        # frames synthesises the smear a physical shutter would leave.
        self.assertIn(f"tmix=frames={BLEND_FRAMES}",
                      shot_filter(source_seconds=3.0, trim_start=0.9, duration=1.2))

    def test_blending_can_be_turned_off(self):
        self.assertNotIn("tmix", shot_filter(source_seconds=3.0, trim_start=0.9,
                                             duration=1.2, blend=1))

    def test_it_always_lands_on_the_delivery_frame_rate(self):
        self.assertTrue(shot_filter(source_seconds=3.0, trim_start=0.9,
                                    duration=1.2).endswith("fps=30"))

    def test_a_trim_past_the_end_cannot_divide_by_zero(self):
        chain = shot_filter(source_seconds=3.0, trim_start=3.0, duration=1.2)
        self.assertIn("setpts=24.0000*PTS", chain)


if __name__ == "__main__":
    unittest.main()
