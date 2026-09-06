"""Cutting generated clips into beat slots."""
import unittest

from app.pipeline.phase4_assembly.assemble import BLEND_FRAMES, shot_filter


class ShotFilterTests(unittest.TestCase):
    def test_it_squeezes_the_source_into_the_beat_slot(self):
        # 3s bought, 1.2s shown at 100 BPM.
        self.assertIn("setpts=0.4000*PTS",
                      shot_filter(source_seconds=3.0, trim_start=0.0, duration=1.2))

    def test_the_speed_is_computed_per_clip_not_assumed(self):
        # A film retimed to 90 BPM changes this without regenerating anything,
        # so a hardcoded factor would silently desync the whole edit.
        slower = shot_filter(source_seconds=3.0, trim_start=0.0, duration=1.3333)
        self.assertIn("setpts=0.4444*PTS", slower)

    def test_a_trim_shortens_the_usable_source(self):
        # Starting 0.9s in leaves 2.1s to fit the slot, not 3s.
        self.assertIn("trim=start=0.900",
                      shot_filter(source_seconds=3.0, trim_start=0.9, duration=1.3333))
        self.assertIn("setpts=0.6349*PTS",
                      shot_filter(source_seconds=3.0, trim_start=0.9, duration=1.3333))

    def test_it_blends_real_frames_into_a_shutter(self):
        # Generated frames are individually sharp, and a run of sharp frames
        # reads as a slideshow however many there are. Blending pairs of real
        # frames synthesises the smear a physical shutter would leave.
        self.assertIn(f"tmix=frames={BLEND_FRAMES}",
                      shot_filter(source_seconds=3.0, trim_start=0.0, duration=1.2))

    def test_blending_can_be_turned_off(self):
        self.assertNotIn("tmix", shot_filter(source_seconds=3.0, trim_start=0.0,
                                             duration=1.2, blend=1))

    def test_it_always_lands_on_the_delivery_frame_rate(self):
        self.assertTrue(shot_filter(source_seconds=3.0, trim_start=0.0,
                                    duration=1.2).endswith("fps=30"))

    def test_a_trim_past_the_end_cannot_divide_by_zero(self):
        chain = shot_filter(source_seconds=3.0, trim_start=3.0, duration=1.2)
        self.assertIn("setpts=", chain)


if __name__ == "__main__":
    unittest.main()
