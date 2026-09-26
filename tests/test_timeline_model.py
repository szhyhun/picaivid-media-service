"""Timing rules for the editable timeline.

The load-bearing property: shot length comes from the beat grid, not from the
generated video. That is what lets a shot be positioned before its pixels
exist, and what lets a change of music retime a whole film for free.
"""
import unittest

from app.db.models import Clip
from app.db.models.timeline import (
    PAIR_SOURCE_SECONDS,
    SINGLE_SOURCE_SECONDS,
    SOURCE_SECONDS,
    Timeline,
    TimelineClip,
)


def a_timeline(bpm=100.0, shots=4):
    tl = Timeline(beat_grid={"bpm": bpm, "beats_per_shot": 2})
    tl.timeline_clips = [
        TimelineClip(order_index=i, trim_start=0.0, shot_uid=f"s{i}")
        for i in range(shots)
    ]
    return tl


class GridTests(unittest.TestCase):
    def test_shot_length_is_two_beats_at_the_films_tempo(self):
        # Measured in reference films, and it never varies within a film.
        self.assertAlmostEqual(a_timeline(bpm=100).grid_seconds, 1.2)
        self.assertAlmostEqual(a_timeline(bpm=90).grid_seconds, 1.3333, places=3)
        self.assertAlmostEqual(a_timeline(bpm=113).grid_seconds, 1.0619, places=3)

    def test_every_usable_tempo_fits_inside_what_we_buy(self):
        # If this ever failed, changing the music would mean regenerating.
        for bpm in (60, 80, 90, 100, 113, 128, 140):
            self.assertLessEqual(a_timeline(bpm=bpm).grid_seconds, SOURCE_SECONDS)


class RetimeTests(unittest.TestCase):
    def test_changing_tempo_retimes_the_whole_film(self):
        tl = a_timeline(bpm=100, shots=4)
        self.assertAlmostEqual(tl.retime(), 4.8)
        tl.beat_grid = {**tl.beat_grid, "bpm": 90}
        self.assertAlmostEqual(tl.retime(), 5.3332, places=3)

    def test_in_times_are_derived_from_order_not_stored_as_truth(self):
        tl = a_timeline(bpm=100, shots=3)
        tl.retime()
        self.assertEqual([c.in_time for c in tl.ordered_clips()], [0.0, 1.2, 2.4])
        # Reversing the order must renumber, not preserve stale values.
        for c in tl.timeline_clips:
            c.order_index = 2 - c.order_index
        tl.retime()
        self.assertEqual([c.in_time for c in tl.ordered_clips()], [0.0, 1.2, 2.4])

    def test_a_pinned_shot_ignores_the_grid_and_its_neighbours_do_not(self):
        tl = a_timeline(bpm=100, shots=3)
        tl.ordered_clips()[1].duration_override = 2.4
        self.assertAlmostEqual(tl.retime(), 1.2 + 2.4 + 1.2)


class TrimTests(unittest.TestCase):
    def test_a_five_second_pair_has_headroom_for_a_central_two_beat_window(self):
        trim_start = (PAIR_SOURCE_SECONDS - 1.2) / 2.0
        self.assertAlmostEqual(trim_start, 1.9)
        clip = TimelineClip(order_index=0, kind="pair_keyframes",
                            trim_start=trim_start)
        self.assertAlmostEqual(clip.duration_for(1.2), 1.2)

    def test_duration_is_clamped_to_the_source_we_actually_bought(self):
        clip = TimelineClip(order_index=0, kind="pair_keyframes", trim_start=4.5)
        self.assertAlmostEqual(clip.max_duration(), 0.5)
        self.assertAlmostEqual(clip.duration_for(1.2), 0.5)

    def test_an_untrimmed_shot_has_the_whole_source_available(self):
        self.assertAlmostEqual(
            TimelineClip(order_index=0, kind="pair_keyframes",
                         trim_start=0.0).max_duration(),
            PAIR_SOURCE_SECONDS)

    def test_a_single_is_bought_shorter_than_a_pair(self):
        # A single invents everything after its one frame, so we buy less of
        # it. Its trim ceiling has to follow, or the editor would offer a
        # window that runs off the end of the clip.
        single = TimelineClip(order_index=0, kind="single_image", trim_start=0.0)
        pair = TimelineClip(order_index=1, kind="pair_keyframes", trim_start=0.0)
        self.assertAlmostEqual(single.max_duration(), SINGLE_SOURCE_SECONDS)
        self.assertAlmostEqual(pair.max_duration(), PAIR_SOURCE_SECONDS)
        self.assertLess(single.max_duration(), pair.max_duration())

    def test_every_usable_tempo_still_fits_inside_a_single(self):
        # The shorter buy is only safe while a two-beat window still fits in
        # it; below ~40 BPM it would not, and retiming would need new video.
        for bpm in (60, 80, 90, 100, 113, 128, 140):
            self.assertLessEqual(a_timeline(bpm=bpm).grid_seconds,
                                 SINGLE_SOURCE_SECONDS)


class StateTests(unittest.TestCase):
    def test_a_shot_exists_before_its_video_does(self):
        # This is what lets a photo be dropped onto a live timeline: the slot
        # is real and correctly timed while the clip is still rendering.
        self.assertEqual(TimelineClip(order_index=0).state, "planned")

    def test_state_is_read_from_the_clip_rather_than_duplicated(self):
        for status, expected in (("pending", "generating"), ("rendering", "generating"),
                                 ("complete", "ready"), ("failed", "failed")):
            shot = TimelineClip(order_index=0, clip_id=1)
            # A real Clip: the relationship is typed, and the point of the test
            # is that state is read from it rather than stored twice.
            shot.clip = Clip(status=status)
            self.assertEqual(shot.state, expected)


if __name__ == "__main__":
    unittest.main()
