"""Editorial eligibility and beat selection.

The film wants 20-30 beats out of 30-60 photos, so this is a selection problem,
not a coverage problem. These tests pin the two judgements that decide what a
viewer actually sees.
"""
from __future__ import annotations

import unittest

from app.models.openclip import ROOM_TYPES
from app.pipeline.phase1_analyze.editorial import (
    EDITORIAL_TIER,
    SHOT_BEATS,
    Tier,
    beat_kind,
    is_eligible,
    room_budget,
    shot_budget,
    shot_seconds,
    tier_for,
)


class TierTests(unittest.TestCase):
    def test_every_room_type_has_an_explicit_tier(self):
        """A new ROOM_TYPES entry must be given a tier deliberately, not fall
        through to a default that quietly puts a garage in the film."""
        self.assertEqual(set(ROOM_TYPES), set(EDITORIAL_TIER))

    def test_service_rooms_are_excluded_by_default(self):
        for room in ("garage", "storage", "laundry room", "basement", "attic"):
            with self.subTest(room=room):
                self.assertIs(tier_for(room), Tier.SERVICE)
                self.assertFalse(is_eligible(room))

    def test_selling_features_are_eligible(self):
        for room in ("kitchen", "living room", "bedroom", "aerial view", "patio"):
            with self.subTest(room=room):
                self.assertIs(tier_for(room), Tier.FEATURE)
                self.assertTrue(is_eligible(room))

    def test_apartment_amenities_are_a_feature_not_a_service_space(self):
        """For a condo the gym and lounge are much of the pitch."""
        self.assertIs(tier_for("amenities"), Tier.FEATURE)
        self.assertTrue(is_eligible("amenities"))

    def test_circulation_is_connective(self):
        for room in ("entrance", "hallway", "shared space"):
            with self.subTest(room=room):
                self.assertIs(tier_for(room), Tier.CONNECTIVE)

    def test_service_rooms_return_when_nothing_else_exists(self):
        """A studio with a kitchen, a bath and a laundry closet still needs a film."""
        self.assertTrue(is_eligible("laundry room", allow_service=True))

    def test_unknown_room_type_is_connective_not_feature(self):
        """An unrecognised label must not earn hero screen time by accident."""
        self.assertIs(tier_for("wine cellar"), Tier.CONNECTIVE)
        self.assertIs(tier_for(None), Tier.CONNECTIVE)


class BeatKindTests(unittest.TestCase):
    """A single good photo beats omitting the room. Measured across six labelled
    listings, 40% of same-room pairs have no geometric overlap and cannot be
    linked by any merge policy, so requiring pairs would drop good rooms."""

    def test_verified_pair_earns_real_camera_motion(self):
        self.assertEqual(beat_kind(2, True), "multi_view_move")

    def test_unverified_pair_cuts_rather_than_interpolating(self):
        self.assertEqual(beat_kind(2, False), "sequence_cut")

    def test_single_photo_still_earns_a_beat(self):
        self.assertEqual(beat_kind(1, False), "single_image_move")

    def test_single_photo_is_never_a_multi_view_move(self):
        self.assertEqual(beat_kind(1, True), "single_image_move")


class RhythmTests(unittest.TestCase):
    """Measured from three professionally produced listing films, all of which
    cut on a metronomic grid of exactly two beats per shot."""

    def test_shot_length_is_two_beats(self):
        self.assertEqual(SHOT_BEATS, 2)
        self.assertAlmostEqual(shot_seconds(120), 1.0, places=3)
        self.assertAlmostEqual(shot_seconds(100), 1.200, places=3)
        self.assertAlmostEqual(shot_seconds(90), 1.333, places=3)
        self.assertAlmostEqual(shot_seconds(113), 1.062, places=3)

    def test_reproduces_the_reference_films_within_one_shot(self):
        for seconds, bpm, observed in [(43.0, 113, 35), (70.7, 90, 48), (59.2, 100, 46)]:
            with self.subTest(bpm=bpm):
                self.assertLessEqual(abs(shot_budget(seconds, bpm) - observed), 1)

    def test_tempo_not_room_count_drives_the_edit(self):
        """The same duration at a faster tempo buys more shots."""
        self.assertGreater(shot_budget(60, 120), shot_budget(60, 80))

    def test_room_budget_follows_from_the_shot_budget(self):
        self.assertLess(room_budget(60, 100), shot_budget(60, 100))
        self.assertGreaterEqual(room_budget(60, 100), 1)

    def test_guard_rails_hold(self):
        self.assertEqual(shot_budget(1, 100), 8)
        self.assertEqual(shot_budget(3600, 100), 60)


if __name__ == "__main__":
    unittest.main()
