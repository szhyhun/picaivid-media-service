"""The ordered shot line: selection, structure, and keyframe preference.

Structure is measured from three professionally produced listing films; see
VIDEO_PRODUCTION_PLAN.md. These tests pin the behaviours that make the output a
film rather than a slideshow.
"""
from __future__ import annotations

import unittest

from app.pipeline.phase1_analyze.shot_line import (
    MAX_HERO_SHOTS,
    MAX_SERVICE_SHOTS,
    RoomCandidate,
    build_shot_line,
    pick_hero,
)


def room(room_id, room_type, n_photos, *, quality=0.5, pairs=0, start=100):
    ids = list(range(start, start + n_photos))
    cand = RoomCandidate(room_id=room_id, room_type=room_type,
                         photo_ids=ids, quality=quality)
    cand.pairs = [(ids[2 * i], ids[2 * i + 1], 0.9 - 0.1 * i) for i in range(pairs)]
    return cand


def a_property():
    """A listing with the mix a real one has: some paired rooms, some singles,
    some service rooms."""
    return [
        room("aerial", "aerial view", 6, quality=0.9, pairs=2, start=100),
        room("front", "exterior front", 2, quality=0.7, pairs=1, start=200),
        room("entry", "entrance", 1, quality=0.4, start=300),
        room("living", "living room", 4, quality=0.8, pairs=2, start=400),
        room("kitchen", "kitchen", 3, quality=0.8, pairs=1, start=500),
        room("bed1", "bedroom", 2, quality=0.6, pairs=1, start=600),
        room("bath1", "bathroom", 2, quality=0.5, start=700),
        room("garage", "garage", 2, quality=0.5, start=800),
        room("laundry", "laundry room", 1, quality=0.3, start=900),
        room("storage", "storage", 1, quality=0.2, start=950),
    ]


class HeroTests(unittest.TestCase):
    def test_hero_is_the_most_photographed_feature(self):
        """A photographer taking six photos of one view is telling us something."""
        self.assertEqual(pick_hero(a_property()).room_id, "aerial")

    def test_service_rooms_can_never_be_hero(self):
        rooms = [room("garage", "garage", 20, quality=1.0, start=100),
                 room("kitchen", "kitchen", 2, quality=0.3, start=200)]
        self.assertEqual(pick_hero(rooms).room_id, "kitchen")

    def test_no_features_means_no_hero(self):
        self.assertIsNone(pick_hero([room("g", "garage", 3, start=100)]))


class StructureTests(unittest.TestCase):
    def setUp(self):
        self.line = build_shot_line(a_property(), target_seconds=45, bpm=100)

    def test_the_film_opens_in_act_one_and_closes_on_the_hero(self):
        self.assertEqual(self.line.shots[0].act, 1)
        self.assertTrue(self.line.shots[-1].is_hero)

    def test_acts_do_not_interleave(self):
        """Act 1 establishes, act 2 tours, act 3 closes -- in that order."""
        acts = [s.act for s in self.line.shots if not s.is_hero]
        self.assertEqual(acts, sorted(acts))

    def test_the_hero_is_a_motif_not_a_block(self):
        """Wallace Dr returned to its view at 4s, 18s and the close. Clustering
        the hero's shots would read as a slideshow of one room."""
        idx = [s.index for s in self.line.shots if s.is_hero]
        self.assertGreater(len(idx), 1)
        gaps = [b - a for a, b in zip(idx, idx[1:])]
        self.assertTrue(all(g > 1 for g in gaps), f"hero shots adjacent: {idx}")

    def test_hero_does_not_take_over_the_film(self):
        hero_shots = sum(s.is_hero for s in self.line.shots)
        self.assertLessEqual(hero_shots, MAX_HERO_SHOTS)

    def test_service_rooms_get_one_brief_mention_between_them(self):
        """The reference film gave the garage 1.5s -- one beat for all utility
        space, not one beat per utility room."""
        service = [s for s in self.line.shots
                   if s.room_type in ("garage", "laundry room", "storage")]
        self.assertLessEqual(len(service), MAX_SERVICE_SHOTS)

    def test_service_appears_late_if_at_all(self):
        for s in self.line.shots:
            if s.room_type in ("garage", "laundry room", "storage"):
                self.assertGreater(s.index, len(self.line.shots) * 0.6)


class RhythmTests(unittest.TestCase):
    def test_every_shot_is_the_same_length(self):
        """The reference films never vary shot length within a film."""
        line = build_shot_line(a_property(), target_seconds=45, bpm=100)
        self.assertEqual(len({s.duration_seconds for s in line.shots}), 1)

    def test_shot_length_follows_the_tempo(self):
        slow = build_shot_line(a_property(), target_seconds=45, bpm=80)
        fast = build_shot_line(a_property(), target_seconds=45, bpm=120)
        self.assertGreater(slow.shots[0].duration_seconds,
                           fast.shots[0].duration_seconds)

    def test_shots_are_contiguous_on_the_grid(self):
        line = build_shot_line(a_property(), target_seconds=45, bpm=100)
        for a, b in zip(line.shots, line.shots[1:]):
            self.assertAlmostEqual(a.start_seconds + a.duration_seconds,
                                   b.start_seconds, places=2)


class KeyframeTests(unittest.TestCase):
    """Two real photographs anchor both ends of a generated clip; a single photo
    anchors only the first frame. Pairs are strictly preferable."""

    def test_paired_rooms_use_keyframes(self):
        line = build_shot_line(a_property(), target_seconds=45, bpm=100)
        living = [s for s in line.shots if s.room_id == "living"]
        self.assertTrue(any(s.uses_keyframes for s in living))

    def test_a_pair_is_never_reused_within_a_room(self):
        line = build_shot_line(a_property(), target_seconds=45, bpm=100)
        seen = set()
        for s in line.shots:
            if s.uses_keyframes:
                key = (s.room_id, tuple(s.photo_ids))
                self.assertNotIn(key, seen)
                seen.add(key)

    def test_rooms_without_pairs_fall_back_to_single_images(self):
        line = build_shot_line([room("bath", "bathroom", 2, start=100)],
                               target_seconds=20, bpm=100)
        self.assertTrue(all(s.kind == "single_image" for s in line.shots))

    def test_keyframe_fraction_is_reported(self):
        line = build_shot_line(a_property(), target_seconds=45, bpm=100)
        self.assertGreater(line.keyframe_fraction(), 0.0)
        self.assertLessEqual(line.keyframe_fraction(), 1.0)


class SelectionTests(unittest.TestCase):
    def test_a_short_film_drops_rooms_rather_than_compressing_shots(self):
        """Shot length is set by the music, so a 15s film shows fewer rooms --
        it does not show every room faster."""
        short = build_shot_line(a_property(), target_seconds=15, bpm=100)
        long = build_shot_line(a_property(), target_seconds=60, bpm=100)
        self.assertEqual(short.shots[0].duration_seconds,
                         long.shots[0].duration_seconds)
        self.assertLess(len(short.shots), len(long.shots))

    def test_rooms_with_no_photos_are_never_scheduled(self):
        rooms = a_property() + [RoomCandidate("empty", "kitchen", [])]
        line = build_shot_line(rooms, target_seconds=45, bpm=100)
        self.assertNotIn("empty", {s.room_id for s in line.shots})

    def test_an_empty_property_produces_an_empty_line(self):
        line = build_shot_line([], target_seconds=45, bpm=100)
        self.assertEqual(line.shots, [])
        self.assertEqual(line.duration, 0)

    def test_output_is_deterministic(self):
        """Two runs of one listing must produce the same film."""
        a = build_shot_line(a_property(), target_seconds=45, bpm=100)
        b = build_shot_line(a_property(), target_seconds=45, bpm=100)
        self.assertEqual([(s.room_id, tuple(s.photo_ids), s.move.label) for s in a.shots],
                         [(s.room_id, tuple(s.photo_ids), s.move.label) for s in b.shots])


if __name__ == "__main__":
    unittest.main()
