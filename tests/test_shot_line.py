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
    schedule_resolved_close_ups,
)
from app.pipeline.phase1_analyze.detail_crop import DetailCrop, ResolvedDetail
from app.pipeline.phase1_analyze.editorial import shot_budget


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


def resolved_details(shots):
    """Deterministic stand-in for crop validation; no model is needed here."""
    return {
        shot.index: ResolvedDetail(
            crop=DetailCrop(x=100, y=200, width=900, height=506,
                            score=0.1 + shot.index / 1000),
            source_size=(2048, 1365),
        )
        for shot in shots
    }


class HeroTests(unittest.TestCase):
    def test_hero_is_the_most_photographed_non_establishing_feature(self):
        """Aerials and facades establish the property; they are not its hero."""
        self.assertEqual(pick_hero(a_property()).room_id, "living")

    def test_establishing_feature_is_a_fallback_when_it_is_all_we_have(self):
        self.assertEqual(
            pick_hero([room("aerial", "aerial view", 3, start=100)]).room_id,
            "aerial",
        )

    def test_service_rooms_can_never_be_hero(self):
        rooms = [room("garage", "garage", 20, quality=1.0, start=100),
                 room("kitchen", "kitchen", 2, quality=0.3, start=200)]
        self.assertEqual(pick_hero(rooms).room_id, "kitchen")

    def test_no_features_means_no_hero(self):
        self.assertIsNone(pick_hero([room("g", "garage", 3, start=100)]))


class StructureTests(unittest.TestCase):
    def setUp(self):
        self.line = build_shot_line(a_property(), target_seconds=45, bpm=100)

    def test_the_film_opens_in_act_one_and_closes_on_an_exterior(self):
        self.assertEqual(self.line.shots[0].act, 1)
        self.assertIn(self.line.shots[-1].room_type, {
            "aerial view", "exterior front", "front yard", "building exterior",
        })

    def test_establishing_material_is_not_repeated_before_the_tour(self):
        """Aerial/facade images establish the property once, then bookend it."""
        body = self.line.shots[:-1]
        establishing = [s for s in body if s.room_type in {
            "aerial view", "exterior front", "front yard", "building exterior",
        }]
        self.assertLessEqual(len(establishing), 2)

    def test_acts_do_not_interleave(self):
        """Act 1 establishes, act 2 tours, act 3 closes -- in that order."""
        acts = [s.act for s in self.line.shots if not s.is_hero]
        self.assertEqual(acts, sorted(acts))

    def test_every_room_is_shown_in_one_continuous_run(self):
        """You look at a room, then you move on. Cutting away to a bedroom in
        the middle of the living room and back reads as a shuffle, not a tour.

        This replaces an earlier "hero is a motif" assumption: measuring the
        reference films showed no mid-film motif, and the owner confirmed a
        room's shots run consecutively.
        """
        # The final shot is a deliberate bookend -- all three reference films
        # close on an exterior or aerial they have already shown -- so it is
        # excluded from the continuity rule.
        body = self.line.shots[:-1]
        seen_runs = {}
        for i, shot in enumerate(body):
            if shot.is_close_up:
                continue          # a close-up belongs to the run it follows
            seen_runs.setdefault(shot.room_id, []).append(i)
        for room_id, positions in seen_runs.items():
            span = positions[-1] - positions[0] + 1
            self.assertLessEqual(
                span - len(positions), 2,
                f"{room_id} is split across the film: {positions}")

    def test_hero_does_not_take_over_the_film(self):
        """Allowing the closing bookend and its close-up on top of the cap."""
        hero_shots = sum(s.is_hero and not s.is_close_up
                         for s in self.line.shots[:-1])
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

    def test_every_shot_requests_two_seconds_of_source_footage(self):
        """Generation length is deliberately independent of the music tempo."""
        for bpm in (80, 100, 120):
            line = build_shot_line(a_property(), target_seconds=45, bpm=bpm)
            self.assertTrue(all(
                shot.generation_duration_seconds == 2.0
                for shot in line.shots
            ))

    def test_shots_are_contiguous_on_the_grid(self):
        line = build_shot_line(a_property(), target_seconds=45, bpm=100)
        for a, b in zip(line.shots, line.shots[1:]):
            self.assertAlmostEqual(a.start_seconds + a.duration_seconds,
                                   b.start_seconds, places=2)

    def test_close_up_target_is_an_upper_bound_not_a_forced_quota(self):
        """A film skips a detail rather than inserting generic material."""
        line = build_shot_line(a_property(), target_seconds=45, bpm=100)
        close_ups = sum(s.is_close_up for s in line.shots)
        wides = len(line.shots) - close_ups
        self.assertLessEqual(close_ups, round(wides * 0.25 / 0.75))

    def test_full_footage_fill_uses_the_requested_grid(self):
        """When there is enough distinct material, a 45s cut fills its grid."""
        rooms = [room(f"room-{i}", f"type-{i}", 1, start=1000 + i)
                 for i in range(25)]
        line = build_shot_line(
            rooms, target_seconds=45, bpm=100, include_close_ups=False)
        self.assertEqual(
            len(line.shots),
            round(shot_budget(45, 100) * 0.75),
        )
        self.assertAlmostEqual(
            line.duration,
            len(line.shots) * line.shots[0].duration_seconds,
            places=2,
        )


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


class CloseUpTests(unittest.TestCase):
    """Measured from the three reference films: about one shot in four is a
    close-up, recurring every two to three shots, thinning at both ends."""

    def setUp(self):
        wides = build_shot_line(
            a_property(), target_seconds=60, bpm=100, include_close_ups=False)
        self.line = type(wides)(
            shots=schedule_resolved_close_ups(
                wides.shots, resolved_details(wides.shots)),
            bpm=wides.bpm,
            hero_room_id=wides.hero_room_id,
            dropped_rooms=wides.dropped_rooms,
        )
        self.shots = self.line.shots

    def test_close_ups_never_exceed_a_quarter_of_the_film(self):
        share = sum(s.is_close_up for s in self.shots) / len(self.shots)
        self.assertLessEqual(share, 0.25)

    def test_a_close_up_is_cropped_from_the_shot_it_follows(self):
        """Cutting into the frame just seen reads as one continuous look; a
        detail from another room reads as an unrelated picture."""
        for prev, shot in zip(self.shots, self.shots[1:]):
            if shot.is_close_up:
                self.assertEqual(shot.room_id, prev.room_id)
                self.assertEqual(shot.photo_ids[0], prev.photo_ids[0])

    def test_close_ups_never_run_back_to_back(self):
        for a, b in zip(self.shots, self.shots[1:]):
            self.assertFalse(a.is_close_up and b.is_close_up)

    def test_the_approach_and_the_close_stay_wide(self):
        """The reference films open and end on wides."""
        self.assertFalse(self.shots[0].is_close_up)
        self.assertFalse(self.shots[1].is_close_up)
        self.assertFalse(self.shots[-1].is_close_up)

    def test_a_close_up_has_one_source_frame(self):
        for s in self.shots:
            if s.is_close_up:
                self.assertEqual(len(s.photo_ids), 1)
                self.assertEqual(s.kind, "single_image")
                self.assertIsNotNone(s.crop_box)
                self.assertIsNotNone(s.crop_score)

    def test_unvalidated_sources_never_become_close_ups(self):
        wides = build_shot_line(
            a_property(), target_seconds=60, bpm=100, include_close_ups=False)
        scheduled = schedule_resolved_close_ups(wides.shots, {})
        self.assertEqual(len(scheduled), len(wides.shots))
        self.assertFalse(any(s.is_close_up for s in scheduled))

    def test_bedrooms_and_bathrooms_are_not_detail_eligible(self):
        """A valid generic crop is still not enough for these room types."""
        details = [s for s in self.shots if s.is_close_up]
        self.assertTrue(details)
        self.assertFalse(any(s.room_type in {"bedroom", "bathroom"}
                             for s in details))

    def test_only_one_detail_is_selected_per_room(self):
        detail_rooms = [s.room_id for s in self.shots if s.is_close_up]
        self.assertEqual(len(detail_rooms), len(set(detail_rooms)))

    def test_shots_stay_contiguous_after_insertion(self):
        for a, b in zip(self.shots, self.shots[1:]):
            self.assertAlmostEqual(a.start_seconds + a.duration_seconds,
                                   b.start_seconds, places=2)

    def test_indices_are_renumbered_after_insertion(self):
        self.assertEqual([s.index for s in self.shots],
                         list(range(len(self.shots))))

    def test_short_edits_can_be_wide_only(self):
        """Details are optional; a compact edit may have no usable candidate."""
        tiny = build_shot_line(
            [room("bed", "bedroom", 4, start=100)],
            target_seconds=4,
            bpm=100,
            include_close_ups=False,
        )
        scheduled = schedule_resolved_close_ups(tiny.shots, resolved_details(tiny.shots))
        self.assertFalse(any(s.is_close_up for s in scheduled))
