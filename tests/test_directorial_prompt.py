"""Rendering the shot line as a director's script for one-pass generation.

A one-pass model generates a whole multi-shot sequence from references plus a
prompt. The prompt is where the pipeline's ordering work goes -- it instructs
the generator rather than surrendering to it.
"""
from __future__ import annotations

import unittest

from app.pipeline.phase1_analyze.directorial_prompt import (
    build_script,
    split_for_generation,
)
from app.pipeline.phase1_analyze.shot_line import build_shot_line
from tests.test_shot_line import a_property


class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.line = build_shot_line(a_property(), target_seconds=30, bpm=100)
        self.script = build_script(self.line.shots, bpm=100)

    def test_every_shot_is_numbered_and_timed(self):
        for shot in self.line.shots:
            self.assertIn(f"Shot {shot.index + 1} (", self.script.prompt)

    def test_the_script_states_the_shot_length_not_the_beat(self):
        """A generator has no use for tempo, only for how long a shot holds."""
        expected = f"{self.line.shots[0].duration_seconds:.2f} seconds long"
        self.assertIn(expected, self.script.prompt)

    def test_it_forbids_the_transitions_the_films_never_use(self):
        for banned in ("No dissolves", "no fades", "no speed ramps"):
            self.assertIn(banned, self.script.prompt)

    def test_camera_moves_are_carried_into_direction(self):
        self.assertIn("push in", self.script.prompt)

    def test_room_ids_never_leak_into_the_prompt(self):
        """Instance ids like "bedroom-a" disambiguate for us, not for a model."""
        for suffix in ("bedroom-a", "living-room-staged", "component-"):
            self.assertNotIn(suffix, self.script.prompt)

    def test_references_are_deduplicated(self):
        ids = self.script.reference_photo_ids
        self.assertEqual(len(ids), len(set(ids)))

    def test_an_empty_line_produces_an_empty_script(self):
        empty = build_script([], bpm=100)
        self.assertEqual(empty.prompt, "")
        self.assertEqual(empty.shot_count, 0)


class SplitTests(unittest.TestCase):
    """One-pass models cap at 30s, so longer films are a few passes."""

    def test_no_segment_exceeds_the_single_pass_limit(self):
        line = build_shot_line(a_property(), target_seconds=90, bpm=100)
        for segment in split_for_generation(line.shots, max_seconds=30):
            self.assertLessEqual(
                sum(s.duration_seconds for s in segment), 30.0 + 1e-6)

    def test_every_shot_survives_the_split(self):
        line = build_shot_line(a_property(), target_seconds=90, bpm=100)
        segments = split_for_generation(line.shots, max_seconds=30)
        flat = [s for seg in segments for s in seg]
        self.assertEqual([s.index for s in flat],
                         [s.index for s in line.shots])

    def test_a_short_film_needs_one_pass(self):
        line = build_shot_line(a_property(), target_seconds=20, bpm=100)
        self.assertEqual(len(split_for_generation(line.shots, max_seconds=30)), 1)

    def test_seams_prefer_act_boundaries(self):
        """An act is a coherent stretch of story, so a seam there is the least
        visible place to put one."""
        line = build_shot_line(a_property(), target_seconds=60, bpm=100)
        segments = split_for_generation(line.shots, max_seconds=30)
        if len(segments) > 1:
            for earlier, later in zip(segments, segments[1:]):
                self.assertLessEqual(earlier[-1].act, later[0].act)

    def test_an_empty_line_produces_no_segments(self):
        self.assertEqual(split_for_generation([]), [])


if __name__ == "__main__":
    unittest.main()
