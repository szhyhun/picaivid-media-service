"""Camera movement matches what the reference films actually do."""
from __future__ import annotations

import collections
import unittest

import app.pipeline.phase1_analyze.camera_move as camera_move

from app.pipeline.phase1_analyze.camera_move import (
    FREQ_PAN,
    FREQ_STATIC,
    PAN_RIGHT_SHARE,
    choose_move,
    required_headroom,
    reverse_label,
    reverse_vector,
)


class DistributionTests(unittest.TestCase):
    """Sampled over many seeds, the generator should reproduce the measured
    frequencies -- 87% pan, 66% push in, only 4% static."""

    def setUp(self):
        self.moves = [choose_move(f"photo-{i}") for i in range(4000)]

    def test_almost_nothing_is_static(self):
        static = sum(m.is_static() for m in self.moves) / len(self.moves)
        self.assertAlmostEqual(static, FREQ_STATIC, delta=0.02)

    def test_most_shots_pan(self):
        moving = [m for m in self.moves if not m.is_static()]
        panning = sum(1 for m in moving if m.pan) / len(moving)
        self.assertAlmostEqual(panning, FREQ_PAN, delta=0.03)

    def test_pans_favour_the_right(self):
        pans = [m.pan for m in self.moves if m.pan]
        right = sum(1 for p in pans if p > 0) / len(pans)
        self.assertAlmostEqual(right, PAN_RIGHT_SHARE, delta=0.04)

    def test_push_in_dominates_pull_out(self):
        moving = [m for m in self.moves if not m.is_static()]
        pushes = sum(1 for m in moving if m.zoom > 0)
        pulls = sum(1 for m in moving if m.zoom < 0)
        self.assertGreater(pushes, 5 * pulls)

    def test_movement_compounds(self):
        """The canonical shot is a push plus a pan, not one axis alone."""
        moving = [m for m in self.moves if not m.is_static()]
        multi = sum(1 for m in moving if bool(m.zoom) + bool(m.pan) + bool(m.tilt) >= 2)
        self.assertGreater(multi / len(moving), 0.8)


class DeterminismTests(unittest.TestCase):
    def test_same_photo_always_gets_the_same_move(self):
        """Two runs of one listing must produce the same plan, or review and
        regression testing are impossible."""
        self.assertEqual(choose_move("abc"), choose_move("abc"))

    def test_different_photos_get_different_moves(self):
        labels = {choose_move(f"p{i}").label for i in range(50)}
        self.assertGreater(len(labels), 3)


class EstablishingTests(unittest.TestCase):
    def test_establishing_shots_pull_out(self):
        """In the films the pull out is reserved for exteriors and aerials,
        where opening the frame reveals the setting."""
        for i in range(20):
            move = choose_move(f"aerial-{i}", is_establishing=True)
            if not move.is_static():
                self.assertLess(move.zoom, 0)


class HeadroomTests(unittest.TestCase):
    def test_a_move_needs_more_pixels_than_the_output_frame(self):
        move = choose_move("x")
        if not move.is_static():
            self.assertGreater(required_headroom(move, 1.2), 1.0)

    def test_static_hold_needs_no_headroom(self):
        static = next(choose_move(f"s{i}") for i in range(500)
                      if choose_move(f"s{i}").is_static())
        self.assertEqual(required_headroom(static, 1.2), 1.0)

    def test_longer_shots_need_more_headroom(self):
        move = choose_move("y")
        if not move.is_static():
            self.assertGreater(required_headroom(move, 3.0),
                               required_headroom(move, 1.0))


if __name__ == "__main__":
    unittest.main()


class ReverseTests(unittest.TestCase):
    """Swapping which photograph is the start frame reverses the move."""

    def test_it_inverts_every_atom_of_a_compound_move(self):
        self.assertEqual(
            reverse_label("push in + pan right + tilt up"),
            "pull out + pan left + tilt down",
        )

    def test_a_static_hold_has_no_direction_to_reverse(self):
        self.assertEqual(reverse_label("static hold"), "static hold")

    def test_reversing_twice_returns_the_original(self):
        for label in ("push in + pan left", "pull out + tilt down", "pan right"):
            self.assertEqual(
                reverse_label(reverse_label(label)), label)

    def test_it_negates_the_vector(self):
        out = reverse_vector(
            {"zoom": 0.067, "pan": 0.046, "tilt": -0.05, "roll_deg": 1.36})
        self.assertAlmostEqual(out["zoom"], -0.067)
        self.assertAlmostEqual(out["pan"], -0.046)
        self.assertAlmostEqual(out["tilt"], 0.05)

    def test_a_missing_vector_reverses_to_zero_rather_than_failing(self):
        self.assertEqual(reverse_vector(None),
                         {"zoom": 0.0, "pan": 0.0, "tilt": 0.0, "roll_deg": 0.0})




class ComposedMoveTests(unittest.TestCase):
    """Moves are built from independent axes, not picked from a flat list."""

    def test_it_splits_a_compound_label_into_axes(self):
        self.assertEqual(
            camera_move.axes_of("push in + pan right + tilt up"),
            {"zoom": "push in", "pan": "pan right", "tilt": "tilt up"},
        )

    def test_removing_an_axis_leaves_the_others_intact(self):
        axes = camera_move.axes_of("push in + pan right + tilt up")
        axes["pan"] = None
        self.assertEqual(camera_move.compose_label(axes), "push in + tilt up")

    def test_it_always_composes_in_zoom_pan_tilt_order(self):
        # Order is how the prompt decides its lead clause, so it cannot depend
        # on the order a customer happened to click the buttons in.
        self.assertEqual(
            camera_move.compose_label(
                {"tilt": "tilt up", "pan": "pan left", "zoom": "pull out"}),
            "pull out + pan left + tilt up",
        )

    def test_removing_every_axis_is_a_hold(self):
        self.assertEqual(
            camera_move.compose_label({"zoom": None, "pan": None, "tilt": None}),
            "static hold",
        )

    def test_unknown_atoms_are_dropped_rather_than_raising(self):
        # Labels are persisted data; a vocabulary change must not make an old
        # row unreadable.
        self.assertEqual(
            camera_move.axes_of("push in + dolly zoom"),
            {"zoom": "push in", "pan": None, "tilt": None},
        )

    def test_an_echoed_move_normalises_to_the_same_label(self):
        # The editor sends every shot's move back on any edit -- a drag, a
        # trim, a delete. If a re-sent move did not compare equal to the one
        # the shot already had, a single drag would lock every move on the
        # timeline and freeze the planner out of the whole film.
        for echoed in ("pull out + pan right", "pan right + pull out"):
            self.assertEqual(
                camera_move.compose_label(camera_move.axes_of(echoed)),
                "pull out + pan right",
            )
