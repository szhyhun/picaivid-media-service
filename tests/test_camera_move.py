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


class SelectableMoveTests(unittest.TestCase):
    """The moves a customer can choose, and what choosing one does."""

    def test_choosing_a_move_keeps_the_axes_it_did_not_touch(self):
        # Over 80% of reference shots move on two or more axes, so collapsing a
        # choice down to one would make the film flatter than its source.
        self.assertEqual(
            camera_move.with_primary("push in + pan right + tilt up", "pull out"),
            "pull out + pan right + tilt up",
        )

    def test_choosing_a_move_replaces_only_its_own_axis(self):
        self.assertEqual(
            camera_move.with_primary("push in + pan right", "pan left"),
            "pan left + push in",
        )

    def test_a_chosen_move_outranks_the_zoom_first_reading(self):
        # The select's whole purpose is overriding what was measured. Reading
        # a chosen "pan left" back as the shot's old "pull out" would make the
        # control appear to do nothing.
        label = camera_move.with_primary("pull out + pan right", "pan left")
        self.assertEqual(camera_move.primary_of(label, chosen=True), "pan left")
        self.assertEqual(camera_move.primary_of(label), "pull out")

    def test_holding_still_clears_everything(self):
        # A hold is the absence of movement; it cannot keep a pan.
        self.assertEqual(
            camera_move.with_primary("pan right + tilt up", "static hold"),
            "static hold",
        )

    def test_it_reads_the_primary_axis_back_out_of_a_compound_label(self):
        self.assertEqual(camera_move.primary_of("push in + pan right"), "push in")
        self.assertEqual(camera_move.primary_of("pan left + tilt down"), "pan left")
        self.assertEqual(camera_move.primary_of("static hold"), "static hold")
        self.assertEqual(camera_move.primary_of(None), "static hold")

    def test_a_chosen_move_produces_a_vector_at_the_measured_rates(self):
        # A chosen move has no seed behind it, and the renderer reads the
        # vector rather than the words.
        vector = camera_move.vector_from_label("pull out + pan left")
        self.assertAlmostEqual(vector["zoom"], -camera_move.ZOOM_OUT_RATE)
        self.assertAlmostEqual(vector["pan"], -camera_move.PAN_RATE)
        self.assertAlmostEqual(vector["tilt"], 0.0)

    def test_a_hold_has_no_motion_on_any_axis(self):
        self.assertEqual(
            camera_move.vector_from_label("static hold"),
            {"zoom": 0.0, "pan": 0.0, "tilt": 0.0, "roll_deg": 0.0},
        )

    def test_roll_is_carried_through_rather_than_invented(self):
        self.assertAlmostEqual(
            camera_move.vector_from_label("push in", roll_deg=1.36)["roll_deg"], 1.36)
