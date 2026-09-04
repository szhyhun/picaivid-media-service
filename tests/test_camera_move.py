"""Camera movement matches what the reference films actually do."""
from __future__ import annotations

import collections
import unittest

from app.pipeline.phase1_analyze.camera_move import (
    FREQ_PAN,
    FREQ_STATIC,
    PAN_RIGHT_SHARE,
    choose_move,
    required_headroom,
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
