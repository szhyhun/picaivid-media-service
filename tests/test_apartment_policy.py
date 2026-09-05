import unittest

from app.pipeline.phase1_analyze.editorial import is_eligible
from app.pipeline.phase1_analyze.shot_line import RoomCandidate, space_scope, story_position


class ApartmentPolicyTests(unittest.TestCase):
    def test_function_does_not_prove_shared_use(self):
        for label in ("gym", "hallway", "pool", "kitchen", "amenities"):
            self.assertEqual(space_scope(label), "unknown")

    def test_shared_spaces_follow_home_and_precede_outdoors(self):
        def position(label):
            return story_position(RoomCandidate(label, label, [1]))
        for label in ("building lobby", "elevator lobby", "shared space"):
            self.assertEqual(space_scope(label), "shared")
            self.assertGreater(position(label), position("bathroom"))
            self.assertLess(position(label), position("patio"))

    def test_wardrobes_do_not_earn_default_coverage(self):
        self.assertFalse(is_eligible("walk-in closet"))
        self.assertTrue(is_eligible("gym"))
