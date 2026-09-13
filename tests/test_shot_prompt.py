"""Per-shot prompts, against the rules measured in REFERENCE_FILM_FINDINGS.md."""
import unittest

from app.pipeline.phase1_analyze.directorial_prompt import (
    build_shot_prompt,
    shot_class,
)


class _Shot:
    def __init__(self, **kw):
        self.__dict__.update({
            "room_type": "living room", "kind": "single_image",
            "is_close_up": False, "act": 2, "move": "push in",
            "detail_subject": None, "prompt_override": None, **kw})


class ShotClassTests(unittest.TestCase):
    def test_it_reads_the_seven_directorial_classes(self):
        self.assertEqual(shot_class("aerial view", act=1), "aerial")
        self.assertEqual(shot_class("front yard", act=1), "approach")
        self.assertEqual(shot_class("entrance"), "threshold")
        self.assertEqual(shot_class("kitchen"), "interior")
        self.assertEqual(shot_class("backyard", act=3), "outdoor")
        self.assertEqual(shot_class("living room", is_close_up=True), "detail")

    def test_an_exterior_at_the_end_is_a_closing_shot_not_an_approach(self):
        # All three reference films end outdoors or on an exterior; two end on
        # the front exterior as the literal last shot.
        self.assertEqual(shot_class("exterior front", act=1), "approach")
        self.assertEqual(shot_class("exterior front", act=1, is_last=True), "closing")


class PromptTests(unittest.TestCase):
    def test_a_pair_shot_states_direction_by_reference_order(self):
        # The frameImages payload binds reference one to the start and reference
        # two to the end. The prompt must say the same thing without inventing
        # a textual left/right direction that geometry may contradict.
        prompt = build_shot_prompt(_Shot(kind="pair_keyframes"))
        self.assertIn(
            "from the first reference frame to the second reference frame",
            prompt,
        )
        self.assertIn("constant speed", prompt)
        self.assertNotIn("pan right", prompt)

    def test_a_single_shot_uses_one_simple_axis(self):
        # With one photograph nothing else determines where the camera goes, so
        # the provider gets a small, unambiguous motion vocabulary.
        prompt = build_shot_prompt(_Shot(move="pan right"))
        self.assertIn("straight horizontal pan to the right", prompt)
        for banned in ("tilt", "roll", "orbit", "parallax"):
            self.assertNotIn(banned, prompt.lower())

    def test_opening_and_closing_single_shots_have_bookend_moves(self):
        opening = build_shot_prompt(_Shot(move="pan left"), is_first=True)
        closing = build_shot_prompt(_Shot(move="push in"), is_last=True)
        self.assertIn("straight push in", opening)
        self.assertIn("straight pull back", closing)

    def test_it_never_describes_the_room(self):
        # The model already has the pixels; describing the building invites it
        # to invent a different one.
        prompt = build_shot_prompt(_Shot(room_type="kitchen")).lower()
        self.assertNotIn("kitchen", prompt)

    def test_every_prompt_forbids_the_things_the_films_never_do(self):
        prompt = build_shot_prompt(_Shot())
        for rule in ("Static architecture", "no easing", "No people",
                     "No cars moving", "No text", "No logos",
                     "No cuts, no dissolves", "no speed changes"):
            self.assertIn(rule, prompt)

    def test_single_interior_prompt_does_not_request_angled_motion(self):
        prompt = build_shot_prompt(
            _Shot(room_type="living room", move="push in + pan right + tilt up"))
        self.assertIn("straight push in", prompt)
        self.assertNotIn("tilt", prompt.lower())
        self.assertNotIn("roll", prompt.lower())

    def test_source_window_keeps_the_same_slow_prompt_at_native_speed(self):
        # The editor now trims a native-speed centre window; squeeze no longer
        # asks the provider to accelerate the entire source clip.
        fast = build_shot_prompt(_Shot(), squeeze_factor=2.5)
        slow = build_shot_prompt(_Shot(), squeeze_factor=1.0)
        self.assertTrue(fast.startswith("Slow"))
        self.assertEqual(fast, slow)

    def test_outdoor_and_approach_shots_are_given_something_that_moves(self):
        # Only 4% of reference shots hold still. A frozen world under a moving
        # camera reads as a slideshow.
        self.assertIn("breeze", build_shot_prompt(_Shot(room_type="backyard", act=3)))
        self.assertIn("breeze", build_shot_prompt(_Shot(room_type="front yard", act=1)))

    def test_a_close_up_names_its_subject(self):
        prompt = build_shot_prompt(
            _Shot(is_close_up=True, detail_subject="Fireplace"))
        self.assertIn("fireplace", prompt)

    def test_an_author_override_wins_outright(self):
        self.assertEqual(
            build_shot_prompt(_Shot(prompt_override="  my own words  ")),
            "my own words")


if __name__ == "__main__":
    unittest.main()
