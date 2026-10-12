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
            "detail_subject": None, "prompt_override": None, "prompt_note": None, **kw})


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
        for banned in ("tilt the camera", "roll the camera", "orbit the camera",
                      "parallax move"):
            self.assertNotIn(banned, prompt.lower())

    def test_opening_and_closing_single_shots_have_bookend_moves(self):
        # Position owns the zoom axis only. The measured pan survives it, so an
        # opening shot pushes in *and* keeps drifting the way it was measured.
        opening = build_shot_prompt(_Shot(move="pan left"), is_first=True)
        closing = build_shot_prompt(_Shot(move="push in"), is_last=True)
        self.assertIn("Slow push in", opening)
        self.assertIn("drifting gently to the left", opening)
        self.assertIn("straight pull back", closing)

    def test_it_never_describes_the_room(self):
        # The model already has the pixels; describing the building invites it
        # to invent a different one.
        prompt = build_shot_prompt(_Shot(room_type="kitchen")).lower()
        self.assertNotIn("kitchen", prompt)

    def test_every_prompt_uses_positive_stability_language(self):
        prompt = build_shot_prompt(_Shot())
        for rule in (
            "Use only visual details visible in the supplied reference frame",
            "complete visual source of truth",
            "same scene identity and exact architecture",
            "every visible furnishing, fixture, surface, opening",
            "empty area",
            "Perfectly stabilized motorized camera rig",
            "level frame",
            "straight verticals",
            "object placement",
            "controlled",
            "constant speed",
        ):
            self.assertIn(rule, prompt)
        self.assertEqual(prompt.count("architecture"), 1)
        self.assertEqual(prompt.count("object placement"), 1)
        self.assertNotIn("No ", prompt)
        self.assertNotIn("Do not ", prompt)
        self.assertNotIn("do not ", prompt)
        self.assertNotIn("Photorealistic.", prompt)

    def test_pair_prompt_requests_stabilized_motion_without_changing_the_path(self):
        prompt = build_shot_prompt(_Shot(kind="pair_keyframes"))
        self.assertIn("from the first reference frame to the second reference frame", prompt)
        self.assertIn("Preserve both supplied endpoint compositions", prompt)
        self.assertIn("constant speed", prompt)
        self.assertIn("Perfectly stabilized motorized camera rig", prompt)
        self.assertIn("object placement", prompt)

    def test_a_compound_move_names_every_axis_it_was_measured_with(self):
        # 87% of the singles in a real listing move on two or three axes, and
        # naming only the dominant one threw the rest away.
        prompt = build_shot_prompt(
            _Shot(room_type="living room", move="push in + pan right + tilt up"))
        self.assertIn("Slow push in", prompt)
        self.assertIn("drifting gently to the right", prompt)
        self.assertIn("rising gently", prompt)

    def test_a_single_axis_move_is_still_asked_for_straight(self):
        # With nothing else on the move, "straight" suppresses drift the model
        # would otherwise invent.
        prompt = build_shot_prompt(_Shot(room_type="living room", move="push in"))
        self.assertIn("straight push in", prompt)

    def test_entrance_strips_an_old_persisted_drift(self):
        prompt = build_shot_prompt(
            _Shot(room_type="entrance", move="push in + pan right + tilt up"))
        self.assertIn("Slow, straight push in", prompt)
        self.assertNotIn("drifting", prompt)
        self.assertNotIn("rising", prompt)

    def test_paired_room_single_keeps_pan_but_strips_tilt(self):
        prompt = build_shot_prompt(
            _Shot(room_type="kitchen", move="push in + pan right + tilt up"),
            room_has_pair=True,
        )
        self.assertIn("drifting gently to the right", prompt)
        self.assertNotIn("rising", prompt)

    def test_no_prompt_ever_asks_the_camera_to_tilt_or_roll(self):
        # A pitch and a slanted horizon share the word "tilt", and a slanted
        # horizon is exactly what the stability invariants exist to prevent.
        for move in ("push in + pan right + tilt up", "pull out + tilt down",
                     "tilt up", "push in"):
            prompt = build_shot_prompt(
                _Shot(room_type="living room", move=move)).lower()
            self.assertNotIn("tilt", prompt)
            # "controlled" contains the substring, so match the word itself.
            self.assertNotIn("roll ", prompt)
            self.assertNotIn("rolling", prompt)

    def test_source_window_keeps_the_same_slow_prompt_at_native_speed(self):
        # The editor now trims a native-speed centre window; squeeze no longer
        # asks the provider to accelerate the entire source clip.
        fast = build_shot_prompt(_Shot(), squeeze_factor=2.5)
        slow = build_shot_prompt(_Shot(), squeeze_factor=1.0)
        self.assertTrue(fast.startswith("Slow"))
        self.assertEqual(fast, slow)

    def test_scene_specific_motion_is_not_invented_from_room_labels(self):
        # A room label does not prove that curtains, foliage, or any other
        # moving subject exists in the source photo.
        for room_type in ("living room", "backyard", "front yard", "aerial view"):
            prompt = build_shot_prompt(_Shot(room_type=room_type, act=1))
            self.assertNotIn("curtain", prompt.lower())
            self.assertNotIn("breeze", prompt.lower())

    def test_a_close_up_names_its_subject(self):
        prompt = build_shot_prompt(
            _Shot(is_close_up=True, detail_subject="Fireplace"))
        self.assertIn("fireplace", prompt)

    def test_an_author_override_keeps_custom_text_and_safety_contract(self):
        prompt = build_shot_prompt(_Shot(prompt_override="  my own words  "))
        self.assertTrue(prompt.startswith("my own words "))
        self.assertIn("Perfectly stabilized motorized camera rig", prompt)


if __name__ == "__main__":
    unittest.main()


class PromptNoteTests(unittest.TestCase):
    """A customer's note is added to our direction, never a replacement for it."""

    def test_the_note_is_appended_after_our_direction(self):
        prompt = build_shot_prompt(
            _Shot(prompt_note='Keep the pool in frame.'))
        self.assertTrue(prompt.endswith('Keep the pool in frame.'))
        # Everything we decided is still in there.
        self.assertIn('exact architecture', prompt)
        self.assertIn('push in', prompt.lower())

    def test_an_empty_note_adds_nothing(self):
        self.assertEqual(
            build_shot_prompt(_Shot(prompt_note='   ')),
            build_shot_prompt(_Shot()),
        )

    def test_an_override_still_wins_outright(self):
        # The override is the development tool; the note is the customer's.
        prompt = build_shot_prompt(_Shot(prompt_override='mine', prompt_note='theirs'))
        self.assertTrue(prompt.startswith('mine'))
        # The safety contract survives an override, and the note still follows.
        self.assertIn('exact architecture', prompt)
        self.assertTrue(prompt.endswith('theirs'))


class PromptEconomyTests(unittest.TestCase):
    """Luma documents a 2-4 sentence target and penalises vague wording, so the
    contract is said once and in the frame count the shot actually has."""

    def test_the_source_of_truth_is_stated_once(self):
        # "Use only what is visible" and "treat it as the source of truth" were
        # the same instruction twice; repetition spends the sentence budget
        # without adding a constraint.
        prompt = build_shot_prompt(_Shot())
        self.assertEqual(prompt.count("source of truth"), 1)
        self.assertEqual(prompt.count("Use only visual details"), 1)

    def test_a_single_shot_is_told_it_has_one_reference_frame(self):
        prompt = build_shot_prompt(_Shot(kind="single_image"))
        self.assertIn("the supplied reference frame, which is the complete", prompt)
        self.assertNotIn("reference frame or frames", prompt)

    def test_a_pair_is_told_it_has_two(self):
        prompt = build_shot_prompt(_Shot(kind="pair_keyframes"))
        self.assertIn("the supplied reference frames, which are the complete", prompt)
        self.assertNotIn("reference frame or frames", prompt)

    def test_an_overridden_prompt_still_carries_the_invariants(self):
        # The override replaces the direction, never the safety contract -- and
        # it needs the shot's own frame count like any other prompt.
        prompt = build_shot_prompt(
            _Shot(kind="pair_keyframes", prompt_override="Drift left past the window."))
        self.assertTrue(prompt.startswith("Drift left past the window."))
        self.assertIn("the supplied reference frames, which are", prompt)
