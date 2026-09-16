"""The prompt shown in the editable timeline is the prompt we will send."""
import unittest
from types import SimpleNamespace

from app.main import _timeline_generation_prompt


class TimelinePromptTests(unittest.TestCase):
    def _clip(self, **overrides):
        values = {
            "clip": None,
            "kind": "single_image",
            "room_type": "living room",
            "is_close_up": False,
            "act": 2,
            "move_label": "pan right",
            "prompt_override": None,
            "detail_subject": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_planned_shot_exposes_current_resolved_prompt(self):
        prompt = _timeline_generation_prompt(self._clip(), 0, 1)
        self.assertIn("Slow pull back", prompt)
        self.assertIn("Perfectly stabilized motorized camera rig", prompt)
        self.assertNotIn("No ", prompt)

    def test_ready_shot_exposes_immutable_prompt_that_was_sent(self):
        actual = "provider prompt from the completed request"
        prompt = _timeline_generation_prompt(
            self._clip(clip=SimpleNamespace(prompt_used=actual)), 0, 1)
        self.assertEqual(prompt, actual)


if __name__ == "__main__":
    unittest.main()
