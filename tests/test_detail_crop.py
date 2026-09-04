"""Manufacturing detail shots by cropping.

Real-estate shoots do not contain close-ups: 2 of 295 photos across the six
labelled listings read as a detail rather than a room. Details are made by
cropping into room photographs, so the crop decision has to be sound.
"""
from __future__ import annotations

import unittest

import numpy as np

from app.pipeline.phase1_analyze.detail_crop import (
    CROP_WIDTH_FRACTION,
    MIN_DETAIL_SCORE,
    OUTPUT_ASPECT,
    best_crop,
    candidate_boxes,
    score_crops,
)


class CandidateBoxTests(unittest.TestCase):
    def test_crops_are_a_real_close_up_not_the_whole_frame(self):
        boxes = candidate_boxes(2048, 1365)
        widths = {w for _, _, w, _ in boxes}
        self.assertEqual(len(widths), 1)
        self.assertAlmostEqual(widths.pop() / 2048, CROP_WIDTH_FRACTION, places=2)

    def test_crops_match_the_output_aspect(self):
        for x, y, w, h in candidate_boxes(2048, 1365):
            self.assertAlmostEqual(w / h, OUTPUT_ASPECT, places=1)

    def test_every_box_lies_inside_the_frame(self):
        W, H = 2048, 1365
        for x, y, w, h in candidate_boxes(W, H):
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + w, W)
            self.assertLessEqual(y + h, H)

    def test_boxes_cover_the_frame_not_just_the_centre(self):
        """A detail is as often at the edge of a frame as in the middle."""
        boxes = candidate_boxes(2048, 1365)
        xs = {x for x, _, _, _ in boxes}
        ys = {y for _, y, _, _ in boxes}
        self.assertGreater(len(xs), 3)
        self.assertGreater(len(ys), 2)

    def test_a_very_wide_frame_does_not_produce_a_crop_taller_than_it_is(self):
        boxes = candidate_boxes(2000, 200)
        for _, y, _, h in boxes:
            self.assertLessEqual(y + h, 200)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.detail = np.array([1.0, 0.0, 0.0])
        self.blank = np.array([0.0, 1.0, 0.0])

    def test_a_detail_like_crop_scores_above_a_blank_one(self):
        crops = np.stack([self.detail, self.blank])
        scores = score_crops(crops, self.detail, self.blank)
        self.assertGreater(scores[0], scores[1])

    def test_score_is_a_difference_not_a_similarity(self):
        """A crop must be positively appealing, not merely not-blank."""
        neutral = np.array([0.0, 0.0, 1.0])
        scores = score_crops(neutral[None, :], self.detail, self.blank)
        self.assertAlmostEqual(float(scores[0]), 0.0, places=6)


class BestCropTests(unittest.TestCase):
    def setUp(self):
        self.boxes = [(0, 0, 100, 56), (100, 0, 100, 56), (200, 0, 100, 56)]

    def test_picks_the_highest_scoring_window(self):
        crop = best_crop(self.boxes, np.array([0.01, 0.05, 0.02]))
        self.assertIsNotNone(crop)
        self.assertEqual(crop.x, 100)
        self.assertAlmostEqual(crop.score, 0.05)

    def test_a_frame_with_no_detail_yields_nothing(self):
        """Better to skip the beat than cut to a close-up of a blank wall."""
        self.assertIsNone(best_crop(self.boxes, np.array([-0.05, -0.02, -0.09])))

    def test_the_threshold_is_what_rejects_a_weak_frame(self):
        weak = np.array([MIN_DETAIL_SCORE - 0.001] * 3)
        strong = np.array([MIN_DETAIL_SCORE + 0.001] * 3)
        self.assertIsNone(best_crop(self.boxes, weak))
        self.assertIsNotNone(best_crop(self.boxes, strong))

    def test_empty_input_is_handled(self):
        self.assertIsNone(best_crop([], np.array([])))

    def test_box_is_pil_ready(self):
        crop = best_crop(self.boxes, np.array([0.9, 0.0, 0.0]))
        self.assertEqual(crop.box, (0, 0, 100, 56))


if __name__ == "__main__":
    unittest.main()


class _FakeImage:
    """Minimal stand-in: crops record what was asked for, size is fixed."""

    def __init__(self, tag): self.tag = tag; self.size = (2048, 1365)
    def crop(self, box): return _FakeImage(self.tag)


class _FakeModel:
    """Scores every crop by the tag of the photo it came from, so a test can say
    which photo should win without needing CLIP."""

    def __init__(self, by_tag): self.by_tag = by_tag
    def encode_image(self, batch):
        import torch
        return torch.tensor(batch)


class RoomLinkedDetailTests(unittest.TestCase):
    """A detail must be of the room it follows: the reference films cut from a
    wide living room to that same living room's fireplace."""

    def setUp(self):
        import torch
        self.detail = np.array([1.0, 0.0])
        self.blank = np.array([0.0, 1.0])
        # preprocess turns a fake image into a vector whose score we control
        self.scores = {}

        def preprocess(img):
            import torch
            return torch.tensor(self.scores.get(img.tag, [0.0, 1.0]))
        self.preprocess = preprocess
        self.model = _FakeModel({})
        self.loader = lambda pid: _FakeImage(pid)

    def _run(self, room_photos, prefer=None, min_score=0.0):
        from app.pipeline.phase1_analyze.detail_crop import details_for_rooms
        return details_for_rooms(room_photos, self.loader, self.model,
                                 self.preprocess, self.detail, self.blank,
                                 prefer=prefer, min_score=min_score)

    def test_detail_comes_from_the_rooms_own_photos(self):
        """A living-room detail must never be sourced from the kitchen."""
        self.scores = {1: [1.0, 0.0], 2: [1.0, 0.0], 9: [1.0, 0.0]}
        found = self._run({"living": [1, 2], "kitchen": [9]})
        self.assertIn(1, [found["living"][0]])
        self.assertEqual(found["kitchen"][0], 9)

    def test_a_preferred_frame_wins_for_continuity(self):
        """Cutting into the frame just shown reads as the same look continuing,
        so it beats a marginally prettier crop elsewhere in the room."""
        self.scores = {1: [0.6, 0.0], 2: [0.9, 0.0]}
        found = self._run({"living": [1, 2]}, prefer={"living": [1]})
        self.assertEqual(found["living"][0], 1)

    def test_rooms_with_nothing_worth_cropping_are_absent(self):
        self.scores = {1: [0.0, 1.0]}
        found = self._run({"living": [1]}, min_score=0.5)
        self.assertNotIn("living", found)

    def test_only_the_given_rooms_are_searched(self):
        """Work is bounded to shots that reach the film: scoring every photo in a
        listing is ~5,900 CLIP encodes, nearly all of them wasted."""
        seen = []
        self.loader = lambda pid: (seen.append(pid), _FakeImage(pid))[1]
        self.scores = {1: [1.0, 0.0]}
        self._run({"living": [1]})
        self.assertEqual(seen, [1])
