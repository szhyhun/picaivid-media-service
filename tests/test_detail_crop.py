"""Tests for object-aware cinematic detail selection."""
from __future__ import annotations

import unittest

from app.pipeline.phase1_analyze.detail_crop import (
    DETAIL_SUBJECTS_BY_ROOM,
    DETAIL_ANALYSIS_VERSION,
    DETAIL_METRICS_KEY,
    OUTPUT_ASPECT,
    _Detection,
    _resolved_detection,
    _subject_for_label,
    crop_for_subject,
    detail_subjects,
    is_detail_eligible,
    resolve_persisted_detail_candidates,
)


class EditorialEligibilityTests(unittest.TestCase):
    def test_only_rooms_with_named_subjects_are_eligible(self):
        self.assertTrue(is_detail_eligible("kitchen"))
        self.assertTrue(is_detail_eligible("patio"))
        for room_type in ("bedroom", "bathroom", "hallway", "office",
                          "garage", "storage", "laundry room"):
            self.assertFalse(is_detail_eligible(room_type))

    def test_grill_is_an_allowed_patio_subject(self):
        subjects = detail_subjects("patio")
        self.assertIn("grill", {subject.key for subject in subjects})

    def test_room_type_scopes_subjects_but_does_not_create_a_detail(self):
        self.assertEqual(detail_subjects("unclassified space"), ())
        self.assertNotIn("bedroom", DETAIL_SUBJECTS_BY_ROOM)


class SubjectMappingTests(unittest.TestCase):
    def test_detector_label_maps_to_the_product_subject(self):
        kitchen = detail_subjects("kitchen")
        self.assertEqual(_subject_for_label("oven", kitchen).key, "cooking")
        self.assertEqual(_subject_for_label("cook range hood", kitchen).key, "range_hood")


class CropGeometryTests(unittest.TestCase):
    def test_crop_keeps_a_complete_subject_with_output_aspect(self):
        crop = crop_for_subject(
            (400, 350, 1000, 800), (2048, 1365), score=0.8,
        )
        self.assertIsNotNone(crop)
        self.assertAlmostEqual(crop.width / crop.height, OUTPUT_ASPECT, places=2)
        self.assertLessEqual(crop.x, 400)
        self.assertLessEqual(crop.y, 350)
        self.assertGreaterEqual(crop.x + crop.width, 1000)
        self.assertGreaterEqual(crop.y + crop.height, 800)

    def test_partial_subject_touching_an_image_edge_is_rejected(self):
        self.assertIsNone(crop_for_subject(
            (1, 400, 600, 1000), (2048, 1365), score=0.8,
        ))

    def test_tiny_detection_cannot_create_a_postage_stamp_detail(self):
        self.assertIsNone(crop_for_subject(
            (500, 500, 600, 560), (2048, 1365), score=0.8,
        ))

    def test_crop_is_clamped_inside_its_source_image(self):
        crop = crop_for_subject(
            (1500, 800, 1900, 1200), (2048, 1365), score=0.8,
        )
        self.assertIsNotNone(crop)
        self.assertGreaterEqual(crop.x, 0)
        self.assertGreaterEqual(crop.y, 0)
        self.assertLessEqual(crop.x + crop.width, 2048)
        self.assertLessEqual(crop.y + crop.height, 1365)

    def test_dedicated_portrait_detail_keeps_its_original_frame(self):
        subject = next(item for item in detail_subjects("entrance") if item.key == "entry_light")
        detail = _resolved_detection(
            42,
            (1000, 1500),
            _Detection(subject=subject, score=0.8, box=(350, 100, 650, 1300)),
        )
        self.assertIsNotNone(detail)
        self.assertEqual(detail.framing, "source_frame")
        self.assertIsNone(detail.crop)


class PersistedDetailTests(unittest.TestCase):
    def test_saved_detail_is_selected_without_live_model_inference(self):
        shot = type("Shot", (), {"room_id": "kitchen", "room_type": "kitchen", "is_close_up": False})()
        resolved = resolve_persisted_detail_candidates(
            [shot],
            {"kitchen": [7]},
            {7: {
                DETAIL_METRICS_KEY: {
                    "version": DETAIL_ANALYSIS_VERSION,
                    "status": "complete",
                    "objects": [{
                        "subject_key": "cooking",
                        "subject_label": "Stove or oven",
                        "subject_box": [400, 300, 1100, 900],
                        "detector_score": 0.9,
                        "crop_box": [300, 250, 1200, 756],
                        "selection_score": 0.9,
                        "source_size": [2048, 1365],
                        "framing": "object_crop",
                    }],
                },
            }},
        )
        self.assertEqual(len(resolved["kitchen"]), 1)
        self.assertEqual(resolved["kitchen"][0].subject_key, "cooking")


if __name__ == "__main__":
    unittest.main()
