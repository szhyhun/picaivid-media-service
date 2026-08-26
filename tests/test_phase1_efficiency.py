"""Regression tests for Phase 1's one-pass model-work contract."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.pipeline.phase1_analyze.analyzer import (
    Phase1Analyzer,
    _score_service_room_labels,
)


def _photo(photo_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=photo_id,
        embedding=None,
        room_override=None,
        room_label=None,
        filename=f"photo-{photo_id}.jpg",
        manual_metadata={},
    )


class OnePassOpenCLIPTests(unittest.TestCase):
    def test_embeddings_use_one_batch_model_call(self) -> None:
        photos = [_photo(index) for index in range(1, 5)]
        images = {
            photo.id: Image.new("RGB", (16, 16), color=(photo.id, 0, 0))
            for photo in photos
        }
        embeddings = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
        analyzer = Phase1Analyzer(db=None)
        with (
            patch("app.pipeline.phase1_analyze.analyzer.settings.OPENCLIP_BATCH_SIZE", 16),
            patch(
                "app.pipeline.phase1_analyze.analyzer.openclip_model.get_batch_embeddings",
                return_value=embeddings,
            ) as encode_batch,
            patch(
                "app.pipeline.phase1_analyze.analyzer.openclip_model.get_embedding",
                side_effect=AssertionError("single-photo fallback should not run"),
            ),
        ):
            analyzer._compute_embeddings(photos, images)

        encode_batch.assert_called_once()
        self.assertEqual([photo.embedding for photo in photos], embeddings.tolist())
        for image in images.values():
            image.close()

    def test_room_classification_reuses_stored_embeddings(self) -> None:
        photos = [_photo(index) for index in range(1, 4)]
        for index, photo in enumerate(photos):
            photo.embedding = np.eye(3, dtype=np.float32)[index].tolist()
        analyzer = Phase1Analyzer(db=None)
        with (
            patch(
                "app.pipeline.phase1_analyze.analyzer.openclip_model.classify_embeddings",
                return_value=[("kitchen", 0.8), ("bedroom", 0.7), ("bathroom", 0.6)],
            ) as classify_embeddings,
            patch(
                "app.pipeline.phase1_analyze.analyzer.openclip_model.classify_room",
                side_effect=AssertionError("images must not be re-encoded"),
            ),
        ):
            analyzer._classify_rooms(photos, {})

        classify_embeddings.assert_called_once()
        self.assertEqual([photo.room_label for photo in photos], ["kitchen", "bedroom", "bathroom"])

    def test_service_scoring_reuses_stored_embedding(self) -> None:
        with (
            patch(
                "app.pipeline.phase1_analyze.analyzer.openclip_model.score_embedding",
                return_value={},
            ) as score_embedding,
            patch(
                "app.pipeline.phase1_analyze.analyzer.openclip_model.score_labels",
                side_effect=AssertionError("image scoring must not run"),
            ),
        ):
            _score_service_room_labels(embedding=[0.0, 1.0, 0.0])

        score_embedding.assert_called_once()


if __name__ == "__main__":
    unittest.main()
