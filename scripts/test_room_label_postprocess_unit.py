#!/usr/bin/env python3
"""Unit checks for exterior room-label postprocessing."""
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.phase1_analyze.analyzer import Phase1Analyzer


@dataclass
class DummyPhoto:
    id: int
    position: int
    room_label: str
    room_override: str | None = None


@dataclass
class DummySimilarity:
    photo_a_id: int
    photo_b_id: int
    relation_confidence: float
    is_connected: int = 1


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *_args, **_kwargs):
        return self

    def all(self):
        return list(self._rows)


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *_args, **_kwargs):
        return _FakeQuery(self._rows)


def test_relabel_front_to_aerial_when_bracketed() -> None:
    photos = [DummyPhoto(1, 0, "front yard")]
    photos.append(DummyPhoto(2, 1, "aerial view"))
    photos.append(DummyPhoto(3, 2, "front yard"))
    photos.append(DummyPhoto(4, 3, "aerial view"))
    for i in range(5, 13):
        photos.append(DummyPhoto(i, i - 1, "living room"))
    analyzer = Phase1Analyzer(db=None)  # db not used by this method
    analyzer._postprocess_exterior_room_labels(photos)  # noqa: SLF001

    assert photos[2].room_label == "aerial view", "Expected early front-yard frame to relabel to aerial view"


def test_relabel_late_front_to_backyard_with_context() -> None:
    photos = [
        DummyPhoto(10, 0, "front yard"),
        DummyPhoto(11, 1, "entrance"),
        DummyPhoto(12, 2, "living room"),
        DummyPhoto(13, 3, "patio"),
        DummyPhoto(14, 4, "backyard"),
        DummyPhoto(15, 5, "front yard"),
    ]
    analyzer = Phase1Analyzer(db=None)
    analyzer._postprocess_exterior_room_labels(photos)  # noqa: SLF001

    assert photos[-1].room_label == "backyard", "Expected late front-yard frame to relabel to backyard"


def test_respect_manual_override() -> None:
    photos = [
        DummyPhoto(20, 0, "front yard", room_override="front yard"),
        DummyPhoto(21, 1, "aerial view"),
        DummyPhoto(22, 2, "front yard"),
        DummyPhoto(23, 3, "aerial view"),
        DummyPhoto(24, 4, "front yard"),
    ]
    analyzer = Phase1Analyzer(db=None)
    analyzer._postprocess_exterior_room_labels(photos)  # noqa: SLF001

    assert photos[0].room_label == "front yard", "Manual override photo should never be relabeled"


def test_relabel_interior_patio_to_kitchen_by_sequence_context() -> None:
    photos = [
        DummyPhoto(100, 10, "living room"),
        DummyPhoto(101, 11, "patio"),
        DummyPhoto(102, 12, "kitchen"),
        DummyPhoto(103, 13, "kitchen"),
        DummyPhoto(104, 14, "dining room"),
    ]
    analyzer = Phase1Analyzer(db=None)
    analyzer._postprocess_interior_room_labels(photos)  # noqa: SLF001

    assert photos[1].room_label == "kitchen", "Expected interior patio false-positive to relabel to kitchen"


def test_relabel_living_to_dining_by_strong_adjacent_geometry() -> None:
    photos = [
        DummyPhoto(200, 20, "living room"),
        DummyPhoto(201, 21, "dining room"),
        DummyPhoto(202, 22, "bedroom"),
    ]
    similarities = [
        DummySimilarity(photo_a_id=200, photo_b_id=201, relation_confidence=0.72)
    ]
    analyzer = Phase1Analyzer(db=_FakeDB(similarities))
    analyzer._postprocess_interior_room_labels(photos, job_id=1)  # noqa: SLF001

    assert photos[0].room_label == "dining room", "Expected living room shot to relabel to dining room from strong adjacent dining geometry"


if __name__ == "__main__":
    test_relabel_front_to_aerial_when_bracketed()
    test_relabel_late_front_to_backyard_with_context()
    test_respect_manual_override()
    test_relabel_interior_patio_to_kitchen_by_sequence_context()
    test_relabel_living_to_dining_by_strong_adjacent_geometry()
    print("All room-label postprocess checks passed")
