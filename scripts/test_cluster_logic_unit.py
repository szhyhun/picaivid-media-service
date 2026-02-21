#!/usr/bin/env python3
"""Fast unit-style checks for story ordering and duplicate handling logic."""
from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.phase1_analyze.clustering import _order_clusters_for_story
from app.pipeline.phase1_analyze.learned_matching import deduplicate_and_split_cluster


@dataclass
class DummyPhoto:
    id: int
    room_label: str
    position: int
    room_override: str | None = None
    final_score: float | None = None
    manual_metadata: dict | None = None


def test_story_ordering() -> None:
    drone = [DummyPhoto(1, "drone shot", 1)]
    entrance = [DummyPhoto(2, "entrance", 2)]
    living = [DummyPhoto(3, "living room", 20)]
    kitchen_master = [
        DummyPhoto(4, "kitchen", 18, manual_metadata={"is_master": True, "master_priority": 3})
    ]
    patio = [DummyPhoto(5, "patio", 40)]
    duplicate = [DummyPhoto(6, "living room", 21)]

    ordered = _order_clusters_for_story(
        [patio, living, duplicate, kitchen_master, entrance, drone],
        duplicate_of_map={6: 3},
    )

    ordered_ids = [cluster[0].id for cluster in ordered]
    assert ordered_ids[0] == 1, f"Expected drone first, got {ordered_ids}"
    assert ordered_ids[1] == 2, f"Expected entrance second, got {ordered_ids}"
    assert ordered_ids.index(4) < ordered_ids.index(3), f"Master kitchen should precede living: {ordered_ids}"
    assert ordered_ids[-1] == 6, f"Duplicate cluster should be last: {ordered_ids}"



def test_opposite_side_preference() -> None:
    # Four same-room clusters at progressive capture positions.
    clusters = [
        [DummyPhoto(11, "living room", 10)],
        [DummyPhoto(12, "living room", 20)],
        [DummyPhoto(13, "living room", 30)],
        [DummyPhoto(14, "living room", 40)],
    ]

    ordered = _order_clusters_for_story(clusters)
    ordered_positions = [c[0].position for c in ordered]
    assert ordered_positions == [10, 40, 20, 30], (
        "Expected alternating sides [10, 40, 20, 30], "
        f"got {ordered_positions}"
    )



def test_dedup_singleton_cluster() -> None:
    photo_ids = [1, 2, 3]
    cluster_ids = [1, 2, 3]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    adjacency = np.zeros((3, 3), dtype=np.float32)

    clusters, duplicate_of = deduplicate_and_split_cluster(
        cluster_ids,
        photo_ids,
        embeddings,
        adjacency,
        max_size=3,
    )

    assert duplicate_of == {2: 1}, f"Expected duplicate map {{2: 1}}, got {duplicate_of}"
    assert [1, 3] in clusters, f"Expected canonical cluster [1, 3], got {clusters}"
    assert [2] in clusters, f"Expected duplicate singleton [2], got {clusters}"


if __name__ == "__main__":
    test_story_ordering()
    test_opposite_side_preference()
    test_dedup_singleton_cluster()
    print("All unit checks passed")
