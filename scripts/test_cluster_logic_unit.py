#!/usr/bin/env python3
"""Fast unit-style checks for story ordering and duplicate handling logic."""
from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.pipeline.phase1_analyze.learned_matching as lm
from app.pipeline.phase1_analyze.clustering import _order_clusters_for_story
from app.pipeline.phase1_analyze.learned_matching import (
    _annotate_pair_source_with_oracle,
    accepts_very_far_pair,
    build_component_edge_mask,
    cross_room_min_inliers_required,
    deduplicate_and_split_cluster,
    enforce_transition_quality,
)


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


def test_dedup_drop_duplicates() -> None:
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
        keep_duplicate_singletons=False,
    )

    assert duplicate_of == {2: 1}, f"Expected duplicate map {{2: 1}}, got {duplicate_of}"
    assert clusters == [[1, 3]], f"Expected dropped duplicate output [[1, 3]], got {clusters}"


def test_split_prefers_cutting_long_capture_gap() -> None:
    # Simulates a 4-photo oversized cluster where middle boundary has stronger
    # geometry but a much larger capture-order gap (different room instance).
    photo_ids = list(range(1, 11))
    cluster_ids = [2, 3, 9, 10]
    embeddings = np.eye(10, dtype=np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    adjacency = np.zeros((10, 10), dtype=np.float32)

    # Transition strengths in cluster order [2,3,9,10]:
    # 2-3: 0.49 (adjacent in capture order)
    # 3-9: 0.55 (strong but huge gap)
    # 9-10: 0.43 (adjacent in capture order)
    def link(a: int, b: int, score: float) -> None:
        ia = photo_ids.index(a)
        ib = photo_ids.index(b)
        adjacency[ia, ib] = score
        adjacency[ib, ia] = score

    link(2, 3, 0.49)
    link(3, 9, 0.55)
    link(9, 10, 0.43)

    clusters, duplicate_of = deduplicate_and_split_cluster(
        cluster_ids,
        photo_ids,
        embeddings,
        adjacency,
        max_size=3,
    )

    assert duplicate_of == {}, f"Expected no duplicates, got {duplicate_of}"
    assert [2, 3] in clusters and [9, 10] in clusters, (
        f"Expected long-gap split into [2,3] and [9,10], got {clusters}"
    )


def test_transition_quality_splits_weak_semantic_chain() -> None:
    photo_ids = [2094, 2095, 2096, 2097, 2098, 2099]
    ordered_clusters = [[2095, 2094, 2099]]

    adjacency = np.zeros((6, 6), dtype=np.float32)
    similarity = np.eye(6, dtype=np.float32)
    edge_has_geometry = np.zeros((6, 6), dtype=bool)

    # Semantic-only links (no geometry) must split under strict geometry mode.
    adjacency[0, 1] = adjacency[1, 0] = 0.819
    adjacency[0, 5] = adjacency[5, 0] = 0.792
    similarity[0, 1] = similarity[1, 0] = 0.819
    similarity[0, 5] = similarity[5, 0] = 0.792

    similarity_records = [
        {"photo_a_id": 2094, "photo_b_id": 2095, "geometric_inliers": 0, "pair_source": "both"},
        {"photo_a_id": 2094, "photo_b_id": 2099, "geometric_inliers": 0, "pair_source": "dinov2_topk_semantic_recovery"},
        {"photo_a_id": 2095, "photo_b_id": 2099, "geometric_inliers": 0, "pair_source": "dinov2_topk"},
    ]

    refined = enforce_transition_quality(
        ordered_clusters=ordered_clusters,
        photo_ids=photo_ids,
        adjacency=adjacency,
        similarity=similarity,
        edge_has_geometry=edge_has_geometry,
        similarity_records=similarity_records,
    )

    assert refined == [[2095], [2094], [2099]], f"Expected semantic-only edges to split, got {refined}"


def test_transition_quality_keeps_geometric_chain() -> None:
    photo_ids = [1, 2, 3]
    ordered_clusters = [[1, 2, 3]]

    adjacency = np.zeros((3, 3), dtype=np.float32)
    similarity = np.eye(3, dtype=np.float32)
    edge_has_geometry = np.zeros((3, 3), dtype=bool)

    adjacency[0, 1] = adjacency[1, 0] = 0.36
    adjacency[1, 2] = adjacency[2, 1] = 0.31
    similarity[0, 1] = similarity[1, 0] = 0.74
    similarity[1, 2] = similarity[2, 1] = 0.79
    edge_has_geometry[0, 1] = edge_has_geometry[1, 0] = True
    edge_has_geometry[1, 2] = edge_has_geometry[2, 1] = True

    similarity_records = [
        {
            "photo_a_id": 1,
            "photo_b_id": 2,
            "geometric_inliers": 17,
            "cross_left_to_right": 0.11,
            "cross_right_to_left": 0.02,
            "cross_center_to_center": 0.03,
            "kornia_overlap_ratio": 0.20,
        },
        {
            "photo_a_id": 2,
            "photo_b_id": 3,
            "geometric_inliers": 14,
            "cross_left_to_right": 0.09,
            "cross_right_to_left": 0.01,
            "cross_center_to_center": 0.04,
            "kornia_overlap_ratio": 0.18,
        },
    ]

    refined = enforce_transition_quality(
        ordered_clusters=ordered_clusters,
        photo_ids=photo_ids,
        adjacency=adjacency,
        similarity=similarity,
        edge_has_geometry=edge_has_geometry,
        similarity_records=similarity_records,
    )

    assert refined == ordered_clusters, f"Expected chain preserved, got {refined}"


def test_component_edge_mask_prunes_weak_dist2_semantic() -> None:
    photo_ids = [1, 2, 3]
    adjacency = np.zeros((3, 3), dtype=np.float32)
    edge_has_geometry = np.zeros((3, 3), dtype=bool)

    adjacency[0, 1] = adjacency[1, 0] = 0.81  # adjacent semantic - should keep
    adjacency[1, 2] = adjacency[2, 1] = 0.80  # adjacent semantic - should keep
    adjacency[0, 2] = adjacency[2, 0] = 0.79  # dist2 semantic - should prune

    similarity_records = [
        {"photo_a_id": 1, "photo_b_id": 2, "pair_source": "temporal_window"},
        {"photo_a_id": 2, "photo_b_id": 3, "pair_source": "temporal_window"},
        {"photo_a_id": 1, "photo_b_id": 3, "pair_source": "temporal_window"},
    ]

    original_mode = lm.GEOMETRY_ONLY_CLUSTER_MEMBERSHIP
    lm.GEOMETRY_ONLY_CLUSTER_MEMBERSHIP = False
    try:
        mask = build_component_edge_mask(
            adjacency=adjacency,
            edge_has_geometry=edge_has_geometry,
            similarity_records=similarity_records,
            photo_ids=photo_ids,
            room_labels=None,
        )
    finally:
        lm.GEOMETRY_ONLY_CLUSTER_MEMBERSHIP = original_mode

    assert bool(mask[0, 1]), "Expected adjacent semantic edge 1-2 to stay"
    assert bool(mask[1, 2]), "Expected adjacent semantic edge 2-3 to stay"
    assert not bool(mask[0, 2]), "Expected weak dist2 semantic edge 1-3 to be pruned"


def test_component_edge_mask_keeps_same_label_adjacent_recovery() -> None:
    photo_ids = [101, 102]
    adjacency = np.zeros((2, 2), dtype=np.float32)
    edge_has_geometry = np.zeros((2, 2), dtype=bool)
    adjacency[0, 1] = adjacency[1, 0] = 0.74  # below base 0.78, above same-label 0.70

    original_mode = lm.GEOMETRY_ONLY_CLUSTER_MEMBERSHIP
    lm.GEOMETRY_ONLY_CLUSTER_MEMBERSHIP = False
    try:
        mask = build_component_edge_mask(
            adjacency=adjacency,
            edge_has_geometry=edge_has_geometry,
            similarity_records=[{"photo_a_id": 101, "photo_b_id": 102, "pair_source": "both|koR"}],
            photo_ids=photo_ids,
            room_labels=["living room", "living room"],
        )
    finally:
        lm.GEOMETRY_ONLY_CLUSTER_MEMBERSHIP = original_mode
    assert bool(mask[0, 1]), "Expected same-label adjacent semantic edge to be recovered"


def test_component_edge_mask_prunes_cross_label_adjacent() -> None:
    photo_ids = [201, 202]
    adjacency = np.zeros((2, 2), dtype=np.float32)
    edge_has_geometry = np.zeros((2, 2), dtype=bool)
    adjacency[0, 1] = adjacency[1, 0] = 0.82  # below cross-label adj min 0.84

    original_mode = lm.GEOMETRY_ONLY_CLUSTER_MEMBERSHIP
    lm.GEOMETRY_ONLY_CLUSTER_MEMBERSHIP = False
    try:
        mask = build_component_edge_mask(
            adjacency=adjacency,
            edge_has_geometry=edge_has_geometry,
            similarity_records=[{"photo_a_id": 201, "photo_b_id": 202, "pair_source": "temporal_window"}],
            photo_ids=photo_ids,
            room_labels=["living room", "kitchen"],
        )
    finally:
        lm.GEOMETRY_ONLY_CLUSTER_MEMBERSHIP = original_mode
    assert not bool(mask[0, 1]), "Expected cross-label adjacent semantic edge to be pruned"


def test_component_edge_mask_keeps_geometric_edge() -> None:
    photo_ids = [10, 11]
    adjacency = np.zeros((2, 2), dtype=np.float32)
    edge_has_geometry = np.zeros((2, 2), dtype=bool)

    adjacency[0, 1] = adjacency[1, 0] = 0.17
    edge_has_geometry[0, 1] = edge_has_geometry[1, 0] = True

    mask = build_component_edge_mask(
        adjacency=adjacency,
        edge_has_geometry=edge_has_geometry,
        similarity_records=[{"photo_a_id": 10, "photo_b_id": 11, "pair_source": "dinov2_topk"}],
        photo_ids=photo_ids,
    )
    assert bool(mask[0, 1]), "Expected geometric edge to remain connected"


def test_geometry_only_membership_prunes_semantic_edges() -> None:
    photo_ids = [301, 302]
    adjacency = np.zeros((2, 2), dtype=np.float32)
    edge_has_geometry = np.zeros((2, 2), dtype=bool)
    adjacency[0, 1] = adjacency[1, 0] = 0.95

    original_mode = lm.GEOMETRY_ONLY_CLUSTER_MEMBERSHIP
    lm.GEOMETRY_ONLY_CLUSTER_MEMBERSHIP = True
    try:
        mask = build_component_edge_mask(
            adjacency=adjacency,
            edge_has_geometry=edge_has_geometry,
            similarity_records=[{"photo_a_id": 301, "photo_b_id": 302, "pair_source": "temporal_window"}],
            photo_ids=photo_ids,
            room_labels=["bedroom", "bedroom"],
        )
    finally:
        lm.GEOMETRY_ONLY_CLUSTER_MEMBERSHIP = original_mode

    assert not bool(mask[0, 1]), "Geometry-only mode must not keep semantic-only edges"


def test_oracle_pair_source_annotation() -> None:
    assert _annotate_pair_source_with_oracle("dinov2_topk", None) == "dinov2_topk"
    assert _annotate_pair_source_with_oracle(
        "dinov2_topk",
        {"oracle": {"mode": "shadow", "decision": "pass"}},
    ).endswith("|koS")
    assert _annotate_pair_source_with_oracle(
        "dinov2_topk",
        {"oracle": {"mode": "gate", "decision": "pass"}},
    ).endswith("|koG")
    assert _annotate_pair_source_with_oracle(
        "dinov2_topk",
        {"oracle": {"mode": "gate", "decision": "reject"}},
    ).endswith("|koR")


def test_cross_room_adjacent_thresholds() -> None:
    # Extremely low semantic adjacent cross-room must demand strict evidence.
    assert cross_room_min_inliers_required(1, 0.01) == 30
    # Moderate semantic adjacent cross-room still stricter than default adjacent.
    assert cross_room_min_inliers_required(1, 0.30) == 22
    # High semantic adjacent cross-room uses adjacent threshold.
    assert cross_room_min_inliers_required(1, 0.60) == 15
    # Non-adjacent cross-room always strict.
    assert cross_room_min_inliers_required(2, 0.90) == 30


def test_very_far_pair_guard() -> None:
    # Similar to false positive pattern: sem too low for very-far pair.
    assert not accepts_very_far_pair(position_gap=56, sem_sim=0.40, num_inliers=50, score=0.70)
    # Inliers too low for very-far pair.
    assert not accepts_very_far_pair(position_gap=56, sem_sim=0.70, num_inliers=30, score=0.70)
    # Score too low for very-far pair.
    assert not accepts_very_far_pair(position_gap=56, sem_sim=0.70, num_inliers=40, score=0.30)
    # Valid very-far pair can still pass.
    assert accepts_very_far_pair(position_gap=56, sem_sim=0.70, num_inliers=40, score=0.60)


def test_blend_geometric_semantic_score_geometry_dominant() -> None:
    original_geo = lm.GEOMETRIC_SCORE_WEIGHT
    original_sem = lm.SEMANTIC_SCORE_WEIGHT
    lm.GEOMETRIC_SCORE_WEIGHT = 0.9
    lm.SEMANTIC_SCORE_WEIGHT = 0.1
    try:
        blended = lm.blend_geometric_semantic_score(geometric_score=0.50, semantic_score=0.90)
    finally:
        lm.GEOMETRIC_SCORE_WEIGHT = original_geo
        lm.SEMANTIC_SCORE_WEIGHT = original_sem

    assert abs(blended - 0.54) < 1e-6, f"Expected 0.54 blended score, got {blended}"


def test_transition_geometry_score_penalizes_incoherent_motion() -> None:
    width, height = 640, 480
    num = 40
    xs = np.linspace(120, 520, num, dtype=np.float32)
    ys = np.linspace(120, 360, num, dtype=np.float32)
    pts0 = np.stack([xs, ys], axis=1)

    # Coherent camera motion: near-uniform horizontal shift.
    pts1_coherent = pts0 + np.array([40.0, 2.0], dtype=np.float32)
    seg = lm._compute_segment_scores(
        pts0,
        pts1_coherent,
        width0=width,
        height0=height,
        width1=width,
        height1=height,
    )
    score_coherent = lm._compute_transition_geometry_score(
        num_matches=num,
        num_inliers=num,
        inlier_points0=pts0,
        inlier_points1=pts1_coherent,
        width0=width,
        height0=height,
        width1=width,
        height1=height,
        segment_scores=seg,
    )

    # Incoherent motion: shuffled/randomized offsets across points.
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 35.0, size=(num, 2)).astype(np.float32)
    pts1_incoherent = pts0 + noise
    seg_bad = lm._compute_segment_scores(
        pts0,
        pts1_incoherent,
        width0=width,
        height0=height,
        width1=width,
        height1=height,
    )
    score_incoherent = lm._compute_transition_geometry_score(
        num_matches=num,
        num_inliers=num,
        inlier_points0=pts0,
        inlier_points1=pts1_incoherent,
        width0=width,
        height0=height,
        width1=width,
        height1=height,
        segment_scores=seg_bad,
    )

    assert score_coherent > score_incoherent, (
        f"Expected coherent score > incoherent score, got {score_coherent:.3f} <= {score_incoherent:.3f}"
    )


if __name__ == "__main__":
    test_story_ordering()
    test_opposite_side_preference()
    test_dedup_singleton_cluster()
    test_dedup_drop_duplicates()
    test_split_prefers_cutting_long_capture_gap()
    test_transition_quality_splits_weak_semantic_chain()
    test_transition_quality_keeps_geometric_chain()
    test_component_edge_mask_prunes_weak_dist2_semantic()
    test_component_edge_mask_keeps_same_label_adjacent_recovery()
    test_component_edge_mask_prunes_cross_label_adjacent()
    test_component_edge_mask_keeps_geometric_edge()
    test_geometry_only_membership_prunes_semantic_edges()
    test_oracle_pair_source_annotation()
    test_cross_room_adjacent_thresholds()
    test_very_far_pair_guard()
    test_blend_geometric_semantic_score_geometry_dominant()
    test_transition_geometry_score_penalizes_incoherent_motion()
    print("All unit checks passed")
