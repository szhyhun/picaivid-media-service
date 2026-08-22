"""Unit coverage for the deterministic VGGT Phase 1 planning primitives."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from app.models import vggt as vggt_runtime
from app.pipeline.phase1_analyze import shot_planner, vggt_pipeline
from app.pipeline.phase1_analyze.clustering import _derive_render_groups
from app.pipeline.phase1_analyze.motion_planner import _requested_motion
from app.pipeline.phase1_analyze.vggt_pipeline import (
    PhotoRelationResult,
    _PhotoArrays,
    _estimate_similarity,
    _compute_pair_relations,
    _depth_metrics,
    _motion_affordance,
    _order_component,
    _project_metrics,
)


def _relation(left: int, right: int, confidence: float = 0.8) -> PhotoRelationResult:
    return PhotoRelationResult(
        photo_a_id=left,
        photo_b_id=right,
        overlap_score=0.7,
        track_support=0.8,
        reprojection_score=0.95,
        relation_confidence=confidence,
        baseline_distance=0.2,
        relative_transform={},
        direction_dx=12.0,
        direction_dy=0.0,
        continuity_type="interpolation_safe",
        is_bridge_edge=False,
        is_connected=True,
        debug_metrics={},
    )


class VGGTPhase1Tests(unittest.TestCase):
    def test_auto_device_prefers_cuda_then_mps_then_cpu(self) -> None:
        with patch.object(vggt_runtime.settings, "VGGT_DEVICE", "auto"), patch.object(vggt_runtime.torch.cuda, "is_available", return_value=True), patch.object(vggt_runtime.torch.backends.mps, "is_available", return_value=False):
            self.assertEqual(vggt_runtime._device(), "cuda")
        with patch.object(vggt_runtime.settings, "VGGT_DEVICE", "auto"), patch.object(vggt_runtime.torch.cuda, "is_available", return_value=False), patch.object(vggt_runtime.torch.backends.mps, "is_available", return_value=True), patch.object(vggt_runtime, "_mps_tensor_smoke_test", return_value=True):
            self.assertEqual(vggt_runtime._device(), "mps")
        with patch.object(vggt_runtime.settings, "VGGT_DEVICE", "auto"), patch.object(vggt_runtime.torch.cuda, "is_available", return_value=False), patch.object(vggt_runtime.torch.backends.mps, "is_available", return_value=False):
            self.assertEqual(vggt_runtime._device(), "cpu")

    def test_similarity_stitching_recovers_scale_rotation_and_translation(self) -> None:
        local = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        expected = 2.0 * (local @ rotation.T) + np.array([3.0, -2.0, 1.0])
        scale, fitted_rotation, translation = _estimate_similarity(local, expected)
        np.testing.assert_allclose(scale * (local @ fitted_rotation.T) + translation, expected, atol=1e-6)

    def test_projection_metrics_are_measured_and_finite(self) -> None:
        height = width = 4
        y, x = np.mgrid[:height, :width]
        points = np.stack((x, y, np.ones_like(x)), axis=-1).astype(np.float64)
        source = _PhotoArrays(
            extrinsic=np.hstack((np.eye(3), np.zeros((3, 1)))), intrinsic=np.eye(3),
            depth=np.ones((height, width)), depth_conf=np.ones((height, width)),
            point_map=points, point_conf=np.ones((height, width)), world_points=points,
        )
        metrics = _project_metrics(source, source)
        self.assertGreater(metrics["visible_fraction"], 0.9)
        self.assertGreater(metrics["depth_consistency"], 0.9)
        self.assertLess(metrics["reprojection_error"], 1e-6)

    def test_depth_metrics_are_scale_invariant(self) -> None:
        depth = np.linspace(1.0, 8.0, 64, dtype=np.float64).reshape(8, 8)
        first = _depth_metrics(depth)
        second = _depth_metrics(depth * 25.0)
        self.assertAlmostEqual(first["depth_variance"], second["depth_variance"])
        self.assertEqual(first["depth_layers"], second["depth_layers"])

    def test_omega_depth_unprojection_uses_world_to_camera_pose(self) -> None:
        depth = torch.ones((1, 1, 1, 1), dtype=torch.float32)
        extrinsic = torch.tensor([[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0]]])
        intrinsic = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        points = vggt_runtime._unproject_depth(depth, extrinsic, intrinsic)
        np.testing.assert_allclose(points.numpy(), np.array([[[[-1.0, -2.0, -2.0]]]]))

    def test_beam_search_is_deterministic_and_uses_verified_edges(self) -> None:
        relations = [_relation(1, 2), _relation(2, 3), _relation(3, 4)]
        positions = {1: 0, 2: 1, 3: 2, 4: 3}
        self.assertEqual(_order_component([1, 2, 3, 4], relations, positions), [1, 2, 3, 4])
        self.assertEqual(_order_component([1, 2, 3, 4], relations, positions), [1, 2, 3, 4])

    def test_motion_affordance_ignores_unconnected_internal_pairs(self) -> None:
        connected = [_relation(1, 2, 0.60), _relation(2, 3, 0.60)]
        unrelated = _relation(1, 3, 0.0)
        unrelated.is_connected = False
        unrelated.continuity_type = "unrelated"
        self.assertEqual(_motion_affordance([1, 2, 3], connected + [unrelated]), "parallax")

    def test_relation_thresholds_use_joint_vggt_geometry_for_interpolation(self) -> None:
        height = width = 8
        y, x = np.mgrid[:height, :width]
        points = np.stack((x, y, np.ones_like(x)), axis=-1).astype(np.float64)
        base = _PhotoArrays(
            extrinsic=np.hstack((np.eye(3), np.zeros((3, 1)))), intrinsic=np.eye(3),
            depth=np.ones((height, width)), depth_conf=np.ones((height, width)),
            point_map=points, point_conf=np.ones((height, width)), world_points=points,
        )
        shifted_extrinsic = np.hstack((np.eye(3), np.array([[-0.05], [0.0], [0.0]])))
        shifted = _PhotoArrays(
            extrinsic=shifted_extrinsic, intrinsic=np.eye(3),
            depth=np.ones((height, width)), depth_conf=np.ones((height, width)),
            point_map=points, point_conf=np.ones((height, width)), world_points=points,
        )
        relation = _compute_pair_relations({1: base, 2: shifted}, {1: "living", 2: "living"})[0]
        self.assertEqual(relation.continuity_type, "interpolation_safe")
        self.assertTrue(relation.is_connected)
        self.assertEqual(relation.debug_metrics["verification_source"], "vggt_joint_geometry")

    def test_wide_angle_views_remain_one_same_scene(self) -> None:
        angle = np.deg2rad(86.0)
        rotation = np.array([
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ])
        left = _PhotoArrays(
            extrinsic=np.hstack((np.eye(3), np.zeros((3, 1)))), intrinsic=np.eye(3),
            depth=np.ones((2, 2)), depth_conf=np.ones((2, 2)),
            point_map=np.ones((2, 2, 3)), point_conf=np.ones((2, 2)), world_points=np.ones((2, 2, 3)),
        )
        right = _PhotoArrays(
            extrinsic=np.hstack((rotation, np.array([[0.5], [0.0], [0.0]]))), intrinsic=np.eye(3),
            depth=np.ones((2, 2)), depth_conf=np.ones((2, 2)),
            point_map=np.ones((2, 2, 3)), point_conf=np.ones((2, 2)), world_points=np.ones((2, 2, 3)),
        )
        measured = {
            "visible_fraction": 0.40,
            "frustum_fraction": 0.84,
            "depth_consistency": 0.76,
            "reprojection_error": 0.011,
            "median_image_dx": 142.0,
            "median_image_dy": 4.0,
        }
        with patch.object(vggt_pipeline, "_project_metrics", side_effect=[measured, measured]):
            relation = _compute_pair_relations({1: left, 2: right}, {1: "dining room", 2: "dining room"})[0]
        self.assertEqual(relation.continuity_type, "same_scene")
        self.assertTrue(relation.is_connected)
        self.assertGreater(relation.relation_confidence, 0.45)

    def test_auto_room_labels_do_not_split_geometry_group(self) -> None:
        photos = [
            SimpleNamespace(room_override=None, room_label="dining room", manual_metadata={}),
            SimpleNamespace(room_override=None, room_label="garage", manual_metadata={}),
        ]
        groups = _derive_render_groups(photos, "interior")
        self.assertEqual(groups, [photos])

    def test_geometry_component_is_partitioned_into_pairs(self) -> None:
        photos = [
            SimpleNamespace(room_override=None, room_label="living room", manual_metadata={})
            for _ in range(5)
        ]
        groups = _derive_render_groups(photos, "interior")
        self.assertEqual([len(group) for group in groups], [2, 2, 1])

    def test_explicit_hero_remains_a_single_photo_group(self) -> None:
        photos = [
            SimpleNamespace(room_override=None, room_label="living room", manual_metadata={}),
            SimpleNamespace(room_override=None, room_label="living room", manual_metadata={"editorial_role": "hero"}),
            SimpleNamespace(room_override=None, room_label="living room", manual_metadata={}),
        ]
        groups = _derive_render_groups(photos, "interior")
        self.assertEqual([len(group) for group in groups], [1, 1, 1])

    def test_explicit_hero_and_single_image_fallback(self) -> None:
        auto = SimpleNamespace(id=1, final_score=0.99, position=0, manual_metadata={})
        hero = SimpleNamespace(id=2, final_score=0.10, position=1, manual_metadata={"editorial_role": "hero"})
        cluster = SimpleNamespace(hero_photo_id=None, room_type="living room", sfm_eligible=False, geometry_confidence=0.5, overlap_score=0.0, recommended_motion="subtle_pan", recommended_duration=3.0, id=7, scene_component_id=None)
        self.assertEqual(shot_planner._hero_photo(cluster, [auto, hero]).id, 2)
        shot = shot_planner._build_shot(cluster, [hero], None)
        self.assertEqual(shot["shot_type"], "single_image_move")
        self.assertTrue(shot["rejection_reasons"])

    def test_geometry_group_becomes_multi_view_storyboard_shot(self) -> None:
        first = SimpleNamespace(id=1, final_score=0.9, position=0, manual_metadata={})
        second = SimpleNamespace(id=2, final_score=0.8, position=1, manual_metadata={})
        cluster = SimpleNamespace(
            hero_photo_id=None, room_type="dining room", sfm_eligible=True,
            geometry_confidence=0.9, overlap_score=0.4, recommended_motion="parallax",
            recommended_duration=3.0, id=7, scene_component_id=4,
        )
        shot = shot_planner._build_shot(cluster, [first, second], None)
        self.assertEqual(shot["shot_type"], "verified_multi_view")
        self.assertEqual(shot["ordered_photo_ids"], [1, 2])

    def test_similar_neighbor_is_marked_skip_but_retained(self) -> None:
        first = {
            "cluster_id": 1, "order_index": 0, "ordered_photo_ids": [10],
            "skip_recommended": False,
            "evidence": {"hard_editorial_role": "auto", "hero_quality": 0.9},
        }
        second = {
            "cluster_id": 2, "order_index": 1, "ordered_photo_ids": [20],
            "skip_recommended": False,
            "evidence": {"hard_editorial_role": "auto", "hero_quality": 0.7},
        }
        duplicate = SimpleNamespace(
            continuity_type="same_scene", overlap_score=0.90, relation_confidence=0.90,
            relative_transform={"rotation_degrees": 5.0, "normalized_baseline": 0.03},
        )
        shots = [first, second]
        shot_planner._mark_redundant_neighbors(shots, {(10, 20): duplicate})
        self.assertEqual(len(shots), 2)
        self.assertFalse(first["skip_recommended"])
        self.assertTrue(second["skip_recommended"])
        self.assertEqual(second["duplicate_of_cluster_id"], 1)

    def test_transition_only_interpolates_verified_relation(self) -> None:
        previous = {"ordered_photo_ids": [1]}
        current = {"ordered_photo_ids": [2]}
        self.assertEqual(shot_planner._transition_type(previous, current, {(1, 2): _relation(1, 2)}), "interpolate")
        self.assertEqual(shot_planner._transition_type(previous, current, {}), "editorial_cut")

    def test_unsafe_orbit_request_falls_back_to_single_image_motion(self) -> None:
        photo = SimpleNamespace(manual_metadata={"camera_motion": "orbit_right"})
        cluster = SimpleNamespace(sfm_eligible=False, image_count=1)
        motion, reason = _requested_motion([photo], cluster, ["static", "micro_push_in", "subtle_pan"])
        self.assertEqual(motion, "micro_push_in")
        self.assertIn("single-image", reason)


if __name__ == "__main__":
    unittest.main()
