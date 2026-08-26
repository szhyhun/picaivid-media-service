"""Unit coverage for the V2-only VGGT Phase 1 planning primitives."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from app.models import vggt as vggt_runtime
from app.pipeline.phase1_analyze import shot_planner, vggt_pipeline
from app.pipeline.phase1_analyze.clustering import _derive_render_groups
from app.pipeline.phase1_analyze.motion_planner import _requested_motion
from app.pipeline.phase1_analyze.pose_graph import solve_component_poses
from app.pipeline.phase1_analyze.vggt_pipeline import (
    PhotoRelationResult,
    _motion_affordance,
    _order_component,
)


def _relation(
    left: int,
    right: int,
    confidence: float = 0.8,
    *,
    continuity: str = "same_scene",
    translation: list[float] | None = None,
) -> PhotoRelationResult:
    return PhotoRelationResult(
        photo_a_id=left,
        photo_b_id=right,
        overlap_score=0.7,
        reprojection_score=0.95,
        relation_confidence=confidence,
        baseline_distance=0.2,
        relative_transform={
            "relative_pose": {
                "rotation": np.eye(3).tolist(),
                "translation": translation or [-1.0, 0.0, 0.0],
                "scale": 1.0,
            }
        },
        direction_dx=12.0,
        direction_dy=0.0,
        continuity_type=continuity,
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

    def test_omega_depth_unprojection_uses_world_to_camera_pose(self) -> None:
        depth = torch.ones((1, 1, 1, 1), dtype=torch.float32)
        extrinsic = torch.tensor([[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0]]])
        intrinsic = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        points = vggt_runtime._unproject_depth(depth, extrinsic, intrinsic)
        np.testing.assert_allclose(points.numpy(), np.array([[[[-1.0, -2.0, -2.0]]]]))

    def test_pair_pose_graph_propagates_component_camera_centers(self) -> None:
        poses, diagnostics = solve_component_poses(
            [1, 2, 3],
            [_relation(1, 2), _relation(2, 3)],
        )
        np.testing.assert_allclose(poses[1].center, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(poses[2].center, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(poses[3].center, [2.0, 0.0, 0.0])
        self.assertEqual(diagnostics["missing_pose_count"], 0)
        self.assertEqual(diagnostics["pose_edges"], 2)

    def test_pose_graph_handles_reversed_relation_orientation(self) -> None:
        forward, _ = solve_component_poses([1, 2], [_relation(1, 2)])
        reverse, _ = solve_component_poses([1, 2], [_relation(2, 1)])
        self.assertAlmostEqual(
            float(np.linalg.norm(forward[2].center - forward[1].center)),
            float(np.linalg.norm(reverse[2].center - reverse[1].center)),
        )

    def test_pose_graph_prefers_strongest_spanning_edges(self) -> None:
        poses, diagnostics = solve_component_poses(
            [1, 2, 3],
            [
                _relation(1, 2, confidence=0.1, translation=[-10.0, 0.0, 0.0]),
                _relation(1, 3, confidence=0.9),
                _relation(3, 2, confidence=0.8),
            ],
        )
        np.testing.assert_allclose(poses[2].center, [2.0, 0.0, 0.0])
        self.assertEqual(diagnostics["pose_edges"], 2)
        self.assertEqual(diagnostics["cycle_edges"], 1)
        self.assertGreater(diagnostics["median_cycle_translation_error"], 0.0)

    def test_pipeline_never_runs_a_whole_listing_prediction(self) -> None:
        graph_result = ([], [[1]], [], {"coordinate_scope": "component_local", "computed_pairs": 0})
        with (
            patch.object(vggt_pipeline, "_v2_scene_graph", return_value=graph_result),
            patch.object(vggt_pipeline.vggt_model, "runtime_metadata", return_value={"model": "test"}),
            patch.object(vggt_pipeline.vggt_model, "predict", side_effect=AssertionError("global predict called")),
            patch.object(vggt_pipeline, "_release_accelerator_cache"),
        ):
            geometries, relations, components = vggt_pipeline.run_vggt_scene_pipeline(
                [Image.fromarray(np.full((16, 16, 3), 100, dtype=np.uint8))],
                [1],
                ["bedroom"],
                [0],
                job_id=7,
            )
        self.assertEqual(relations, [])
        self.assertEqual(len(components), 1)
        self.assertEqual(geometries[0].local_metrics["geometry_source"], "vggt_pairwise_pose_graph")
        self.assertEqual(components[0].debug_metrics["runtime"]["coordinate_scope"], "component_local")

    def test_beam_search_is_deterministic_and_uses_verified_edges(self) -> None:
        relations = [_relation(1, 2), _relation(2, 3), _relation(3, 4)]
        positions = {1: 0, 2: 1, 3: 2, 4: 3}
        self.assertEqual(_order_component([1, 2, 3, 4], relations, positions), [1, 2, 3, 4])
        self.assertEqual(_order_component([1, 2, 3, 4], relations, positions), [1, 2, 3, 4])

    def test_motion_affordance_requires_authorized_interpolation(self) -> None:
        self.assertEqual(_motion_affordance([1, 2], [_relation(1, 2)]), "micro_push_in")
        safe = _relation(1, 2, continuity="interpolation_safe")
        self.assertEqual(_motion_affordance([1, 2], [safe]), "multi_view")

    def test_auto_room_labels_do_not_split_geometry_group(self) -> None:
        photos = [
            SimpleNamespace(room_override=None, room_label="dining room", manual_metadata={}),
            SimpleNamespace(room_override=None, room_label="garage", manual_metadata={}),
        ]
        self.assertEqual(_derive_render_groups(photos, "interior"), [photos])

    def test_geometry_component_is_partitioned_into_pairs(self) -> None:
        photos = [
            SimpleNamespace(room_override=None, room_label="living room", manual_metadata={})
            for _ in range(5)
        ]
        self.assertEqual(
            [len(group) for group in _derive_render_groups(photos, "interior")],
            [2, 2, 1],
        )

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
        self.assertFalse(first["skip_recommended"])
        self.assertTrue(second["skip_recommended"])

    def test_transition_only_interpolates_verified_relation(self) -> None:
        previous = {"ordered_photo_ids": [1]}
        current = {"ordered_photo_ids": [2]}
        safe = _relation(1, 2, continuity="interpolation_safe")
        self.assertEqual(shot_planner._transition_type(previous, current, {(1, 2): safe}), "interpolate")
        self.assertEqual(shot_planner._transition_type(previous, current, {}), "editorial_cut")

    def test_unsafe_orbit_request_falls_back_to_single_image_motion(self) -> None:
        photo = SimpleNamespace(manual_metadata={"camera_motion": "orbit_right"})
        cluster = SimpleNamespace(sfm_eligible=False, image_count=1)
        motion, reason = _requested_motion([photo], cluster, ["static", "micro_push_in", "subtle_pan"])
        self.assertEqual(motion, "micro_push_in")
        self.assertIn("single-image", reason)


if __name__ == "__main__":
    unittest.main()
