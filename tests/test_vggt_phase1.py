"""Unit coverage for the deterministic VGGT Phase 1 planning primitives."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.models import vggt as vggt_runtime
from app.pipeline.phase1_analyze import shot_planner
from app.pipeline.phase1_analyze.motion_planner import _requested_motion
from app.pipeline.phase1_analyze.vggt_pipeline import (
    PhotoRelationResult,
    _PhotoArrays,
    _estimate_similarity,
    _compute_pair_relations,
    _depth_metrics,
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

    def test_beam_search_is_deterministic_and_uses_verified_edges(self) -> None:
        relations = [_relation(1, 2), _relation(2, 3), _relation(3, 4)]
        positions = {1: 0, 2: 1, 3: 2, 4: 3}
        self.assertEqual(_order_component([1, 2, 3, 4], relations, positions), [1, 2, 3, 4])
        self.assertEqual(_order_component([1, 2, 3, 4], relations, positions), [1, 2, 3, 4])

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

    def test_explicit_hero_and_single_image_fallback(self) -> None:
        auto = SimpleNamespace(id=1, final_score=0.99, position=0, manual_metadata={})
        hero = SimpleNamespace(id=2, final_score=0.10, position=1, manual_metadata={"editorial_role": "hero"})
        cluster = SimpleNamespace(hero_photo_id=None, room_type="living room", sfm_eligible=False, geometry_confidence=0.5, overlap_score=0.0, recommended_motion="subtle_pan", recommended_duration=3.0, id=7, scene_component_id=None)
        self.assertEqual(shot_planner._hero_photo(cluster, [auto, hero]).id, 2)
        shot = shot_planner._build_shot(cluster, [hero], None)
        self.assertEqual(shot["shot_type"], "single_image_move")
        self.assertTrue(shot["rejection_reasons"])

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
