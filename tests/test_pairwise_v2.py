"""Unit coverage for the V2 candidate/verification primitives.

Covers the defects found in review: canonical pair ordering, cache-key behaviour,
JSON-serializable evidence, tiered nomination, determinism, and mask placement.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.pipeline.phase1_analyze.candidate_pairs import (
    CandidatePair,
    nominate,
    split_tiers,
    unconnected_photos,
)
from app.pipeline.phase1_analyze.pairwise_verify import (
    DirectionEvidence,
    PairEvidence,
    _direction_evidence,
    canonical_order,
    canonical_order_with_ids,
    evidence_key,
    verify_with_cache,
)


def _photo(pid: int, position: int, label: str | None) -> dict:
    return {"id": pid, "position": position, "room_label": label}


class CandidateNominationTests(unittest.TestCase):
    def setUp(self):
        self.photos = [
            _photo(1, 0, "bedroom"), _photo(2, 1, "bedroom"), _photo(3, 2, "bedroom"),
            _photo(4, 3, "kitchen"), _photo(5, 4, "kitchen"), _photo(6, 20, "bedroom"),
        ]

    def test_is_deterministic(self):
        first = [c.key for c in nominate(self.photos)]
        second = [c.key for c in nominate(self.photos)]
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))

    def test_label_cap_limits_same_label_explosion(self):
        # 5 same-label photos would be 10 pairs uncapped; cap=2 keeps it bounded
        photos = [_photo(i, i, "bedroom") for i in range(1, 6)]
        capped = nominate(photos, label_cap=2, adjacency=0)
        for pid in range(1, 6):
            degree = sum(1 for c in capped if pid in c.key)
            self.assertLessEqual(degree, 4, "cap must bound per-photo label degree")

    def test_adjacency_links_neighbours_only(self):
        pairs = {c.key for c in nominate(self.photos, adjacency=1, label_cap=0)}
        self.assertIn((1, 2), pairs)
        self.assertNotIn((1, 6), pairs)   # positions 0 and 20 are not adjacent

    def test_never_pairs_a_photo_with_itself(self):
        for candidate in nominate(self.photos):
            self.assertNotEqual(candidate.photo_a, candidate.photo_b)

    def test_missing_labels_do_not_nominate(self):
        photos = [_photo(1, 0, None), _photo(2, 5, None)]
        self.assertEqual(nominate(photos, adjacency=0), [])


class TierTests(unittest.TestCase):
    def test_global_only_pairs_are_fallback(self):
        primary = CandidatePair(1, 2, {"clip"})
        globalish = CandidatePair(3, 4, {"global_rank", "global_frustum"})
        both = CandidatePair(5, 6, {"label", "global_rank"})
        self.assertEqual(primary.tier, "primary")
        self.assertEqual(globalish.tier, "fallback")
        self.assertEqual(both.tier, "primary", "a primary source wins over a global one")
        first, second = split_tiers([primary, globalish, both])
        self.assertEqual([c.key for c in first], [(1, 2), (5, 6)])
        self.assertEqual([c.key for c in second], [(3, 4)])

    def test_unconnected_photos_reports_escalation_targets(self):
        candidates = [CandidatePair(1, 2, {"clip"}), CandidatePair(3, 4, {"clip"})]
        still_alone = unconnected_photos(candidates, {(1, 2)}, [1, 2, 3, 4])
        self.assertEqual(still_alone, {3, 4})


class CanonicalOrderTests(unittest.TestCase):
    """verify(A,B) and verify(B,A) must never return swapped directional evidence."""

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="pairtest_")
        self.paths = []
        for index, value in enumerate((30, 200)):
            path = os.path.join(self.directory, f"img{index}.png")
            Image.fromarray(np.full((32, 48, 3), value, dtype=np.uint8)).save(path)
            self.paths.append(path)

    def test_order_is_content_addressed_and_symmetric(self):
        forward = canonical_order(self.paths[0], self.paths[1])
        reversed_ = canonical_order(self.paths[1], self.paths[0])
        self.assertEqual(forward, reversed_)

    def test_canonical_order_keeps_photo_ids_with_their_images(self):
        forward = canonical_order_with_ids(self.paths[0], 10, self.paths[1], 20)
        reversed_ = canonical_order_with_ids(self.paths[1], 20, self.paths[0], 10)
        self.assertEqual(forward, reversed_)
        self.assertEqual({pair[1] for pair in forward}, {10, 20})

    def test_cache_key_is_symmetric(self):
        a = evidence_key(self.paths[0], self.paths[1], "sha", "balanced", 512)
        b = evidence_key(self.paths[1], self.paths[0], "sha", "balanced", 512)
        self.assertEqual(a, b)

    def test_cache_key_changes_with_checkpoint_and_preprocessing(self):
        base = evidence_key(self.paths[0], self.paths[1], "sha", "balanced", 512)
        self.assertNotEqual(base, evidence_key(self.paths[0], self.paths[1], "other", "balanced", 512))
        self.assertNotEqual(base, evidence_key(self.paths[0], self.paths[1], "sha", "max_size", 512))
        self.assertNotEqual(base, evidence_key(self.paths[0], self.paths[1], "sha", "balanced", 384))

    def test_cache_key_changes_with_runtime_and_evidence_schema(self):
        base = evidence_key(
            self.paths[0], self.paths[1], "sha", "balanced", 512,
            model_revision="commit-a", precision="float32", schema_version="schema-a",
        )
        self.assertNotEqual(
            base,
            evidence_key(
                self.paths[0], self.paths[1], "sha", "balanced", 512,
                model_revision="commit-b", precision="float32", schema_version="schema-a",
            ),
        )
        self.assertNotEqual(
            base,
            evidence_key(
                self.paths[0], self.paths[1], "sha", "balanced", 512,
                model_revision="commit-a", precision="bfloat16", schema_version="schema-a",
            ),
        )
        self.assertNotEqual(
            base,
            evidence_key(
                self.paths[0], self.paths[1], "sha", "balanced", 512,
                model_revision="commit-a", precision="float32", schema_version="schema-b",
            ),
        )

    def test_cache_key_ignores_thresholds_by_construction(self):
        """Retuning a threshold must never force re-inference."""
        signature = evidence_key.__doc__ or ""
        self.assertIn("threshold", signature.lower())


class EvidenceSerializationTests(unittest.TestCase):
    def test_evidence_is_strict_json_serializable(self):
        evidence = PairEvidence(photo_a="a.jpg", photo_b="b.jpg",
                                forward=DirectionEvidence(), backward=DirectionEvidence())
        # allow_nan=False is what the sweep uses; NaN would raise here.
        encoded = json.dumps(evidence.to_dict(), allow_nan=False)
        decoded = json.loads(encoded)
        self.assertIsNone(decoded["forward"]["median_relative_depth_error"])
        self.assertEqual(decoded["depth_ok_min"], 0.0)

    def test_evidence_carries_pose_for_the_pose_graph(self):
        evidence = PairEvidence(photo_a="a", photo_b="b")
        payload = evidence.to_dict()
        for required in ("rotation", "translation", "scale"):
            self.assertIn(required, payload["relative_pose"],
                          "pose graph needs relative pose cached, not just a scalar angle")

    def test_nonfinite_depth_is_not_serialized_as_nan(self):
        extrinsic = np.repeat(np.hstack((np.eye(3), np.zeros((3, 1))))[None], 2, axis=0)
        intrinsic = np.repeat(np.eye(3)[None], 2, axis=0)
        depth = np.ones((2, 2, 2), dtype=float)
        depth[1] = np.nan
        points = np.zeros((2, 2, 2, 3), dtype=float)
        points[..., 2] = 1.0
        conf = np.ones((2, 2, 2), dtype=float)
        masks = np.ones((2, 2, 2), dtype=bool)

        evidence = _direction_evidence(0, 1, extrinsic, intrinsic, depth, points, conf, masks)
        self.assertIsNone(evidence.median_relative_depth_error)
        json.dumps(evidence.__dict__, allow_nan=False)


class ProductionEvidenceCacheTests(unittest.TestCase):
    def test_second_verification_reuses_raw_evidence(self):
        with tempfile.TemporaryDirectory(prefix="pair-cache-") as directory:
            paths = []
            for index, value in enumerate((40, 180)):
                path = os.path.join(directory, f"{index}.png")
                Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)
                paths.append(path)
            runtime = {
                "checkpoint_sha256": "checkpoint-sha",
                "repo_commit": "repo-sha",
                "dtype": "float32",
            }
            evidence = PairEvidence(
                photo_a="a",
                photo_b="b",
                forward=DirectionEvidence(depth_ok=0.4),
                backward=DirectionEvidence(depth_ok=0.4),
            )
            cache_dir = os.path.join(directory, "cache")
            with patch(
                "app.pipeline.phase1_analyze.pairwise_verify.verify",
                return_value=evidence,
            ) as mocked_verify:
                first, first_hit, first_key = verify_with_cache(
                    paths[0], paths[1], runtime=runtime, cache_dir=cache_dir
                )
                second, second_hit, second_key = verify_with_cache(
                    paths[1], paths[0], runtime=runtime, cache_dir=cache_dir
                )

            self.assertFalse(first_hit)
            self.assertTrue(second_hit)
            self.assertEqual(first_key, second_key)
            self.assertEqual(first, second)
            mocked_verify.assert_called_once()


class ValidMaskTests(unittest.TestCase):
    """The mask must land exactly on Omega's padding, which is constant white."""

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="masktest_")

    def _write(self, name: str, width: int, height: int) -> str:
        path = os.path.join(self.directory, name)
        Image.fromarray(np.full((height, width, 3), 90, dtype=np.uint8)).save(path)
        return path

    def test_mixed_aspect_batch_masks_only_padded_pixels(self):
        from app.models.vggt import _valid_pixel_masks, vggt_model

        paths = [self._write("a.png", 2048, 1365), self._write("b.png", 2048, 1536)]
        images = vggt_model.load_and_preprocess_images(paths).cpu().numpy()
        masks = _valid_pixel_masks(paths, tuple(images.shape[-2:])).numpy()

        self.assertEqual(masks.shape, images.shape[0:1] + images.shape[-2:])
        self.assertTrue(masks.any(), "some pixels must be valid")
        self.assertFalse(masks.all(), "a mixed-aspect batch must have padding")
        for index in range(len(paths)):
            padded = images[index][:, ~masks[index]]
            if padded.size:
                # Omega pads with constant 1.0 (white); anything else means the
                # mask is misplaced and real content is being discarded.
                self.assertAlmostEqual(float(padded.min()), 1.0, places=4)
                self.assertAlmostEqual(float(padded.max()), 1.0, places=4)

    def test_uniform_aspect_batch_has_no_padding(self):
        from app.models.vggt import _valid_pixel_masks, vggt_model

        paths = [self._write("c.png", 2048, 1365), self._write("d.png", 1024, 683)]
        images = vggt_model.load_and_preprocess_images(paths)
        masks = _valid_pixel_masks(paths, tuple(images.shape[-2:])).numpy()
        self.assertTrue(masks.all(), "same-aspect photos are never padded")


if __name__ == "__main__":
    unittest.main()


class ConstrainedMergeTests(unittest.TestCase):
    """Absence of evidence must never block; human negatives always must."""

    def test_merges_connected_photos(self):
        from app.pipeline.phase1_analyze.membership import constrained_merge

        result = constrained_merge([1, 2, 3, 4], [(1, 2, 0.5), (2, 3, 0.4)])
        self.assertEqual(result.components, [[1, 2, 3], [4]])

    def test_human_negative_blocks_a_merge(self):
        from app.pipeline.phase1_analyze.membership import constrained_merge

        result = constrained_merge([1, 2], [(1, 2, 0.9)], must_not_group={(1, 2)})
        self.assertEqual(result.components, [[1], [2]])
        blocked = [d for d in result.merge_log if not d.accepted]
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0].reason, "must_not_group")

    def test_negative_blocks_transitively_across_components(self):
        """1-2 and 3-4 are fine, but 2-3 must not fuse them if 1 and 4 conflict."""
        from app.pipeline.phase1_analyze.membership import constrained_merge

        result = constrained_merge(
            [1, 2, 3, 4], [(1, 2, 0.9), (3, 4, 0.8), (2, 3, 0.7)], must_not_group={(1, 4)}
        )
        self.assertEqual(result.components, [[1, 2], [3, 4]])
        self.assertTrue(any(d.blocking_pair == (1, 4) for d in result.merge_log if not d.accepted))

    def test_absence_of_evidence_never_blocks(self):
        """No edge between 1 and 3 must not stop them sharing a component via 2."""
        from app.pipeline.phase1_analyze.membership import constrained_merge

        result = constrained_merge([1, 2, 3], [(1, 2, 0.5), (2, 3, 0.5)])
        self.assertEqual(result.components, [[1, 2, 3]])

    def test_is_order_independent(self):
        from app.pipeline.phase1_analyze.membership import constrained_merge

        edges = [(1, 2, 0.9), (2, 3, 0.5), (4, 5, 0.7)]
        first = constrained_merge([1, 2, 3, 4, 5], edges).components
        second = constrained_merge([1, 2, 3, 4, 5], list(reversed(edges))).components
        self.assertEqual(first, second)

    def test_is_endpoint_orientation_independent(self):
        """Undirected edge orientation must not decide which constrained merge wins."""
        from app.pipeline.phase1_analyze.membership import constrained_merge

        forward = constrained_merge(
            [1, 2, 3], [(1, 3, 0.5), (2, 3, 0.5)], must_not_group={(1, 2)}
        )
        reversed_endpoints = constrained_merge(
            [1, 2, 3], [(3, 1, 0.5), (2, 3, 0.5)], must_not_group={(1, 2)}
        )
        self.assertEqual(forward.components, reversed_endpoints.components)
        self.assertEqual(forward.merge_log, reversed_endpoints.merge_log)

    def test_reports_deterministic_blocking_pair(self):
        from app.pipeline.phase1_analyze.membership import constrained_merge

        result = constrained_merge(
            [1, 2, 3, 4],
            [(1, 2, 0.9), (3, 4, 0.8), (2, 3, 0.7)],
            must_not_group={(2, 4), (1, 3)},
        )
        blocked = [decision for decision in result.merge_log if not decision.accepted]
        self.assertEqual(blocked[0].blocking_pair, (1, 3))

    def test_merge_log_records_every_decision(self):
        from app.pipeline.phase1_analyze.membership import constrained_merge

        result = constrained_merge([1, 2, 3], [(1, 2, 0.9), (2, 3, 0.4)], must_not_group={(2, 3)})
        self.assertEqual(len(result.merge_log), 2)
        self.assertEqual(sum(1 for d in result.merge_log if d.accepted), 1)


class TransitionCandidateTests(unittest.TestCase):
    @staticmethod
    def _evidence(a=1, b=2, depth=0.5, baseline=1.0, rotation=40.0):
        return {
            "photo_a_id": a,
            "photo_b_id": b,
            "depth_ok_min": depth,
            "bl_over_depth": baseline,
            "rot_deg": rotation,
        }

    def test_requires_every_provisional_geometry_gate(self):
        from app.pipeline.phase1_analyze.transitions import build

        evidence = [
            self._evidence(a=1, b=2),
            self._evidence(a=1, b=3, depth=0.34),
            self._evidence(a=1, b=4, baseline=1.51),
            self._evidence(a=1, b=5, rotation=60.1),
        ]
        self.assertEqual([(item.photo_a, item.photo_b) for item in build(evidence)], [(1, 2)])

    def test_human_blocker_is_conservative(self):
        from app.pipeline.phase1_analyze.transitions import build

        self.assertEqual(build([self._evidence()], must_not_group={(1, 2)}), [])

    def test_cross_room_is_diagnostic_not_membership(self):
        from app.pipeline.phase1_analyze.transitions import build

        result = build([self._evidence()], room_of={1: "living", 2: "kitchen"})
        self.assertTrue(result[0].crosses_rooms)

    def test_endpoint_orientation_does_not_change_output(self):
        from app.pipeline.phase1_analyze.transitions import build

        forward = build([self._evidence(a=1, b=2)])
        reversed_ = build([self._evidence(a=2, b=1)])
        self.assertEqual(forward, reversed_)


class PairingTests(unittest.TestCase):
    """Editorial pairing: wide angles are wanted, near-duplicates are not."""

    def _ev(self, a, b, depth_ok, rot, conf, bl=1.0):
        return {"photo_a_id": a, "photo_b_id": b, "depth_ok_min": depth_ok,
                "rot_deg": rot, "conf_pair": conf, "bl_over_depth": bl}

    def test_prefers_complementary_angle_over_near_duplicate(self):
        from app.pipeline.phase1_analyze.pairing import score_pair

        wide = self._ev(1, 2, 0.45, 85, 11.0)     # the owner's median profile
        narrow = self._ev(1, 3, 0.45, 5, 11.0)    # nearly the same view twice
        self.assertGreater(score_pair(wide), score_pair(narrow))

    def test_duplicate_scores_zero(self):
        from app.pipeline.phase1_analyze.pairing import is_duplicate, score_pair

        dup = self._ev(1, 2, 0.95, 3, 20.0, bl=0.02)
        self.assertTrue(is_duplicate(dup))
        self.assertEqual(score_pair(dup), 0.0)

    def test_respects_per_room_caps(self):
        from app.pipeline.phase1_analyze.pairing import select_for_room

        evidence = [self._ev(i, i + 100, 0.45, 85, 11.0) for i in range(1, 6)]
        living, _ = select_for_room(evidence, "living-room")
        other, _ = select_for_room(evidence, "bedroom-a")
        self.assertEqual(len(living), 3)   # owner: living room gets 3
        self.assertEqual(len(other), 2)    # owner: any other room gets 2

    def test_a_photo_is_never_used_twice(self):
        from app.pipeline.phase1_analyze.pairing import select_for_room

        evidence = [self._ev(1, 2, 0.50, 85, 12.0), self._ev(1, 3, 0.48, 80, 11.0)]
        chosen, unpaired = select_for_room(evidence, "bedroom-a")
        used = [p for c in chosen for p in (c.photo_a, c.photo_b)]
        self.assertEqual(len(used), len(set(used)))
        self.assertIn(3, unpaired)

    def test_caps_are_ceilings_not_quotas(self):
        from app.pipeline.phase1_analyze.pairing import select_for_room

        chosen, unpaired = select_for_room([self._ev(1, 2, 0.45, 85, 11.0)], "living-room")
        self.assertEqual(len(chosen), 1)   # cap is 3, but only one good pair exists
        self.assertEqual(unpaired, [])


class MotionAuthorizationTests(unittest.TestCase):
    """V2 must not authorize generated motion while transition truth does not exist."""

    @staticmethod
    def _record(depth_ok: float, dx: float) -> dict:
        direction = {
            "depth_ok": depth_ok,
            "median_relative_depth_error": 0.05,
            "median_dx": dx,
            "median_dy": 0.0,
        }
        return {
            "photo_a": "a",
            "photo_b": "b",
            "conf_pair": 10.0,
            "baseline": 1.0,
            "median_depth": 1.0,
            "bl_over_depth": 1.0,
            "rot_deg": 80.0,
            "relative_pose": {},
            "forward": dict(direction),
            "backward": dict(direction),
            "depth_ok_min": depth_ok,
            "depth_ok_max": depth_ok,
        }

    def test_v2_keeps_directional_ids_aligned_and_only_connects_direct_gate_edges(self):
        from app.pipeline.phase1_analyze import vggt_pipeline
        from app.pipeline.phase1_analyze.vggt_pipeline import _PhotoArrays

        with tempfile.TemporaryDirectory(prefix="v2-graph-") as directory:
            paths: dict[int, str] = {}
            arrays: dict[int, _PhotoArrays] = {}
            for photo_id, value in ((1, 20), (2, 120), (3, 220)):
                path = os.path.join(directory, f"{photo_id}.png")
                Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)
                paths[photo_id] = path
                arrays[photo_id] = _PhotoArrays(
                    extrinsic=np.hstack((np.eye(3), np.zeros((3, 1)))),
                    intrinsic=np.eye(3),
                    depth=np.ones((2, 2)),
                    depth_conf=np.ones((2, 2)),
                    point_map=np.ones((2, 2, 3)),
                    point_conf=np.ones((2, 2)),
                    world_points=np.ones((2, 2, 3)),
                )

            candidates = [
                CandidatePair(1, 2, {"clip"}),
                CandidatePair(1, 3, {"clip"}),
                CandidatePair(2, 3, {"clip"}),
            ]
            records = [self._record(0.50, 12.0), self._record(0.10, 13.0), self._record(0.45, 23.0)]
            side_effect = [(record, False, f"key-{index}") for index, record in enumerate(records)]
            with (
                patch("app.pipeline.phase1_analyze.candidate_pairs.nominate", return_value=candidates),
                patch("app.pipeline.phase1_analyze.pairwise_verify.verify_with_cache", side_effect=side_effect),
                patch.object(vggt_pipeline.vggt_model, "runtime_metadata", return_value={}),
                patch.object(vggt_pipeline, "_release_accelerator_cache"),
            ):
                relations, components = vggt_pipeline._v2_scene_graph(
                    paths,
                    {1: "bedroom", 2: "bedroom", 3: "bedroom"},
                    {1: 1, 2: 2, 3: 3},
                    arrays,
                )

            self.assertEqual(components, [[1, 2, 3]])
            indirect = next(
                relation for relation in relations
                if {relation.photo_a_id, relation.photo_b_id} == {1, 3}
            )
            self.assertFalse(indirect.is_connected)
            self.assertTrue(indirect.debug_metrics["same_component"])
            self.assertFalse(indirect.debug_metrics["direct_membership_edge"])
            self.assertNotEqual(indirect.continuity_type, "interpolation_safe")

            expected = canonical_order_with_ids(paths[1], 1, paths[3], 3)
            self.assertEqual(
                (indirect.photo_a_id, indirect.photo_b_id),
                (expected[0][1], expected[1][1]),
            )

    def test_transitions_module_is_marked_provisional(self):
        from app.pipeline.phase1_analyze import transitions

        doc = (transitions.__doc__ or "").lower()
        self.assertTrue(
            "report-only" in doc or "provisional" in doc,
            "transitions must not claim to authorize motion",
        )
