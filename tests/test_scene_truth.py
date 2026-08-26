"""Validation rules for human-labeled scene ground truth (Scene-Graph V2, Stage 0)."""
from __future__ import annotations

import unittest

from app.main import _truth_validation_warnings
from app.schemas.scene_truth import RoomInstancePayload, SceneTruthSetPayload

VALID_KEYS = {"k1", "k2", "k3", "k4", "k5"}


def _payload(**overrides) -> SceneTruthSetPayload:
    base = {
        "room_instances": [
            RoomInstancePayload(instance="bedroom-a", photo_keys=["k1", "k2"]),
            RoomInstancePayload(instance="bedroom-b", photo_keys=["k3", "k4"]),
        ],
    }
    base.update(overrides)
    return SceneTruthSetPayload(**base)


class SceneTruthValidationTests(unittest.TestCase):
    def test_clean_payload_has_no_warnings(self):
        payload = _payload(duplicates=[["k1", "k2"]], preferred_pairs=[["k3", "k4"]], must_not_group=[["k1", "k3"]])
        self.assertEqual(_truth_validation_warnings(payload, VALID_KEYS), [])

    def test_photo_in_two_rooms_is_flagged(self):
        payload = _payload(room_instances=[
            RoomInstancePayload(instance="bedroom-a", photo_keys=["k1", "k2"]),
            RoomInstancePayload(instance="bedroom-b", photo_keys=["k2", "k3"]),
        ])
        warnings = _truth_validation_warnings(payload, VALID_KEYS)
        self.assertTrue(any("in two rooms" in warning for warning in warnings))

    def test_unknown_photo_is_flagged(self):
        payload = _payload(room_instances=[RoomInstancePayload(instance="bedroom-a", photo_keys=["k1", "ghost"])])
        warnings = _truth_validation_warnings(payload, VALID_KEYS)
        self.assertTrue(any("does not belong to this listing" in warning for warning in warnings))

    def test_duplicate_pair_must_share_a_room(self):
        payload = _payload(duplicates=[["k1", "k3"]])
        warnings = _truth_validation_warnings(payload, VALID_KEYS)
        self.assertTrue(any("not in the same room instance" in warning for warning in warnings))

    def test_must_not_group_pair_inside_one_room_is_contradictory(self):
        payload = _payload(must_not_group=[["k1", "k2"]])
        warnings = _truth_validation_warnings(payload, VALID_KEYS)
        self.assertTrue(any("must-not-group but share room" in warning for warning in warnings))

    def test_open_plan_group_must_reference_known_rooms(self):
        payload = _payload(open_plan_groups=[["bedroom-a", "kitchen-ghost"]])
        warnings = _truth_validation_warnings(payload, VALID_KEYS)
        self.assertTrue(any("unknown room" in warning for warning in warnings))

    def test_open_plan_group_needs_two_rooms(self):
        payload = _payload(open_plan_groups=[["bedroom-a"]])
        warnings = _truth_validation_warnings(payload, VALID_KEYS)
        self.assertTrue(any("at least two rooms" in warning for warning in warnings))

    def test_self_pair_is_rejected(self):
        payload = _payload(duplicates=[["k1", "k1"]])
        warnings = _truth_validation_warnings(payload, VALID_KEYS)
        self.assertTrue(any("cannot pair with itself" in warning for warning in warnings))

    def test_duplicate_room_names_are_flagged(self):
        payload = _payload(room_instances=[
            RoomInstancePayload(instance="bedroom-a", photo_keys=["k1"]),
            RoomInstancePayload(instance="bedroom-a", photo_keys=["k2"]),
        ])
        warnings = _truth_validation_warnings(payload, VALID_KEYS)
        self.assertTrue(any("Duplicate room instance name" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()


class OpenPlanPairTests(unittest.TestCase):
    """A cinematic pair may span an open-plan seam; a duplicate may not."""

    def _payload(self, **kw):
        base = {
            "room_instances": [
                RoomInstancePayload(instance="kitchen", photo_keys=["k1"]),
                RoomInstancePayload(instance="living", photo_keys=["k2"]),
            ],
            "open_plan_groups": [["kitchen", "living"]],
        }
        base.update(kw)
        return SceneTruthSetPayload(**base)

    def test_preferred_pair_across_open_plan_link_is_allowed(self):
        payload = self._payload(preferred_pairs=[["k1", "k2"]])
        self.assertEqual(_truth_validation_warnings(payload, VALID_KEYS), [])

    def test_preferred_pair_without_open_plan_link_is_flagged(self):
        payload = self._payload(open_plan_groups=[], preferred_pairs=[["k1", "k2"]])
        warnings = _truth_validation_warnings(payload, VALID_KEYS)
        self.assertTrue(any("not in the same room instance" in w for w in warnings))

    def test_duplicate_across_open_plan_link_is_still_flagged(self):
        payload = self._payload(duplicates=[["k1", "k2"]])
        warnings = _truth_validation_warnings(payload, VALID_KEYS)
        self.assertTrue(any("not in the same room instance" in w for w in warnings))


class ReadableWarningTests(unittest.TestCase):
    """Warnings must name photos the way the labeler sees them, not by UUID."""

    def test_warnings_use_tile_positions_not_uuids(self):
        payload = SceneTruthSetPayload(
            room_instances=[
                RoomInstancePayload(instance="kitchen", photo_keys=["k1"]),
                RoomInstancePayload(instance="bathroom", photo_keys=["k2"]),
            ],
            preferred_pairs=[["k1", "k2"]],
        )
        describe = {"k1": "#39", "k2": "#46"}
        warnings = _truth_validation_warnings(payload, VALID_KEYS, describe)
        self.assertEqual(len(warnings), 1)
        self.assertIn("#39", warnings[0])
        self.assertIn("#46", warnings[0])
        self.assertNotIn("k1", warnings[0])

    def test_unknown_photo_is_named_gracefully(self):
        payload = SceneTruthSetPayload(
            room_instances=[RoomInstancePayload(instance="kitchen", photo_keys=["deadbeef-not-here"])]
        )
        warnings = _truth_validation_warnings(payload, VALID_KEYS, {"k1": "#1"})
        self.assertTrue(any("unknown photo deadbeef" in w for w in warnings))
