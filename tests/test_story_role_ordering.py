"""Story-role bucketing for every ROOM_TYPES value.

`_story_role` matches on substrings of `cluster.room_type`, which made three
categories sort wrongly:
  * "aerial view" contains no "drone" token and fell through to
    detail_or_service, so drone openers landed late in the tour.
  * "exterior back" matched the generic "exterior" token before the yard/patio
    check and became front_exterior, so a tour could open on the back of the house.
  * "entrance" did not match the shorter "entry" token and fell through to
    detail_or_service.
Both surfaced to the owner as "you don't start from the front of the house".
"""
from __future__ import annotations

import unittest

from app.models.openclip import ROOM_TYPES
from app.pipeline.phase1_analyze.shot_planner import (
    _order_component_blocks,
    _sort_key,
    _story_role,
)


class _Cluster:
    def __init__(self, room_type: str, *, cluster_id: int = 1,
                 component_id: int | None = None, sequence_order: int = 0) -> None:
        self.room_type = room_type
        self.id = cluster_id
        self.hero_photo_id = None
        self.scene_component_id = component_id
        self.sequence_order = sequence_order


class _Photo:
    manual_metadata: dict = {}
    position = 0
    id = 1
    final_score = 0.5


EXPECTED = {
    "aerial view": "drone_opener",
    "drone shot": "drone_opener",
    "exterior front": "front_exterior",
    "front yard": "front_exterior",
    "exterior back": "outdoor_payoff",
    "backyard": "outdoor_payoff",
    "patio": "outdoor_payoff",
    "pool": "outdoor_payoff",
    "entrance": "approach_entry",
    "hallway": "approach_entry",
    "living room": "social_room",
    "kitchen": "social_room",
    "dining room": "social_room",
    "bedroom": "private_room",
    "bathroom": "private_room",
    "office": "detail_or_service",
    "garage": "detail_or_service",
    "storage": "detail_or_service",
    "laundry room": "detail_or_service",
    "basement": "detail_or_service",
    "attic": "detail_or_service",
}


def _role(room_type: str) -> str:
    photo = _Photo()
    return _story_role(_Cluster(room_type), [photo], photo)


def _order(room_type: str) -> int:
    photo = _Photo()
    cluster = _Cluster(room_type)
    shot = {"story_role": _role(room_type), "confidence": 0.5}
    return _sort_key(cluster, [photo], shot)[0]


class StoryRoleBucketTests(unittest.TestCase):
    def test_every_room_type_buckets_as_expected(self):
        for room_type in sorted(ROOM_TYPES):
            with self.subTest(room_type=room_type):
                self.assertEqual(_role(room_type), EXPECTED[room_type])

    def test_all_room_types_are_covered(self):
        """A new ROOM_TYPES entry must be given an explicit bucket here."""
        self.assertEqual(set(ROOM_TYPES), set(EXPECTED))

    def test_aerial_opens_before_front_which_opens_before_back(self):
        self.assertLess(_order("aerial view"), _order("exterior front"))
        self.assertLess(_order("exterior front"), _order("exterior back"))
        self.assertLess(_order("front yard"), _order("backyard"))

    def test_front_tokens_are_not_stolen_by_the_rear_check(self):
        """'front yard' contains 'yard' but must still lead, not become payoff."""
        self.assertEqual(_role("front yard"), "front_exterior")


def _candidate(room_type, cluster_id, component_id, sequence_order):
    """Build the (sort_key, cluster, shot) triple `_order_component_blocks` consumes."""
    cluster = _Cluster(room_type, cluster_id=cluster_id,
                       component_id=component_id, sequence_order=sequence_order)
    photo = _Photo()
    shot = {"story_role": _role(room_type), "confidence": 0.5, "cluster_id": cluster_id}
    return (_sort_key(cluster, [photo], shot), cluster, shot)


class ComponentBlockOrderingTests(unittest.TestCase):
    """Story stage must outrank geometry-component grouping.

    Front and rear exteriors often reconstruct into one scene component. Grouping
    on the component alone re-sorted a block by `cluster.sequence_order` and could
    emit the rear of the house first -- the owner's original report survived the
    `_story_role` fix in exactly this case.
    """

    def test_rear_first_component_is_emitted_front_before_rear(self):
        # Same component; the rear shot sits at component sequence 0.
        rear = _candidate("exterior back", cluster_id=1, component_id=7, sequence_order=0)
        front = _candidate("exterior front", cluster_id=2, component_id=7, sequence_order=1)
        ordered = _order_component_blocks([rear, front])
        self.assertEqual([shot["story_role"] for _, _, shot in ordered],
                         ["front_exterior", "outdoor_payoff"])

    def test_same_stage_shots_keep_component_local_order(self):
        later = _candidate("bedroom", cluster_id=1, component_id=3, sequence_order=5)
        earlier = _candidate("bathroom", cluster_id=2, component_id=3, sequence_order=1)
        ordered = _order_component_blocks([later, earlier])
        self.assertEqual([cluster.id for _, cluster, _ in ordered], [2, 1])

    def test_opening_and_closing_stay_absolute(self):
        closing = _candidate("kitchen", cluster_id=1, component_id=9, sequence_order=0)
        closing[2]["story_role"] = "closing"
        closing = ((99,) + closing[0][1:], closing[1], closing[2])
        front = _candidate("exterior front", cluster_id=2, component_id=9, sequence_order=5)
        ordered = _order_component_blocks([closing, front])
        self.assertEqual([shot["story_role"] for _, _, shot in ordered],
                         ["front_exterior", "closing"])


if __name__ == "__main__":
    unittest.main()
