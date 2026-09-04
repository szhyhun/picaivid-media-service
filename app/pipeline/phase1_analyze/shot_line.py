"""Build the ordered line of shots that becomes the film.

Takes the rooms a listing actually has and produces an ordered sequence of
shots, each one either a verified pair (two real photos, used as generation
keyframes) or a single photo. This is the hand-off point to video generation.

Ordering is measured, not invented: every shot in three professionally produced
listing films was classified on its fitted cut grid, and each room type's median
position across all three drives `STORY_POSITION` below. The shape that produced:

    front yard -> facade -> entrance -> LIVING -> kitchen/dining
      -> bedrooms/baths -> OUTDOOR (the largest block) -> close on exterior

Three earlier assumptions were overturned by that measurement and are recorded
here so they are not reintroduced:

1. **The living room is the first interior space** in all three films -- before
   the kitchen, before any bedroom.
2. **Outdoor space is the payoff, not punctuation.** It is the largest bucket by
   far (40 of 142 shots) and sits late; Anthony Pl closes on eleven consecutive
   outdoor shots.
3. **There is no mid-film hero motif.** Aerial is absent from one film entirely
   and bookends the other two. An earlier version treated it as a recurring
   theme and inserted it mid-tour, which the data does not support.

Service rooms are excluded by default. The reference films do show a garage for
about 1.5s, but the owner's instruction is explicit and repeated, and
`allow_service=True` remains for the studio-apartment case where excluding them
would leave nothing to cut.

Close-ups are manufactured from selected wide frames after this line is built.
They remain tied to the room and frame immediately before them, so the detail
reads as a continuation rather than an unrelated image.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from app.pipeline.phase1_analyze.camera_move import CameraMove, choose_move
from app.pipeline.phase1_analyze.detail_crop import ResolvedDetail, close_up_rate
from app.pipeline.phase1_analyze.editorial import (
    MIN_SHOTS,
    Tier,
    is_eligible,
    shot_budget,
    shot_seconds,
    tier_for,
)

# Share of runtime each room type earned in the reference film. Used to
# distribute shots when a property has more rooms than the budget allows.
# Anything not listed falls to DEFAULT_WEIGHT.
DWELL_WEIGHT: dict[str, float] = {
    "aerial view": 0.8,        # an establishing/bookend image, not a motif
    "patio": 2.1,
    "pool": 2.1,
    "backyard": 1.6,
    "living room": 1.6,
    "kitchen": 1.4,
    "bathroom": 1.3,
    "bedroom": 1.2,
    "dining room": 0.9,
    "exterior front": 0.8,
    "front yard": 0.8,
    "building exterior": 0.8,
    "amenities": 1.2,
    "office": 0.8,
    "exterior back": 0.8,
    "entrance": 0.5,
    "hallway": 0.3,
    "shared space": 0.3,
}
DEFAULT_WEIGHT = 0.6
SERVICE_WEIGHT = 0.25          # garage got 3% of runtime: present, minimal

# No individual selling feature should take over the film. The strongest room
# may receive up to four consecutive looks, but never a mid-film return.
MAX_HERO_SHOTS = 4
# One short service beat for the whole film, as observed.
MAX_SERVICE_SHOTS = 1
# Wallace Dr spent 2s on the front yard before going inside, and 5 of its 16
# rooms were exterior. Establishing is an approach, not a tour of the outside.
MAX_ACT1_SHOTS = 4
# The reference film averaged 2.2 shots per room. Letting one room take three
# while another takes one produces a film that keeps returning to the same
# kitchen instead of showing the house.
MAX_SHOTS_PER_ROOM = 2
# A house with four bathrooms should not spend a quarter of the film in them.
# The reference film gave bathrooms 13% of runtime across the whole property,
# not per room. Show the best instances of a type and drop the rest.
MAX_ROOMS_PER_TYPE: dict[str, int] = {
    "bathroom": 2,
    "bedroom": 3,
    "hallway": 1,
    "entrance": 1,
}
DEFAULT_MAX_ROOMS_PER_TYPE = 2

# Close-ups, measured by scoring every shot in the three reference films against
# the wide-room centroid and taking each film's top quartile:
#
#   Wallace Dr    10 of 40 shots (25%)   median gap 3
#   Whitehead Pl  13 of 53 (25%)         median gap 2
#   Anthony Pl    12 of 49 (24%)         median gap 2
#
# About one shot in four, recurring every two to three shots. They thin out at
# the very start and end -- the opening approach and the closing run are wides --
# so they are inserted through the body of the film only.
CLOSE_UP_SHARE = 0.25
# A close-up must belong to the room it follows: cutting from a wide living room
# to that same living room's fireplace reads as one continuous look, whereas a
# detail from elsewhere reads as an unrelated picture.

# Where each kind of room sits in the film, measured from all three reference
# films by classifying the middle frame of every shot on the fitted cut grid and
# taking the median normalised position (0 = first shot, 1 = last).
#
#   EXT-FRONT 0.05 (3/3 films)   BEDROOM  0.59 (3/3)
#   LIVING    0.19 (3/3)         BATH     0.60 (3/3)
#   ENTRY     0.25 (3/3)         OUTDOOR  0.71 (3/3, 40 shots -- the largest bucket)
#   DINING    0.29 (3/3)         SERVICE  0.87 (3/3)
#   KITCHEN   0.31 (3/3)         AERIAL   bimodal: opens or closes
#
# Three findings overturned the earlier hand-written ordering:
#
# 1. The living room is the first interior space in all three films, before the
#    kitchen and before any bedroom.
# 2. Outdoor space is the *largest* bucket and sits late -- Anthony Pl ends on
#    eleven consecutive outdoor shots. It is the payoff, not scattered filler.
# 3. Aerial is neither universal nor central: Anthony Pl contains none at all,
#    and where it appears it either opens the film or closes it. Treating it as
#    a mid-film motif was wrong.
STORY_POSITION: dict[str, float] = {
    "aerial view": 0.02,        # opens the film where present
    # The measurement merged front yard, exterior front and building exterior
    # into one EXT-FRONT bucket at 0.05, so it says nothing about their internal
    # order. Spatial logic settles it: you arrive from the street, reach the
    # facade and its door, then step through that same door.
    "front yard": 0.04,         # arriving from the street
    "building exterior": 0.06,
    "exterior front": 0.06,     # the facade and the front door
    "entrance": 0.12,           # inside that door
    "living room": 0.19,
    "hallway": 0.24,
    "dining room": 0.29,
    "kitchen": 0.31,
    "amenities": 0.45,
    "shared space": 0.45,
    "office": 0.56,
    "bedroom": 0.59,
    "bathroom": 0.62,
    "patio": 0.71,              # the outdoor payoff block
    "pool": 0.71,
    "backyard": 0.73,
    "exterior back": 0.75,
    "garage": 0.87,
    "storage": 0.88,
    "laundry room": 0.88,
    "basement": 0.89,
    "attic": 0.89,
}
DEFAULT_POSITION = 0.5

# All three films close outdoors or on an exterior; two end on the front
# exterior as the literal last shot. Where a property has aerials, they bookend.
# A human-labelled room instance ("bedroom-a", "exterior-front", "patio-back")
# names its own type far more reliably than the classifier does. The classifier
# runs ~82% accurate, so ordering by its guess put roughly one room in five into
# the wrong act -- a bedroom in the establishing sequence, a backyard between two
# exteriors. Longest match wins so "exterior front" beats "front".
_NAME_HINTS: list[tuple[str, str]] = [
    ("building-exterior", "building exterior"),
    ("exterior-front", "exterior front"),
    ("exterior-back", "exterior back"),
    ("exterior-side", "exterior back"),
    ("front-yard", "front yard"),
    ("shared", "shared space"),
    ("amenit", "amenities"),
    ("aerial", "aerial view"),
    ("drone", "aerial view"),
    ("backyard", "backyard"),
    ("laundry", "laundry room"),
    ("living", "living room"),
    ("dining", "dining room"),
    ("bathroom", "bathroom"),
    ("ensuite", "bathroom"),
    ("bedroom", "bedroom"),
    ("kitchen", "kitchen"),
    ("entrance", "entrance"),
    ("entry", "entrance"),
    ("hallway", "hallway"),
    ("stairs", "hallway"),
    ("garage", "garage"),
    ("storage", "storage"),
    ("basement", "basement"),
    ("attic", "attic"),
    ("office", "office"),
    ("patio", "patio"),
    ("pool", "pool"),
    ("suite", "bedroom"),
]


def room_type_from_name(room_id: str, fallback: str = "") -> str:
    """Infer a room type from a human-labelled instance name.

    Falls back to the classifier's label only when the name says nothing, which
    is the correct precedence: a person typing "bedroom-a" is ground truth.
    """
    name = (room_id or "").strip().lower()
    for token, room_type in sorted(_NAME_HINTS, key=lambda h: -len(h[0])):
        if token in name:
            return room_type
    return fallback


@dataclass
class RoomCandidate:
    """One room the film could show."""

    room_id: str
    room_type: str
    photo_ids: list[int]
    quality: float = 0.0
    # Verified pairs in this room, strongest first, as (photo_a, photo_b, score).
    # A room with several photos often yields several disjoint pairs, and each
    # one can anchor a separate shot -- two real endpoints beat one every time.
    pairs: list[tuple[int, int, float]] = field(default_factory=list)

    @property
    def best_pair(self) -> tuple[int, int, float] | None:
        return self.pairs[0] if self.pairs else None

    @property
    def tier(self) -> Tier:
        return tier_for(self.room_type)


@dataclass
class Shot:
    """One beat of the film, ready to hand to video generation."""

    index: int
    room_id: str
    room_type: str
    photo_ids: list[int]        # one photo, or two used as keyframes
    kind: str                   # "pair_keyframes" | "single_image"
    act: int
    start_seconds: float
    duration_seconds: float
    move: CameraMove
    is_hero: bool = False
    # A close-up cropped from the shot it follows. `crop_box` is in source-pixel
    # coordinates on `photo_ids[0]`; None means the whole frame is used.
    is_close_up: bool = False
    crop_box: tuple[int, int, int, int] | None = None
    crop_score: float | None = None
    source_size: tuple[int, int] | None = None

    @property
    def uses_keyframes(self) -> bool:
        return self.kind == "pair_keyframes"


@dataclass
class ShotLine:
    shots: list[Shot] = field(default_factory=list)
    bpm: float = 100.0
    hero_room_id: str | None = None
    dropped_rooms: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(s.duration_seconds for s in self.shots)

    def keyframe_fraction(self) -> float:
        """How much of the film is anchored at both ends by real photographs.

        The single most useful quality signal: a keyframe shot constrains the
        generator with two real images, a single-image shot constrains only the
        first frame.
        """
        if not self.shots:
            return 0.0
        return sum(s.uses_keyframes for s in self.shots) / len(self.shots)


def pick_hero(rooms: list[RoomCandidate]) -> RoomCandidate | None:
    """Identify the property's strongest feature for dwell allocation.

    This is not a recurring motif. It is the feature room allowed the most
    consecutive coverage when the listing has enough distinct material.
    Photographer coverage remains useful evidence, but an aerial or front
    facade belongs at the bookends even when the photographer supplied many of
    them. Prefer the strongest non-establishing selling space whenever one
    exists; fall back to an establishing feature only for an exterior-only set.
    """
    features = [r for r in rooms if r.tier is Tier.FEATURE]
    if not features:
        return None
    establishing = {"aerial view", "exterior front", "front yard",
                    "building exterior"}
    story_features = [r for r in features if r.room_type not in establishing]
    return max(story_features or features, key=lambda r: (len(r.photo_ids), r.quality))


def _weight(room: RoomCandidate, hero_id: str | None) -> float:
    if room.tier is Tier.SERVICE:
        return SERVICE_WEIGHT
    base = DWELL_WEIGHT.get(room.room_type, DEFAULT_WEIGHT)
    if room.room_id == hero_id:
        base *= 1.5        # the strongest feature earns extra coverage
    return base


def allocate_shots(rooms: list[RoomCandidate], budget: int,
                   hero_id: str | None) -> dict[str, int]:
    """Distribute the shot budget across rooms by dwell weight.

    A room can receive more than one shot -- that is the revisit behaviour the
    reference film shows (about 2.2 shots per room). A room can also receive
    zero, which is how a property with more rooms than budget drops the weakest
    rather than compressing every shot below the grid.
    """
    if not rooms or budget <= 0:
        return {}
    weights = {r.room_id: _weight(r, hero_id) for r in rooms}
    total = sum(weights.values()) or 1.0

    alloc: dict[str, int] = {}
    for room in rooms:
        alloc[room.room_id] = max(0, int(budget * weights[room.room_id] / total))

    # A room cannot show more shots than it has material for: a pair is one shot
    # and a single photo is one shot, so the cap is the photo count.
    for room in rooms:
        cap = MAX_HERO_SHOTS if room.room_id == hero_id else MAX_SHOTS_PER_ROOM
        alloc[room.room_id] = min(alloc[room.room_id],
                                  max(1, len(room.photo_ids)), cap)

    # Hand out any remainder to the strongest rooms, then trim any overshoot
    # from the weakest, so the total lands exactly on the budget.
    order = sorted(rooms, key=lambda r: (-weights[r.room_id], -r.quality))
    i = 0
    while sum(alloc.values()) < budget and i < len(order) * 4:
        room = order[i % len(order)]
        if alloc[room.room_id] < min(len(room.photo_ids), MAX_SHOTS_PER_ROOM):
            alloc[room.room_id] += 1
        i += 1
    for room in reversed(order):
        while sum(alloc.values()) > budget and alloc[room.room_id] > 0:
            alloc[room.room_id] -= 1
    return alloc


# A suite, guest annexe or secondary unit is shown after the main house has been
# shown, not interleaved with it. Cutting from the main living room to the suite
# living room and back implies they are the same space.
_SECONDARY_MARKERS = ("suite", "secondary", "guest", "annex", "basement-",
                      "lower-", "in-law", "studio-")


def is_secondary(room: RoomCandidate) -> bool:
    """Whether this room belongs to a secondary unit rather than the main home."""
    name = (room.room_id or "").strip().lower()
    return any(marker in name for marker in _SECONDARY_MARKERS)


def story_position(room: RoomCandidate) -> float:
    """Where this room belongs in the film, from the measured medians."""
    return STORY_POSITION.get(room.room_type, DEFAULT_POSITION)


def _act_of(room: RoomCandidate) -> int:
    """Acts are a reading aid over the measured positions, not a separate model.

    Act 1 is the approach up to walking in, act 2 the house, act 3 the payoff and
    close -- which the films spend outdoors.
    """
    pos = story_position(room)
    if pos <= 0.12:
        return 1
    if pos >= 0.70:
        return 3
    return 2


def _rank_in_act(room: RoomCandidate) -> int:
    return int(story_position(room) * 1000)


def build_shot_line(rooms: list[RoomCandidate], *, target_seconds: float,
                    bpm: float, allow_service: bool = False,
                    include_close_ups: bool = True) -> ShotLine:
    """Produce the ordered shot line for a listing."""
    eligible = [r for r in rooms
                if r.photo_ids and is_eligible(r.room_type, allow_service=allow_service)]
    dropped = [r.room_id for r in rooms if r not in eligible]

    # Keep only the strongest few instances of any one room type: a property
    # with four bathrooms should show the best two, not all of them equally.
    kept: list[RoomCandidate] = []
    seen: dict[str, int] = {}
    initial_hero = pick_hero(eligible)
    for room in sorted(eligible, key=lambda r: (-len(r.pairs), -r.quality,
                                                -len(r.photo_ids))):
        cap = MAX_ROOMS_PER_TYPE.get(room.room_type, DEFAULT_MAX_ROOMS_PER_TYPE)
        if room.room_id == (initial_hero.room_id if initial_hero else None):
            cap = max(cap, 1)          # the hero always survives its own cap
        if seen.get(room.room_type, 0) < cap:
            seen[room.room_type] = seen.get(room.room_type, 0) + 1
            kept.append(room)
        else:
            dropped.append(room.room_id)
    eligible = kept

    # Service rooms are kept but minimised, matching the reference film's 1.5s
    # garage. They are dropped only if the film is too short to afford them.
    # About a quarter of the finished film is close-ups, and those are generated
    # from the wide shots rather than from extra rooms. Allocating rooms against
    # the full budget would overshoot the runtime once close-ups are inserted.
    total_budget = shot_budget(target_seconds, bpm)
    budget = max(MIN_SHOTS, round(total_budget * (1.0 - CLOSE_UP_SHARE)))
    if not allow_service and budget < 12:
        service = [r for r in eligible if r.tier is Tier.SERVICE]
        eligible = [r for r in eligible if r.tier is not Tier.SERVICE]
        dropped += [r.room_id for r in service]

    hero = pick_hero(eligible)
    hero_id = hero.room_id if hero else None
    alloc = allocate_shots(eligible, budget, hero_id)

    # Only reachable with allow_service=True. The reference film gave the garage
    # 1.5s -- one short beat, not one beat per service room.
    service = sorted((r for r in eligible if r.tier is Tier.SERVICE),
                     key=lambda r: -r.quality)
    for extra in service[MAX_SERVICE_SHOTS:]:
        if alloc.get(extra.room_id):
            dropped.append(extra.room_id)
        alloc[extra.room_id] = 0
    for keep in service[:MAX_SERVICE_SHOTS]:
        alloc[keep.room_id] = min(alloc.get(keep.room_id, 0), 1)

    # Shots freed by capping service rooms belong to the feature rooms, not to
    # the floor: otherwise a house with three utility rooms produces a film
    # several seconds shorter than the one the owner asked for.
    # Cap the establishing sequence: extra exterior shots belong in the tour,
    # where outdoor space is a feature, not stacked in front of the door.
    act1 = sorted((r for r in eligible if _act_of(r) == 1 and r.room_id != hero_id),
                  key=lambda r: (_rank_in_act(r), -r.quality))
    # The entrance is the transition from outside to inside -- the film's hinge.
    # Reserve it a shot before the cap is applied, or an exterior-heavy listing
    # spends the whole establishing sequence outdoors and cuts straight to a
    # bedroom.
    entrances = [r for r in act1 if r.room_type == "entrance"]
    reserved = 0
    if entrances:
        alloc[entrances[0].room_id] = 1
        reserved = 1
    spent = 0
    for r in act1:
        if entrances and r.room_id == entrances[0].room_id:
            continue
        # One shot per approach step, never two of the same place.
        allowed = min(1, max(0, MAX_ACT1_SHOTS - reserved - spent),
                      alloc.get(r.room_id, 0))
        spent += allowed
        alloc[r.room_id] = allowed

    shortfall = budget - sum(alloc.values())
    if shortfall > 0:
        # The approach cap is editorial, not a temporary allocation limit. Do
        # not refill it with a second aerial/facade after deliberately reducing
        # that opening to one look. Spend the capacity in the property tour;
        # an exterior-only listing can still fall back to its establishing work.
        takers = [r for r in eligible
                  if r.tier is Tier.FEATURE and _act_of(r) != 1]
        if not takers:
            takers = [r for r in eligible if r.tier is Tier.FEATURE]
        takers = sorted(takers,
                        key=lambda r: (-_weight(r, hero_id), -r.quality))
        i = 0
        while shortfall > 0 and takers and i < len(takers) * 6:
            room = takers[i % len(takers)]
            cap = min(len(room.photo_ids),
                      MAX_HERO_SHOTS if room.room_id == hero_id
                      else MAX_SHOTS_PER_ROOM)
            if alloc.get(room.room_id, 0) < cap:
                alloc[room.room_id] = alloc.get(room.room_id, 0) + 1
                shortfall -= 1
            i += 1

    per_shot = shot_seconds(bpm)
    by_id = {r.room_id: r for r in eligible}

    # Lay every room out as one continuous run in measured story order.
    ordered: list[tuple[int, RoomCandidate]] = []
    for act in (1, 2, 3):
        members = sorted((r for r in eligible if _act_of(r) == act),
                         key=lambda r: (is_secondary(r), _rank_in_act(r), -r.quality))
        # Every shot of a room runs consecutively. Cutting away to a bedroom in
        # the middle of the living room and back again reads as a shuffle, not a
        # tour: you look at a room, then you move on.
        for room in members:
            for _ in range(alloc.get(room.room_id, 0)):
                ordered.append((act, room))

    # No mid-film hero motif. The measurement does not support one: aerial is
    # absent from Anthony Pl entirely and bookends the other two, while the
    # outdoor block that looked like a "returning motif" is simply where outdoor
    # space sits -- late, and large. Ordering by measured position already puts
    # it there.
    #
    # What all three films do share is the close: every one ends outdoors or on
    # an exterior, and two end on the front exterior as the literal last shot.
    if ordered:
        closing_types = ("aerial view", "exterior front", "front yard",
                         "building exterior")
        already = ordered[-1][1].room_type in closing_types
        if not already:
            closer = next((r for r in eligible
                           if r.room_type in closing_types), None)
            if closer is not None:
                ordered.append((3, closer))

    shots: list[Shot] = []
    used_pair: dict[str, set[tuple[int, int]]] = {}
    cursor = 0.0
    for i, (act, room) in enumerate(ordered):
        # Prefer verified pairs: two real photographs anchor both ends of the
        # generated clip, so the model only invents the middle. A room with
        # several photos can supply several disjoint pairs, one per appearance;
        # only once those run out do we fall back to a single image.
        taken = used_pair.setdefault(room.room_id, set())
        # Every photograph this room has already put on screen, in a pair or
        # alone. Reusing one is what produced a pair shot followed by a single
        # of the same photo -- a duplicate of a frame the viewer just watched.
        shown = {pid for shot in shots if shot.room_id == room.room_id
                 for pid in shot.photo_ids}
        pair = next((p for p in room.pairs
                     if (p[0], p[1]) not in taken and not ({p[0], p[1]} & shown)),
                    None)
        if pair is None:
            pair = next((p for p in room.pairs
                         if (p[0], p[1]) not in taken
                         and not (taken and {p[0], p[1]} & {x for t in taken for x in t})),
                        None)
        if pair is not None:
            a, b, _ = pair
            photo_ids = [a, b]
            kind = "pair_keyframes"
            taken.add((a, b))
        else:
            unseen = [pid for pid in room.photo_ids if pid not in shown]
            is_closing = i == len(ordered) - 1
            if not unseen and not is_closing:
                # Nothing new left to show: skip rather than repeat a frame.
                continue
            # The closing shot is a deliberate return to the property's face, so
            # it may reuse a frame where an ordinary repeat would not.
            photo_ids = [unseen[0] if unseen else room.photo_ids[0]]
            kind = "single_image"
        establishing = room.room_type in ("aerial view", "exterior front",
                                          "front yard", "building exterior")
        move = choose_move(f"{room.room_id}:{i}", is_establishing=establishing)
        shots.append(Shot(
            index=i, room_id=room.room_id, room_type=room.room_type,
            photo_ids=photo_ids, kind=kind, act=act,
            start_seconds=round(cursor, 3), duration_seconds=round(per_shot, 3),
            move=move, is_hero=(room.room_id == hero_id),
        ))
        cursor += per_shot

    if include_close_ups:
        shots = _insert_close_ups(shots, per_shot)
    return ShotLine(shots=shots, bpm=bpm, hero_room_id=hero_id, dropped_rooms=dropped)


def _unit_for(seed: str) -> float:
    """Deterministic value in [0,1) so a listing always produces the same film."""
    import hashlib

    digest = hashlib.sha256(seed.encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def schedule_resolved_close_ups(
    shots: list[Shot],
    resolved: dict[int, ResolvedDetail],
) -> list[Shot]:
    """Insert only crop-validated close-ups into a selected wide line."""
    if not shots:
        return []
    return _insert_close_ups(shots, shots[0].duration_seconds, resolved)


def _insert_close_ups(
    shots: list[Shot],
    per_shot: float,
    resolved: dict[int, ResolvedDetail] | None = None,
) -> list[Shot]:
    """Add crop-validated close-ups on an approximate four-shot cadence.

    The close-up is a crop of the frame immediately before it, which is what
    makes the pair read as one continuous look rather than two pictures. In the
    API path `resolved` contains crop-validated candidates and every emitted
    detail therefore has a concrete crop box. The no-argument path is retained
    only for inexpensive planner-only tests and CLI previews.

    The opening approach and the closing run stay wide: the reference films thin
    their close-ups out at both ends.
    """
    if len(shots) < 4:
        return shots

    head = 2                      # leave the approach alone
    tail = max(1, len(shots) // 8)  # and the close

    # A quarter of the finished cut means C / (W + C) = 0.25, or C = W / 3.
    # This is a ceiling, never a quota that forces generic material on screen.
    # The API passes `resolved` after testing every selected wide shot. The
    # no-argument path supports dry-run scripts and unit tests only.
    eligible = [
        (idx, shot)
        for idx, shot in enumerate(shots)
        if head <= idx < len(shots) - tail
        and close_up_rate(shot.room_type) > 0
        and (resolved is None or shot.index in resolved)
    ]
    wanted = min(round(len(shots) * CLOSE_UP_SHARE / (1.0 - CLOSE_UP_SHARE)),
                 len(eligible))
    if wanted == 0:
        return shots

    # Work in four-wide-shot windows. This produces the intended rhythm without
    # allowing a high-scoring patio or kitchen to create a burst of consecutive
    # details. One detail per room is enough for a first cut; the full room wide
    # shots already carry the coverage.
    chosen: dict[int, ResolvedDetail | None] = {}
    used_rooms: set[str] = set()
    last_index = -4
    body_end = len(shots) - tail
    for window_start in range(head, body_end, 4):
        options = [
            (idx, shot)
            for idx, shot in eligible
            if window_start <= idx < min(window_start + 4, body_end)
            and idx - last_index >= 4
            and shot.room_id not in used_rooms
        ]
        if not options:
            continue
        # A validated crop's score decides the visual choice. Dry runs do not
        # have that score, so use deterministic room-type weighting instead.
        index, shot = max(
            options,
            key=lambda candidate: (
                resolved[candidate[1].index].crop.score
                if resolved is not None else close_up_rate(candidate[1].room_type),
                close_up_rate(candidate[1].room_type),
                _unit_for(f"{candidate[1].room_id}:{candidate[0]}:closeup"),
            ),
        )
        chosen[index] = resolved[shot.index] if resolved is not None else None
        used_rooms.add(shot.room_id)
        last_index = index
        if len(chosen) >= wanted:
            break

    out: list[Shot] = []
    for i, shot in enumerate(shots):
        out.append(shot)
        if not (head <= i < len(shots) - tail) or shot.is_close_up:
            continue
        detail = chosen.get(i)
        if i not in chosen:
            continue
        out.append(Shot(
            index=-1,
            room_id=shot.room_id,
            room_type=shot.room_type,
            photo_ids=[shot.photo_ids[0]],
            kind="single_image",
            act=shot.act,
            start_seconds=0.0,
            duration_seconds=per_shot,
            move=choose_move(f"{shot.room_id}:closeup:{i}"),
            is_hero=shot.is_hero,
            is_close_up=True,
            crop_box=detail.crop.box if detail else None,
            crop_score=detail.crop.score if detail else None,
            source_size=detail.source_size if detail else None,
        ))

    cursor = 0.0
    renumbered: list[Shot] = []
    for index, shot in enumerate(out):
        renumbered.append(replace(shot, index=index,
                                  start_seconds=round(cursor, 3)))
        cursor += shot.duration_seconds
    return renumbered
