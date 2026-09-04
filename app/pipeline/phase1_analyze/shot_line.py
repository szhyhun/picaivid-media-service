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

Known gap: the films contain **detail shots** -- a garden close-up opening the
film, details within the living room, flowers in the backyard. Those are a shot
type this module has no concept of, because nothing upstream identifies a photo
as a detail rather than a room view. See VIDEO_PRODUCTION_PLAN.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.pipeline.phase1_analyze.camera_move import CameraMove, choose_move
from app.pipeline.phase1_analyze.editorial import (
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
    "aerial view": 2.1,        # the hero motif in the reference film
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

# The hero is revisited, not dwelt on: Wallace Dr showed its view four times
# across 35 shots. More than that and the film stops being a property tour.
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
    "aerial view": 0.02,        # opens the film where present; see AERIAL_CLOSES
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
AERIAL_CLOSES = True

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
    """Identify the property's selling point.

    In the reference film the ocean view is revisited three times and takes 21%
    of runtime. Which room deserves that is property-specific, so it is inferred
    rather than hard-coded: the feature room the photographer shot most, broken
    by quality. A photographer taking six photos of one view is telling us
    something.
    """
    features = [r for r in rooms if r.tier is Tier.FEATURE]
    if not features:
        return None
    return max(features, key=lambda r: (len(r.photo_ids), r.quality))


def _weight(room: RoomCandidate, hero_id: str | None) -> float:
    if room.tier is Tier.SERVICE:
        return SERVICE_WEIGHT
    base = DWELL_WEIGHT.get(room.room_type, DEFAULT_WEIGHT)
    if room.room_id == hero_id:
        base *= 1.5        # the hero is revisited, so it earns extra shots
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
        alloc[room.room_id] = min(alloc[room.room_id],
                                  max(1, len(room.photo_ids)),
                                  MAX_SHOTS_PER_ROOM)

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
                    bpm: float, allow_service: bool = False) -> ShotLine:
    """Produce the ordered shot line for a listing."""
    eligible = [r for r in rooms
                if r.photo_ids and is_eligible(r.room_type, allow_service=allow_service)]
    dropped = [r.room_id for r in rooms if r not in eligible]

    # Keep only the strongest few instances of any one room type: a property
    # with four bathrooms should show the best two, not all of them equally.
    kept: list[RoomCandidate] = []
    seen: dict[str, int] = {}
    for room in sorted(eligible, key=lambda r: (-len(r.pairs), -r.quality,
                                                -len(r.photo_ids))):
        cap = MAX_ROOMS_PER_TYPE.get(room.room_type, DEFAULT_MAX_ROOMS_PER_TYPE)
        if room.room_id == (pick_hero(eligible).room_id if eligible else None):
            cap = max(cap, 1)          # the hero always survives its own cap
        if seen.get(room.room_type, 0) < cap:
            seen[room.room_type] = seen.get(room.room_type, 0) + 1
            kept.append(room)
        else:
            dropped.append(room.room_id)
    eligible = kept

    # Service rooms are kept but minimised, matching the reference film's 1.5s
    # garage. They are dropped only if the film is too short to afford them.
    budget = shot_budget(target_seconds, bpm)
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
        takers = sorted((r for r in eligible if r.tier is Tier.FEATURE),
                        key=lambda r: (-_weight(r, hero_id), -r.quality))
        i = 0
        while shortfall > 0 and takers and i < len(takers) * 6:
            room = takers[i % len(takers)]
            cap = (MAX_HERO_SHOTS if room.room_id == hero_id
                   else min(len(room.photo_ids), MAX_SHOTS_PER_ROOM))
            if alloc.get(room.room_id, 0) < cap:
                alloc[room.room_id] = alloc.get(room.room_id, 0) + 1
                shortfall -= 1
            i += 1

    per_shot = shot_seconds(bpm)
    by_id = {r.room_id: r for r in eligible}

    # Lay rooms out in act order, then interleave repeats so a revisited room
    # comes back later in the film rather than showing twice in a row.
    ordered: list[tuple[int, RoomCandidate]] = []
    for act in (1, 2, 3):
        members = sorted((r for r in eligible if _act_of(r) == act),
                         key=lambda r: (_rank_in_act(r), -r.quality))
        # The approach is walked once. Repeating a front yard after the entrance
        # would step back outside halfway through going in.
        passes = 1 if act == 1 else max(
            (alloc.get(r.room_id, 0) for r in members), default=0)
        for p in range(passes):
            wave = [r for r in members if alloc.get(r.room_id, 0) > p]
            # The reference film alternates -- bedroom, bath, bedroom -- rather
            # than showing every bathroom back to back. Spread same-type rooms
            # so no two neighbours share a type where the material allows it.
            spread: list[RoomCandidate] = []
            pending = list(wave)
            while pending:
                pick = next((r for r in pending
                             if not spread or r.room_type != spread[-1].room_type),
                            pending[0])
                pending.remove(pick)
                spread.append(pick)
            ordered.extend((act, room) for room in spread)

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
        pair = next((p for p in room.pairs
                     if (p[0], p[1]) not in taken
                     and not (taken and {p[0], p[1]} & {x for t in taken for x in t})),
                    None)
        if pair is None:
            pair = next((p for p in room.pairs if (p[0], p[1]) not in taken), None)
        if pair is not None:
            a, b, _ = pair
            photo_ids = [a, b]
            kind = "pair_keyframes"
            taken.add((a, b))
        else:
            offset = sum(1 for s in shots if s.room_id == room.room_id)
            photo_ids = [room.photo_ids[offset % len(room.photo_ids)]]
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

    return ShotLine(shots=shots, bpm=bpm, hero_room_id=hero_id, dropped_rooms=dropped)
