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

Close-ups are discovered from named objects across every photo in a selected
room after this line is built. They remain tied to the room immediately before
them, so the detail reads as a continuation rather than an unrelated image.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

from app.pipeline.phase1_analyze.camera_move import (
    STRAIGHT_SINGLE_ROOM_TYPES,
    CameraMove,
    choose_move,
)
from app.pipeline.phase1_analyze.detail_crop import ResolvedDetail, is_detail_eligible
from app.pipeline.phase1_analyze.editorial import (
    MIN_SHOTS,
    Tier,
    SOURCE_GENERATION_SECONDS,
    is_eligible,
    shot_budget,
    shot_seconds,
    tier_for,
)
from app.pipeline.phase1_analyze.transitions import is_duplicate as is_vggt_duplicate

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
    # Bedrooms earn coverage before bathrooms. A finished tour should show the
    # best private rooms before it spends beats on additional bath coverage.
    "bedroom": 1.3,
    "bathroom": 1.1,
    # Dining is a core social selling space. Its smaller reference-film share
    # does not justify allowing bathrooms to erase it from the property tour.
    "dining room": 1.4,
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
# The reference films use a directional establishing run rather than a single
# obligatory exterior: aerial/garden when it earns its place, two or more
# approach views, then the threshold. Five lets a well-covered property breathe
# without turning the opening into an exterior tour. Sparse listings naturally
# use fewer because this is a candidate cap, never a quota.
MAX_ACT1_SHOTS = 5
MAX_ACT1_AERIAL_SHOTS = 2
MAX_ACT1_APPROACH_SHOTS = 3
# The reference film averaged 2.2 shots per ordinary room. Social rooms are the
# deliberate exception: they are the property's strongest shared space, and a
# well-photographed one can support several distinct looks without repeating a
# frame.
MAX_SHOTS_PER_ROOM = 2
MAX_SOCIAL_SHOTS_PER_ROOM = 4
# The social block should feel substantial without becoming the whole film.
# This is an aggregate ceiling across the main house and any later social unit.
MAX_SOCIAL_SHOTS = 12
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

# Bathrooms are optional supporting coverage. Unlike a strong single image of a
# living room or bedroom, a lone bathroom frame rarely advances the property
# story. Show one only when VGGT verified a real two-photo camera move; never
# use a bathroom singleton merely to consume budget.
PAIR_ONLY_ROOM_TYPES = frozenset({"bathroom"})

# A later kitchen or living room is editorially a separate unit only after a
# meaningful gap in the uploaded capture sequence. This keeps a main social
# room split across adjacent VGGT components together while moving an in-law or
# suite social room behind the main home. It is a story heuristic, never
# geometry or membership evidence.
SECONDARY_SOCIAL_GAP = 8
SECONDARY_UNIT_POSITION = 0.68
SOCIAL_ROOM_TYPES = ("living room", "dining room", "kitchen")
# Kitchen and living-room capture runs can identify a distinct suite; dining
# runs are commonly split by an open-plan component gap and should stay in the
# main social block rather than being mistaken for another unit.
PRIMARY_SOCIAL_ROOM_TYPES = ("kitchen", "living room")

# Selection and story order are separate decisions. These are the main-house
# social spaces that earn one protected beat when present; additional coverage
# still comes from the ordinary evidence and dwell-weight allocation below.
CORE_SOCIAL_TYPES = SOCIAL_ROOM_TYPES

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
    "front yard": 0.90,         # closing return; opening selector can promote it
    "building exterior": 0.06,
    "exterior front": 0.06,     # the facade and the front door
    "entrance": 0.12,           # inside that door
    "living room": 0.19,
    "hallway": 0.24,
    "dining room": 0.29,
    "kitchen": 0.31,
    "amenities": 0.69,
    "shared space": 0.69,
    "building lobby": 0.69,
    "elevator lobby": 0.69,
    "gym": 0.64,
    "walk-in closet": 0.88,
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
    # Explicit outdoor destinations place connecting stairs with the outdoor
    # sequence; the generic "stairs" hint would otherwise make them a hallway.
    if "stairs-to-" in name:
        for destination, room_type in (("patio", "patio"), ("deck", "patio"),
                                       ("backyard", "backyard"), ("pool", "pool")):
            if destination in name.split("stairs-to-", 1)[1]:
                return room_type
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
    # Geometry components may span open-plan labels. These fields preserve the
    # semantic zone and capture order without pretending they are new geometry.
    photo_positions: dict[int, int] = field(default_factory=dict)
    parent_component_id: int | None = None
    # Component-level broad semantic class persisted alongside VGGT geometry.
    # It does not claim which side of the house a photo shows and cannot turn a
    # scene classifier's "backyard" label into evidence of a front facade.
    scene_type: str | None = None
    secondary: bool = False

    @property
    def best_pair(self) -> tuple[int, int, float] | None:
        return self.pairs[0] if self.pairs else None

    @property
    def tier(self) -> Tier:
        return tier_for(self.room_type)


@dataclass
class Shot:
    """One two-beat final edit, with its fixed five-second source request."""

    index: int
    room_id: str
    room_type: str
    photo_ids: list[int]        # one photo, or two used as keyframes
    kind: str                   # "pair_keyframes" | "single_image"
    act: int
    start_seconds: float
    duration_seconds: float
    move: CameraMove
    # This is source footage requested from the video provider, not the final
    # edit hold. The renderer trims it to `duration_seconds` on the music grid.
    generation_duration_seconds: float = SOURCE_GENERATION_SECONDS
    is_hero: bool = False
    # A close-up cropped from the shot it follows. `crop_box` is in source-pixel
    # coordinates on `photo_ids[0]`; None means the whole frame is used.
    is_close_up: bool = False
    crop_box: tuple[int, int, int, int] | None = None
    crop_score: float | None = None
    source_size: tuple[int, int] | None = None
    detail_subject: str | None = None
    detector_score: float | None = None
    subject_box: tuple[int, int, int, int] | None = None
    detail_framing: str | None = None

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


def _disjoint_pair_capacity(room: RoomCandidate) -> int:
    """Count the distinct verified pairs a pair-only room can actually show."""
    used: set[int] = set()
    count = 0
    for left, right, score in room.pairs:
        if score <= 0.0 or left in used or right in used:
            continue
        used.update((left, right))
        count += 1
    return count


def _room_shot_capacity(room: RoomCandidate, hero_id: str | None) -> int:
    """Maximum distinct cinematic beats this room can supply."""
    cap = MAX_HERO_SHOTS if room.room_id == hero_id else MAX_SHOTS_PER_ROOM
    if room.room_type in SOCIAL_ROOM_TYPES and not is_secondary(room):
        cap = max(cap, MAX_SOCIAL_SHOTS_PER_ROOM)
    pairs = _disjoint_pair_capacity(room)
    if room.room_type in PAIR_ONLY_ROOM_TYPES:
        return min(cap, pairs)
    # Each selected pair consumes two source photos but produces one beat.
    singles = max(0, len(room.photo_ids) - 2 * pairs)
    return min(cap, pairs + singles)


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

    # A room cannot show more shots than it has material for. Bathrooms are the
    # deliberate exception: they earn only verified, disjoint pair beats.
    for room in rooms:
        alloc[room.room_id] = min(alloc[room.room_id],
                                  _room_shot_capacity(room, hero_id))

    # Hand out any remainder to the strongest rooms, then trim any overshoot
    # from the weakest, so the total lands exactly on the budget.
    order = sorted(rooms, key=lambda r: (-weights[r.room_id], -r.quality))
    i = 0
    while sum(alloc.values()) < budget and i < len(order) * 4:
        room = order[i % len(order)]
        if alloc[room.room_id] < _room_shot_capacity(room, hero_id):
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
    if room.secondary:
        return True
    name = (room.room_id or "").strip().lower()
    return any(marker in name for marker in _SECONDARY_MARKERS)


def space_scope(room_type: str) -> str:
    """Describe predicted communal use; room function alone cannot prove ownership."""
    if room_type in {"shared space", "building lobby", "elevator lobby"}:
        return "shared"
    return "unknown"


def story_position(room: RoomCandidate) -> float:
    """Where this room belongs in the film, from the measured medians."""
    if space_scope(room.room_type) == "shared":
        return 0.69
    if room.secondary:
        return SECONDARY_UNIT_POSITION
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


def _tour_order_key(room: RoomCandidate) -> tuple[int, int, int, float, str]:
    """Keep the main social rooms together in their deliberate tour order."""
    if not is_secondary(room) and room.room_type in SOCIAL_ROOM_TYPES:
        return (
            0,
            SOCIAL_ROOM_TYPES.index(room.room_type),
            _room_start(room),
            -room.quality,
            room.room_id,
        )
    if is_secondary(room):
        # A secondary unit remains one late block, after the main house.
        return (2, _room_start(room), _room_start(room), -room.quality, room.room_id)
    return (1, _rank_in_act(room), _room_start(room), -room.quality, room.room_id)


def _room_start(room: RoomCandidate) -> int:
    """Earliest capture position, with photo id as deterministic fixture fallback."""
    positions = [room.photo_positions.get(photo_id, photo_id) for photo_id in room.photo_ids]
    return min(positions) if positions else 0


def _mark_secondary_social_rooms(rooms: list[RoomCandidate]) -> list[RoomCandidate]:
    """Move later kitchen and living-room capture runs behind the main home.

    The first kitchen and first living room in capture order are the main-home
    social spaces. Adjacent components can still be one room, but a later run is
    a suite/secondary unit for editorial ordering. Compare candidate starts to
    the first run's start, not its final photo: a primary room may contain a
    late return photo, which must not hide an earlier secondary social room.
    """
    copies = [replace(room, secondary=is_secondary(room)) for room in rooms]
    for room_type in PRIMARY_SOCIAL_ROOM_TYPES:
        candidates = sorted(
            (room for room in copies
             if room.room_type == room_type and room.photo_ids and not room.secondary),
            key=lambda room: (_room_start(room), room.room_id),
        )
        if not candidates:
            continue
        primary_start = _room_start(candidates[0])
        for room in candidates[1:]:
            if _room_start(room) > primary_start + SECONDARY_SOCIAL_GAP:
                room.secondary = True

    # Semantic zones are editorial views of one VGGT component. A late social
    # room identifies a secondary unit, not merely an isolated room: its sibling
    # living/hall/bedroom zones must follow that unit rather than jumping back
    # into the early main-house tour.
    secondary_components = {
        room.parent_component_id
        for room in copies
        if room.secondary and room.parent_component_id is not None
    }
    for room in copies:
        if room.parent_component_id in secondary_components:
            room.secondary = True
    return copies


def _core_social_coverage(rooms: list[RoomCandidate]) -> list[RoomCandidate]:
    """Return the strongest main-house living, dining and kitchen candidates."""
    selected: list[RoomCandidate] = []
    for room_type in CORE_SOCIAL_TYPES:
        candidate = max(
            (room for room in rooms
             if room.room_type == room_type and not is_secondary(room)),
            # Photographer coverage is the strongest same-type importance
            # signal. Pair evidence then decides the better-supported room
            # when two candidates received comparable coverage.
            key=lambda room: (
                len(room.photo_ids),
                bool(room.pairs),
                len(room.pairs),
                room.quality,
                -_room_start(room),
                room.room_id,
            ),
            default=None,
        )
        if candidate is not None:
            selected.append(candidate)
    return selected


_FACADE_APPROACH_TYPES = ("building exterior", "exterior front")
_TOUR_INTERIOR_TYPES = ("entrance", "hallway", "living room", "dining room", "kitchen")


def is_persisted_vggt_duplicate(metrics: dict | None) -> bool:
    """Apply the current duplicate rule to pre-rule-change relation metadata.

    Old completed jobs retain their original JSON decision. Re-reading the
    measured depth/pose fields is cheap and prevents a known duplicate from
    reaching a new shot line just because the job predates the current gate.
    """
    metrics = metrics or {}
    if bool(metrics.get("is_duplicate")):
        return True
    if metrics.get("verification_source") != "vggt_pairwise_v2":
        return False
    def number(key: str, default: float) -> float:
        try:
            return float(metrics.get(key, default))
        except (TypeError, ValueError):
            return default
    return is_vggt_duplicate({
        "depth_ok_min": min(
            number("depth_ok_forward", 0.0),
            number("depth_ok_backward", 0.0),
        ),
        "bl_over_depth": number("bl_over_depth", float("inf")),
        "rot_deg": number("rotation_degrees", float("inf")),
    })


def _opening_source_key(room: RoomCandidate) -> tuple[str, int | str]:
    """Group semantic zones that came from one VGGT component."""
    if room.parent_component_id is not None:
        return ("component", room.parent_component_id)
    return ("room", room.room_id)


def _has_second_opening_view(room: RoomCandidate) -> bool:
    """Whether this room can supply a second non-repeated establishing beat."""
    return len(room.pairs) >= 2 or len(room.photo_ids) >= 4


def _opening_slots(candidates: list[RoomCandidate], limit: int,
                   *, diversify_types: bool = False) -> list[RoomCandidate]:
    """Pick distinct components first, then a second verified view where earned."""
    selected: list[RoomCandidate] = []
    seen_sources: set[tuple[str, int | str]] = set()
    seen_types: set[str] = set()
    for room in candidates:
        source = _opening_source_key(room)
        if source in seen_sources or (diversify_types and room.room_type in seen_types):
            continue
        selected.append(room)
        seen_sources.add(source)
        seen_types.add(room.room_type)
        if len(selected) == limit:
            return selected

    # A well photographed aerial or facade may live in one component. Its
    # second disjoint pair is a new view; replaying the same photo is not.
    for room in candidates:
        if len(selected) == limit:
            break
        if room in selected and _has_second_opening_view(room):
            selected.append(room)
    return selected


def _first_tour_interior_start(rooms: list[RoomCandidate]) -> int | None:
    """Find the first plausible main-house interior in the upload sequence."""
    starts = [
        _room_start(room)
        for room in rooms
        if room.room_type in _TOUR_INTERIOR_TYPES and not is_secondary(room)
    ]
    return min(starts) if starts else None


def _opening_sequence(rooms: list[RoomCandidate]) -> list[RoomCandidate]:
    """Choose an adaptive approach: up to two aerials -> approach -> entry.

    Semantic labels identify explicit front material. A component-level exterior
    label plus capture position before the first main-house interior can make an
    uncertain `backyard` label eligible as a *fallback* approach candidate. It
    remains lower priority than recognised front material, and patios/pools/back
    exteriors never receive that promotion.
    """
    aerial_candidates = sorted(
        (room for room in rooms
         if room.room_type == "aerial view" or room.scene_type == "drone"),
        key=lambda room: (-room.quality, -len(room.pairs), -len(room.photo_ids), _room_start(room), room.room_id),
    )
    aerials = _opening_slots(aerial_candidates, MAX_ACT1_AERIAL_SHOTS)

    entry = max(
        (room for room in rooms if room.room_type == "entrance"),
        key=lambda room: (room.quality, len(room.photo_ids), -_room_start(room)),
        default=None,
    )
    if entry is None:
        # A corridor substitutes for an entry only when the listing lacks a
        # labelled threshold. The first capture is a better approximation of
        # the real approach than a high-quality hallway from a later suite.
        entry = min(
            (room for room in rooms if room.room_type == "hallway" and not is_secondary(room)),
            key=lambda room: (_room_start(room), -room.quality, room.room_id),
            default=None,
        )

    first_interior = _first_tour_interior_start(rooms)
    facades = [room for room in rooms if room.room_type in _FACADE_APPROACH_TYPES]
    # A real facade/door candidate is materially stronger than a room classifier
    # calling a side of the house "front yard". If one exists, use only facade
    # views in the automatic approach. A front-yard detail can still be an
    # explicit editor choice later, but it is not safe to invent that role.
    approach: list[RoomCandidate] = facades or [
        room for room in rooms
        if room.room_type == "front yard" or (
            room.room_type == "backyard" and room.scene_type == "exterior"
            and (first_interior is None or _room_start(room) < first_interior)
        )
    ]
    approach_priority = {
        room_type: index
        for index, room_type in enumerate(
            _FACADE_APPROACH_TYPES if facades else ("front yard", "backyard")
        )
    }
    approach.sort(key=lambda room: (
        approach_priority.get(room.room_type, len(approach_priority)),
        _room_start(room),
        -room.quality,
        room.room_id,
    ))
    approach_limit = min(
        MAX_ACT1_APPROACH_SHOTS,
        max(0, MAX_ACT1_SHOTS - len(aerials) - int(entry is not None)),
    )
    approach = _opening_slots(
        approach,
        approach_limit,
        diversify_types=not bool(facades),
    )
    return [*aerials, *approach, *([entry] if entry is not None else [])]


def _closing_room(rooms: list[RoomCandidate]) -> RoomCandidate | None:
    """Return the most legible exterior bookend, preferring a true facade."""
    priorities = ("exterior front", "building exterior", "aerial view", "front yard")
    for room_type in priorities:
        candidate = max(
            (room for room in rooms if room.room_type == room_type),
            key=lambda room: (room.quality, len(room.photo_ids), -_room_start(room)),
            default=None,
        )
        if candidate is not None:
            return candidate
    return None


def build_shot_line(rooms: list[RoomCandidate], *, target_seconds: float,
                    bpm: float, allow_service: bool = False,
                    include_close_ups: bool = True) -> ShotLine:
    """Produce the ordered shot line for a listing."""
    rooms = _mark_secondary_social_rooms(rooms)
    eligible = [
        room for room in rooms
        if room.photo_ids
        and is_eligible(room.room_type, allow_service=allow_service)
        and (
            room.room_type not in PAIR_ONLY_ROOM_TYPES
            or _disjoint_pair_capacity(room) > 0
        )
    ]
    dropped = [r.room_id for r in rooms if r not in eligible]
    # Select the protected approach before generic semantic caps. The caps are
    # for the body of the tour; applying them first can discard a real front
    # exterior merely because the classifier called several photos "backyard".
    opening = _opening_sequence(eligible)
    opening_ids = {room.room_id for room in opening}
    opening_counts = Counter(room.room_id for room in opening)

    # Keep only the strongest few instances of any one room type: a property
    # with four bathrooms should show the best two, not all of them equally.
    kept: list[RoomCandidate] = []
    seen: dict[str, int] = {}
    initial_hero = pick_hero(eligible)
    for room in sorted(eligible, key=lambda r: (-len(r.photo_ids), -len(r.pairs),
                                                -r.quality, r.room_id)):
        cap = MAX_ROOMS_PER_TYPE.get(room.room_type, DEFAULT_MAX_ROOMS_PER_TYPE)
        if room.room_type in PRIMARY_SOCIAL_ROOM_TYPES and not is_secondary(room):
            # Preserve split main-room components until aggregate shot budgeting.
            kept.append(room)
            continue
        if room.room_id == (initial_hero.room_id if initial_hero else None):
            cap = max(cap, 1)          # the hero always survives its own cap
        if room.room_id in opening_ids:
            # A deliberate opening candidate outranks the generic per-type
            # cap. It is still bounded by MAX_ACT1_SHOTS above.
            seen[room.room_type] = seen.get(room.room_type, 0) + 1
            kept.append(room)
        elif seen.get(room.room_type, 0) < cap:
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
    # Cap and protect the approach. Generic allocation once allowed two aerial
    # components and a side-yard singleton to consume all four slots before a
    # front facade could appear. The approach has a fixed visual grammar, so
    # its candidates are selected by role rather than by generic room weight.
    for room in eligible:
        if room.room_id in opening_counts:
            # One component may earn two opening beats when it has two disjoint
            # verified views. A fallback exterior promoted into Act 1 must not
            # also reappear later under its noisy "backyard" label.
            alloc[room.room_id] = opening_counts[room.room_id]
        elif _act_of(room) == 1:
            alloc[room.room_id] = 0

    # A property film must establish its principal social spaces before it
    # spends coverage on optional bathrooms or a later secondary unit. Reserve
    # the social block explicitly instead of letting room weights scatter it.
    # A very short cut can reserve only what fits after the approach and close.
    close_reserve = int(_closing_room(eligible) is not None)
    core_capacity = max(0, budget - sum(opening_counts.values()) - close_reserve)
    social_pools = {
        kind: sorted(
            (room for room in eligible if room.room_type == kind),
            key=lambda room: (
                is_secondary(room),
                -len(room.pairs),
                -len(room.photo_ids),
                -room.quality,
                _room_start(room),
                room.room_id,
            ),
        )
        for kind in SOCIAL_ROOM_TYPES
    }
    primary_social_pools = {
        kind: [room for room in pool if not is_secondary(room)]
        for kind, pool in social_pools.items()
    }
    secondary_social_pools = {
        kind: [room for room in pool if is_secondary(room)]
        for kind, pool in social_pools.items()
    }
    social_rooms = [room for pool in social_pools.values() for room in pool]
    primary_social_rooms = [
        room for pool in primary_social_pools.values() for room in pool
    ]

    # Rebuild social allocation from zero so the aggregate quota is deliberate.
    # This also makes a seventh kitchen frame compete with other social rooms
    # rather than winning only because its component has the highest weight.
    for room in social_rooms:
        alloc[room.room_id] = 0

    social_capacity = sum(
        _room_shot_capacity(room, hero_id) for room in social_rooms
    )
    social_target = min(MAX_SOCIAL_SHOTS, core_capacity, social_capacity)

    def fill_social(pools: dict[str, list[RoomCandidate]], limit: int) -> int:
        """Fill one social look per room type per pass until the limit is met."""
        added = 0
        while added < limit:
            progressed = False
            for room_type in SOCIAL_ROOM_TYPES:
                for room in pools[room_type]:
                    if alloc.get(room.room_id, 0) >= _room_shot_capacity(room, hero_id):
                        continue
                    alloc[room.room_id] = alloc.get(room.room_id, 0) + 1
                    added += 1
                    progressed = True
                    break
                if added >= limit:
                    break
            if not progressed:
                break
        return added

    social_total = fill_social(primary_social_pools, social_target)
    social_total += fill_social(
        secondary_social_pools,
        social_target - social_total,
    )
    # Protect only the main-house social run from later budget trimming. A
    # secondary-unit look can still give way when an otherwise empty property
    # needs a closing exterior or another primary room.
    core_social_ids = {
        room.room_id
        for room in primary_social_rooms
        if alloc.get(room.room_id, 0) > 0
    }

    # Reserving an absent facade or entry can add shots that generic weighted
    # allocation had already dropped in a short edit. Make room in the body now
    # and keep one beat available for the deliberate exterior closing return.
    if sum(alloc.values()) > budget:
        body_budget = max(0, budget - (1 if _closing_room(eligible) else 0))
        removable = sorted(
            (room for room in eligible
             if room.room_id not in opening_ids | core_social_ids),
            key=lambda room: (_weight(room, hero_id), room.quality, len(room.photo_ids), room.room_id),
        )
        for room in removable:
            while sum(alloc.values()) > body_budget and alloc.get(room.room_id, 0) > 0:
                alloc[room.room_id] -= 1

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
            cap = _room_shot_capacity(room, hero_id)
            if room.room_type in SOCIAL_ROOM_TYPES and social_total >= social_target:
                i += 1
                continue
            if alloc.get(room.room_id, 0) < cap:
                alloc[room.room_id] = alloc.get(room.room_id, 0) + 1
                shortfall -= 1
                if room.room_type in SOCIAL_ROOM_TYPES:
                    social_total += 1
            i += 1

    per_shot = shot_seconds(bpm)
    by_id = {r.room_id: r for r in eligible}

    # Lay every room out as one continuous run in measured story order. Act 1 is
    # deliberately protected above; it is not an arbitrary subset of outdoor
    # components sorted by quality.
    ordered: list[tuple[int, RoomCandidate]] = []
    for room in opening:
        if alloc.get(room.room_id, 0):
            ordered.append((1, room))
    for act in (2, 3):
        members = sorted(
            (r for r in eligible
             if _act_of(r) == act and r.room_id not in opening_ids),
            key=_tour_order_key,
        )
        # The main social block is intentional: enter, see the living room, move
        # through dining, then land in the kitchen before private rooms. Every
        # shot of one room still runs consecutively; cutting away and back reads
        # as a shuffle, not a tour.
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
        closing_types = ("aerial view", "exterior front", "front yard", "building exterior")
        already = ordered[-1][1].room_type in closing_types
        if not already:
            closer = _closing_room(eligible)
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
            if room.room_type in PAIR_ONLY_ROOM_TYPES:
                # Allocation is pair-capped, but preserve the editorial rule
                # if pair selection changes in the future.
                continue
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
        # Threshold shots are straight reveals. When a room has verified pairs,
        # those pairs provide the visual path, so fallback singles do not invent
        # a random vertical move. Other rooms retain the subtle measured drift
        # distribution from the reference films.
        straight_room = room.room_type in STRAIGHT_SINGLE_ROOM_TYPES
        move = choose_move(
            f"{room.room_id}:{i}",
            is_establishing=establishing,
            allow_pan=not straight_room,
            allow_tilt=not straight_room and not room.pairs,
        )
        shots.append(Shot(
            index=i, room_id=room.room_id, room_type=room.room_type,
            photo_ids=photo_ids, kind=kind, act=act,
            start_seconds=round(cursor, 3), duration_seconds=round(per_shot, 3),
            move=move, is_hero=(room.room_id == hero_id),
        ))
        cursor += per_shot

    # A planned room can exhaust its distinct images and be skipped above. Make
    # the response a true contiguous timeline rather than exposing planner-loop
    # indices with holes to the editor.
    shots = [replace(shot, index=index) for index, shot in enumerate(shots)]

    # Object-aware details are saved during Phase 1 and selected by the API after
    # the wide line is built. Planner-only invocation never makes a generic crop
    # simply to fill this optional slot.
    return ShotLine(shots=shots, bpm=bpm, hero_room_id=hero_id, dropped_rooms=dropped)


def schedule_resolved_close_ups(
    shots: list[Shot],
    resolved: dict[str, list[ResolvedDetail]],
) -> list[Shot]:
    """Insert detected, complete object details into their room's wide run."""
    return _insert_close_ups(shots, resolved)


def _insert_close_ups(
    shots: list[Shot],
    resolved: dict[str, list[ResolvedDetail]],
) -> list[Shot]:
    """Add complete object details on the measured approximate four-shot cadence."""
    if len(shots) < 4:
        return shots

    head = 2                      # leave the approach alone
    # Only the literal final bookend is protected. A late patio/outdoor payoff
    # may contain the listing's best detected object and must remain available
    # for a detail before the final exterior or aerial frame.
    tail = 1

    # A quarter of the finished cut means C / (W + C) = 0.25, or C = W / 3.
    # This is a ceiling, never a quota that forces a weak object on screen.
    eligible = [
        (idx, shot)
        for idx, shot in enumerate(shots)
        if head <= idx < len(shots) - tail
        and is_detail_eligible(shot.room_type)
        and resolved.get(shot.room_id)
    ]
    wanted = min(round(len(shots) * CLOSE_UP_SHARE / (1.0 - CLOSE_UP_SHARE)),
                 len(eligible))
    if wanted == 0:
        return shots

    # At least one wide separates details. A room can earn two distinct objects
    # when the story has enough space, but no object is repeated and the 25%
    # ceiling still controls the full film.
    chosen: dict[int, ResolvedDetail] = {}
    used_subjects: set[tuple[str, str]] = set()
    room_counts: dict[str, int] = {}
    last_index = -2
    for index, shot in eligible:
        if len(chosen) >= wanted or index - last_index < 2:
            continue
        if room_counts.get(shot.room_id, 0) >= 2:
            continue
        options = [
            detail for detail in resolved.get(shot.room_id, [])
            if (shot.room_id, detail.subject_key) not in used_subjects
        ]
        if not options:
            continue
        detail = max(
            options,
            key=lambda candidate: (
                candidate.score,
                candidate.detector_score,
                -candidate.source_photo_id,
            ),
        )
        chosen[index] = detail
        used_subjects.add((shot.room_id, detail.subject_key))
        room_counts[shot.room_id] = room_counts.get(shot.room_id, 0) + 1
        last_index = index

    out: list[Shot] = []
    for i, shot in enumerate(shots):
        out.append(shot)
        if not (head <= i < len(shots) - tail) or shot.is_close_up:
            continue
        detail = chosen.get(i)
        if i not in chosen:
            continue
        room_has_pair = any(
            candidate.room_id == shot.room_id and candidate.uses_keyframes
            for candidate in shots
        )
        straight_room = shot.room_type in STRAIGHT_SINGLE_ROOM_TYPES
        out.append(Shot(
            index=-1,
            room_id=shot.room_id,
            room_type=shot.room_type,
            photo_ids=[detail.source_photo_id],
            kind="single_image",
            act=shot.act,
            start_seconds=0.0,
            duration_seconds=shot.duration_seconds,
            generation_duration_seconds=shot.generation_duration_seconds,
            move=choose_move(
                f"{shot.room_id}:closeup:{i}",
                allow_pan=not straight_room,
                allow_tilt=not straight_room and not room_has_pair,
            ),
            is_hero=shot.is_hero,
            is_close_up=True,
            crop_box=detail.crop.box if detail.crop else None,
            crop_score=detail.crop.score if detail.crop else detail.score,
            source_size=detail.source_size,
            detail_subject=detail.subject_label,
            detector_score=detail.detector_score,
            subject_box=detail.subject_box,
            detail_framing=detail.framing,
        ))

    cursor = 0.0
    renumbered: list[Shot] = []
    for index, shot in enumerate(out):
        renumbered.append(replace(shot, index=index,
                                  start_seconds=round(cursor, 3)))
        cursor += shot.duration_seconds
    return renumbered
