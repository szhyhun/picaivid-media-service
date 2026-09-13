"""What earns a place in the video.

A listing arrives with 30-60 photos. The finished film wants roughly 20-30
beats. That makes this a *selection* problem, not a coverage problem -- a
distinction the pipeline did not previously draw, so garages and storage rooms
competed for screen time with the kitchen.

Two independent judgements live here:

`EDITORIAL_TIER` is about the room. Some rooms sell a house, some only connect
it, and some are service spaces a buyer does not want to watch. The owner's
framing: "living room, aerial, exterior, nice bedroom with many matches,
kitchens, suite ... all that is nice and a selling feature of the house; garage
and storage and laundry should not be on the video."

`beat_kind` is about the evidence. A room with two verified, well-separated
photos can carry real camera motion; a room with one good photo can still carry
a short single-image beat. Both are legitimate -- the second is what makes the
recall ceiling survivable. Measured across six labelled listings, 40% of
same-room pairs have no geometric overlap at all and cannot be linked by any
merge policy, so insisting on pairs would silently drop good rooms.
"""
from __future__ import annotations

from enum import Enum


class Tier(str, Enum):
    """How much a room type deserves screen time."""

    FEATURE = "feature"        # sells the property; always eligible
    CONNECTIVE = "connective"  # earns a beat only to carry the story between rooms
    SERVICE = "service"        # excluded unless nothing else is available


EDITORIAL_TIER: dict[str, Tier] = {
    # The property's case for itself.
    "aerial view": Tier.FEATURE,
    "exterior front": Tier.FEATURE,
    "front yard": Tier.FEATURE,
    "building exterior": Tier.FEATURE,
    "living room": Tier.FEATURE,
    "kitchen": Tier.FEATURE,
    "dining room": Tier.FEATURE,
    "bedroom": Tier.FEATURE,
    "bathroom": Tier.FEATURE,
    "office": Tier.FEATURE,
    "backyard": Tier.FEATURE,
    "patio": Tier.FEATURE,
    "pool": Tier.FEATURE,
    "exterior back": Tier.FEATURE,
    "amenities": Tier.FEATURE,      # for apartments this *is* the selling point
    "gym": Tier.FEATURE,
    "building lobby": Tier.CONNECTIVE,
    "elevator lobby": Tier.CONNECTIVE,
    "walk-in closet": Tier.SERVICE,
    # Needed to move between spaces, but not a destination.
    "entrance": Tier.CONNECTIVE,
    "hallway": Tier.CONNECTIVE,
    "shared space": Tier.CONNECTIVE,
    # Nobody watches a listing film for these.
    "garage": Tier.SERVICE,
    "storage": Tier.SERVICE,
    "laundry room": Tier.SERVICE,
    "basement": Tier.SERVICE,
    "attic": Tier.SERVICE,
}

DEFAULT_TIER = Tier.CONNECTIVE

# --------------------------------------------------------------------- rhythm
# Measured from three professionally produced listing films. Every one cuts on a
# metronomic grid of exactly two musical beats per shot, and every implied tempo
# lands on a round number:
#
#   6880 Wallace Dr    43.0s  grid 1.062s  -> 113 BPM   fit error 67 ms
#   2738 Whitehead Pl  70.7s  grid 1.333s  ->  90 BPM   fit error 34 ms
#   2490 Anthony Pl    59.2s  grid 1.200s  -> 100 BPM   fit error 15 ms
#
# So the music's tempo, not the room count, sets the edit. Shot length is
# 120/BPM seconds and never varies within a film.
SHOT_BEATS = 2

# The provider currently returns a three-second source for every beat. The
# editor later trims a stable two-beat window to the selected music grid. Keep
# this independent from `shot_seconds`, which is the final on-timeline duration.
SOURCE_GENERATION_SECONDS = 3.0

# Rooms change roughly every 2.2 shots: Wallace Dr held 16 rooms across 35
# shots, so a room is typically shown from two angles before moving on. Measured
# on one film only -- treat as a starting point, not a law.
SHOTS_PER_ROOM = 2.2

# Title card and closing branding do not sit on the cutting grid. Fitting a
# head/tail allowance to the three films (predicting shot counts of 35/48/46)
# bottoms out at 5s with a mean error of one shot; smaller values overpredict.
TITLE_ALLOWANCE_SECONDS = 5.0

# Guard rails. Under about 8 shots nothing reads as a tour; past 60 a listing
# film becomes a slideshow.
MIN_SHOTS = 8
MAX_SHOTS = 60


def tier_for(room_type: str | None) -> Tier:
    return EDITORIAL_TIER.get((room_type or "").strip().lower(), DEFAULT_TIER)


def is_eligible(room_type: str | None, *, allow_service: bool = False) -> bool:
    """Whether a room may appear at all.

    `allow_service` is the escape hatch for a listing so small that excluding
    service rooms would leave nothing to cut -- a studio apartment with a
    kitchen, a bathroom and a laundry closet still needs a film.
    """
    tier = tier_for(room_type)
    if tier is Tier.SERVICE:
        return allow_service
    return True


def beat_kind(photo_count: int, verified_pair: bool) -> str:
    """What this room can support, given the evidence actually available.

    Deliberately not gated on having a verified pair. A single strong photo of a
    kitchen makes a better beat than omitting the kitchen, and 40% of same-room
    pairs are unlinkable by construction.
    """
    if photo_count >= 2 and verified_pair:
        return "multi_view_move"   # real parallax between two verified views
    if photo_count >= 2:
        return "sequence_cut"      # two photos, no verified geometry: cut, don't interpolate
    return "single_image_move"     # one photo, synthetic move


def shot_seconds(bpm: float) -> float:
    """How long one shot holds, given the music. Two beats, always."""
    return SHOT_BEATS * 60.0 / float(bpm)


def shot_budget(target_seconds: float, bpm: float) -> int:
    """How many shots a cut of this length holds at this tempo.

    The music sets the edit: at 100 BPM a shot is 1.2s, so a 45s film is 37
    shots whether the property has 8 rooms or 30. A property with more rooms
    than the budget must drop rooms, never compress them below the grid.
    """
    usable = max(0.0, target_seconds - TITLE_ALLOWANCE_SECONDS)
    return max(MIN_SHOTS, min(MAX_SHOTS, round(usable / shot_seconds(bpm))))


def room_budget(target_seconds: float, bpm: float) -> int:
    """How many distinct rooms that many shots can cover."""
    return max(1, round(shot_budget(target_seconds, bpm) / SHOTS_PER_ROOM))
