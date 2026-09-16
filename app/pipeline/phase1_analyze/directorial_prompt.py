"""Render a shot line as a directorial script for one-pass video generation.

Models like Seedance 2.5 generate a 30-second multi-shot sequence in a single
pass from many reference images plus a prompt. The obvious worry is that a
free-running model would discard the ordering work -- but the prompt is exactly
where that work goes. The shot line already knows the sequence, the camera move
per shot, and the timing on the music grid; written out, that *is* a director's
script.

So the pipeline's output is not competing with the generator, it is instructing
it. Everything measured from the reference films -- the approach order, the
living-room-first rule, outdoor as the late payoff, close-ups every two or three
shots, push-in dominance, the 120/BPM grid -- becomes prompt text.

Two things this cannot express, and they are the honest limits:

* **Reference-to-shot binding.** ByteDance's documentation does not confirm that
  you can say "reference image 7 is shot 3". The prompt names the room and the
  move; which supplied photograph the model reaches for is its choice.
* **Frame-accurate cuts.** Timestamps are stated, but nothing guarantees a cut
  lands on the beat. Beat-locked cutting is the most consistent finding across
  all three reference films, so this is the real thing being traded away.
"""
from __future__ import annotations

from dataclasses import dataclass

# How a measured camera move reads as direction. The reference films' vocabulary
# is small and concrete, which suits a prompt: nine words cover every shot.
MOVE_PHRASES: dict[str, str] = {
    "push in": "slow push in",
    "pull out": "slow pull back",
    "pan right": "pan right",
    "pan left": "pan left",
    "tilt up": "tilt up",
    "tilt down": "tilt down",
    "static hold": "locked off, no camera movement",
}

# Plain-English room names. The instance ids carry disambiguating suffixes
# ("bedroom-a", "living-room-staged") that mean nothing to a generator.
ROOM_PHRASES: dict[str, str] = {
    "aerial view": "an aerial view of the property",
    "front yard": "the front yard approaching from the street",
    "exterior front": "the front facade and front door",
    "building exterior": "the exterior of the building",
    "entrance": "the entrance hall just inside the front door",
    "hallway": "a hallway",
    "living room": "the living room",
    "dining room": "the dining room",
    "kitchen": "the kitchen",
    "bedroom": "a bedroom",
    "bathroom": "a bathroom",
    "office": "the home office",
    "patio": "the patio",
    "pool": "the pool",
    "backyard": "the back garden",
    "exterior back": "the rear exterior",
    "amenities": "the building amenities",
    "shared space": "the shared residents' area",
    "gym": "the fitness room",
    "building lobby": "the building lobby",
    "elevator lobby": "the elevator lobby",
    "walk-in closet": "the walk-in wardrobe",
}


@dataclass(frozen=True)
class DirectorialScript:
    """A prompt plus the reference images it expects to be given."""

    prompt: str
    reference_photo_ids: list[int]
    duration_seconds: float
    shot_count: int


def _move_phrase(move_label: str) -> str:
    """Turn a compound move label into direction the model can follow."""
    parts = [MOVE_PHRASES.get(p.strip(), p.strip())
             for p in (move_label or "").split("+")]
    parts = [p for p in parts if p]
    if not parts:
        return "slow push in"
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} while you {' and '.join(parts[1:])}"


def _room_phrase(room_type: str, is_close_up: bool, subject: str | None) -> str:
    base = ROOM_PHRASES.get((room_type or "").strip().lower(), "the room")
    if not is_close_up:
        return base
    if subject:
        return f"a close-up of the {subject.lower()} in {base}"
    return f"a close-up detail in {base}"


def build_script(shots, *, bpm: float, property_description: str = "") -> DirectorialScript:
    """Write the shot line out as a numbered, timed director's script.

    Timings are stated per shot even though the generator may not honour them
    exactly: an explicit grid is more likely to produce even pacing than leaving
    rhythm to the model, and the reference films are uniformly metronomic.
    """
    if not shots:
        return DirectorialScript("", [], 0.0, 0)

    total = sum(s.duration_seconds for s in shots)
    header = [
        "A cinematic real estate tour film.",
        # State the shot length, not the beat: a generator has no use for the
        # underlying tempo, only for how long each shot must hold.
        f"{len(shots)} shots over {total:.0f} seconds. "
        f"Every shot is exactly {shots[0].duration_seconds:.2f} seconds long "
        f"-- two beats at {bpm:.0f} BPM -- with no variation.",
        "Hard cuts between shots. No dissolves, no fades, no speed ramps.",
        "Every shot has slow, continuous camera movement at constant speed.",
        "Photorealistic, natural daylight, no people, no text overlays.",
    ]
    if property_description:
        header.insert(1, property_description)

    lines = []
    for shot in shots:
        start = shot.start_seconds
        end = start + shot.duration_seconds
        room = _room_phrase(
            shot.room_type,
            getattr(shot, "is_close_up", False),
            getattr(shot, "detail_subject", None),
        )
        lines.append(
            f"Shot {shot.index + 1} ({start:.1f}-{end:.1f}s): {room}, "
            f"{_move_phrase(shot.move.label)}."
        )

    prompt = "\n".join(header) + "\n\n" + "\n".join(lines)
    photo_ids: list[int] = []
    for shot in shots:
        for pid in shot.photo_ids:
            if pid not in photo_ids:
                photo_ids.append(pid)

    return DirectorialScript(
        prompt=prompt,
        reference_photo_ids=photo_ids,
        duration_seconds=total,
        shot_count=len(shots),
    )


def split_for_generation(shots, *, max_seconds: float = 30.0) -> list[list]:
    """Split a film into segments a one-pass generator can produce.

    Seedance 2.5 generates up to 30 seconds in a single pass and offers
    multi-round extension to continue a scene while keeping the setting
    consistent, so a 60 or 90 second film is a small number of passes rather
    than one per shot.

    Segments break on act boundaries where possible: an act is a coherent stretch
    of the story, so a seam there is the least visible place to put one.
    """
    if not shots:
        return []

    segments: list[list] = []
    current: list = []
    elapsed = 0.0
    for shot in shots:
        would_be = elapsed + shot.duration_seconds
        act_change = bool(current) and shot.act != current[-1].act
        # Break at an act change once the segment is over half full, or whenever
        # the next shot would overflow the model's single-pass limit.
        if current and (would_be > max_seconds or (act_change and elapsed > max_seconds / 2)):
            segments.append(current)
            current, elapsed = [], 0.0
        current.append(shot)
        elapsed += shot.duration_seconds
    if current:
        segments.append(current)
    return segments


# --- Per-shot prompts for clip-by-clip image-to-video generation -------------
#
# Everything above builds ONE prompt for a whole film, for models that generate
# a multi-shot sequence in a single pass. Ray3.2 works the other way:
# one clip per shot, from one or two real photographs. That inverts the job.
# The model already has the pixels, so the prompt's only work is motion.
#
# Rules, all measured in REFERENCE_FILM_FINDINGS.md:
#
# * **Never describe the room.** Describing a building the model can already see
#   invites it to invent a different one. Motion instructions only.
# * **Pairs state direction by frame order.** The prompt says to travel from the
#   first reference frame to the second; it does not invent a free-form
#   left/right direction that the measured geometry may contradict.
# * **Singles must name one**, because nothing else determines it.
# * **Ask for a continuous slow move.** The editor takes a central beat-length
#   window at native speed, rather than accelerating a full source clip.
# * **Do not invent scene motion.** Ambient motion may be added only when a
#   grounded source signal supplies it; a room label alone is not evidence.

_AERIAL_TYPES = frozenset({"aerial view"})
_APPROACH_TYPES = frozenset({"front yard", "exterior front", "building exterior"})
_THRESHOLD_TYPES = frozenset({"entrance", "hallway"})
_OUTDOOR_TYPES = frozenset({"patio", "pool", "backyard", "exterior back"})

# Applies to every shot. The building and visible objects must not change; only
# the camera is allowed to move.
_INVARIANTS = (
    "Photorealistic. Perfectly stabilized motorized camera rig with a level "
    "frame and straight verticals. Preserve exact architecture, geometry, "
    "proportions, object placement, and lighting throughout. Use one "
    "continuous, controlled move at constant speed."
)


def shot_class(room_type: str | None, *, is_close_up: bool = False,
               act: int | None = None, is_last: bool = False) -> str:
    """Which of the seven directorial classes this shot belongs to."""
    room = (room_type or "").strip().lower()
    if is_close_up:
        return "detail"
    if room in _AERIAL_TYPES:
        return "closing" if is_last else "aerial"
    if room in _APPROACH_TYPES:
        return "closing" if (is_last or act == 3) else "approach"
    if room in _THRESHOLD_TYPES:
        return "threshold"
    if room in _OUTDOOR_TYPES:
        return "outdoor"
    return "interior"


def _speed_word(_squeeze_factor: float) -> str:
    """Source windows play at native speed, so one stable cue is enough."""
    return "Slow"


_SINGLE_MOTION_PHRASES: dict[str, str] = {
    "push in": "Slow, straight push in",
    "pull out": "Slow, straight pull back",
    "pan right": "Slow, straight horizontal pan to the right",
    "pan left": "Slow, straight horizontal pan to the left",
    "static hold": "Locked-off camera with only subtle natural movement",
}


def _shot_move_label(shot) -> str:
    """Read the move from either an in-memory Shot or a stored timeline row."""
    move = getattr(shot, "move", None)
    if isinstance(move, str):
        return move
    label = getattr(move, "label", None)
    if label:
        return str(label)
    return str(getattr(shot, "move_label", None) or "")


def _single_motion_label(
    shot,
    cls: str,
    *,
    is_first: bool,
    is_last: bool,
) -> str:
    """Reduce a compound move to one provider-safe camera axis.

    The still-image path has no second camera pose to constrain the model. A
    short vocabulary is intentional: compound pan/tilt/roll prompts are harder
    to follow and harder to review. The 2D preview may retain the full measured
    vector; this is the prompt contract sent to the video provider.
    """
    if cls == "detail":
        return "push in"
    if cls in {"aerial", "closing"} or is_last:
        return "pull out"
    if is_first or cls in {"approach", "threshold"}:
        return "push in"

    label = _shot_move_label(shot).lower()
    if "static hold" in label:
        return "static hold"
    # choose_move writes the zoom atom first. Preserve that primary intent when
    # a legacy compound label also contains a pan/tilt/roll atom.
    if "push in" in label:
        return "push in"
    if "pull out" in label:
        return "pull out"
    if "pan right" in label:
        return "pan right"
    if "pan left" in label:
        return "pan left"
    return "push in"


def build_shot_prompt(shot, *, squeeze_factor: float = 1.0,
                      is_first: bool = False, is_last: bool = False) -> str:
    """Write the motion direction for one generated clip.

    `shot` needs `room_type`, `kind`, `is_close_up`, `act` and `move`; both the
    in-memory `Shot` and a stored `TimelineClip` satisfy that.
    """
    override = getattr(shot, "prompt_override", None)
    if override and override.strip():
        # Custom motion text may replace the default direction, but never the
        # provider-safety contract shared by every shot.
        return f"{override.strip()} {_INVARIANTS}"

    room_type = getattr(shot, "room_type", None)
    is_close_up = bool(getattr(shot, "is_close_up", False))
    kind = getattr(shot, "kind", "single_image")
    act = getattr(shot, "act", None)

    cls = shot_class(room_type, is_close_up=is_close_up, act=act, is_last=is_last)
    speed = _speed_word(squeeze_factor)

    if kind == "pair_keyframes":
        # The frameImages payload binds reference one to the start and reference
        # two to the end. State that direction in the prompt as well, without
        # inventing a textual left/right direction that geometry may contradict.
        motion = (
            f"{speed}, smooth camera movement from the first reference frame "
            "to the second reference frame. "
            "Follow the spatial path and framing change between those two views."
        )
    else:
        motion_label = _single_motion_label(
            shot, cls, is_first=is_first, is_last=is_last)
        motion = f"{_SINGLE_MOTION_PHRASES[motion_label]}."

    subject = getattr(shot, "detail_subject", None)
    if cls == "detail" and subject:
        motion = f"Slow, straight push in toward the {subject.lower()}."

    return f"{motion} {_INVARIANTS}"
