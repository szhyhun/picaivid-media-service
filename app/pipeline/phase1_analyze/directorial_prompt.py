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

from app.pipeline.phase1_analyze.camera_move import (
    STRAIGHT_SINGLE_ROOM_TYPES,
    axes_of,
    compose_label,
)

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

# Applies to every shot. The reference pixels are the source of truth; the
# camera path is the only intentional change. Keep this positive and concrete
# rather than using a negative prompt, which some providers interpret poorly.
_INVARIANTS = (
    "Treat the supplied reference frame or frames as the complete visual source "
    "of truth. Preserve the same scene identity and exact architecture, geometry, "
    "proportions, object placement, lighting, and every visible furnishing, "
    "fixture, surface, opening, reflection, shadow, and empty area exactly as "
    "shown. Photorealistic. Perfectly stabilized motorized camera rig with a "
    "level frame and straight verticals. The camera path is the only intentional "
    "change; use one continuous, controlled move at constant speed."
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


# The lead clause, chosen by the move's dominant axis.
_SINGLE_MOTION_PHRASES: dict[str, str] = {
    "push in": "Slow push in",
    "pull out": "Slow pull back",
    "pan right": "Slow horizontal pan to the right",
    "pan left": "Slow horizontal pan to the left",
    "tilt up": "Slow upward vertical reframe",
    "tilt down": "Slow downward vertical reframe",
    "static hold": "Locked-off camera with only subtle natural movement",
}

# What a secondary axis adds on top of the lead clause. 87% of the singles in
# a real listing move on two or three axes, matching the reference films, so
# naming only the dominant one throws away most of the measured movement --
# and the word "straight" it used to carry actively told the model not to drift.
#
# The vertical axis is never called a "tilt". In photography a tilt is a pitch,
# but the word also names a slanted horizon, and that ambiguity is the one thing
# the invariants ("level frame", "straight verticals") exist to prevent. A
# reframe is unambiguous: the camera rises, the horizon stays level.
_SECONDARY_MOTION_PHRASES: dict[str, str] = {
    "pan right": "drifting gently to the right",
    "pan left": "drifting gently to the left",
    "tilt up": "rising gently",
    "tilt down": "lowering gently",
}


def _motion_sentence(label: str) -> str:
    """Render a compound move as one direction the provider can follow."""
    if label == "static hold":
        return f"{_SINGLE_MOTION_PHRASES['static hold']}."

    atoms = [a for a in (axes_of(label).values()) if a]
    if not atoms:
        return f"{_SINGLE_MOTION_PHRASES['static hold']}."

    lead, rest = atoms[0], [a for a in atoms[1:] if a in _SECONDARY_MOTION_PHRASES]
    if not rest:
        # A single axis genuinely is straight, and saying so suppresses the
        # drift the model would otherwise invent.
        return f"{_SINGLE_MOTION_PHRASES[lead].replace('Slow ', 'Slow, straight ')}."

    clauses = [_SECONDARY_MOTION_PHRASES[a] for a in rest]
    joined = clauses[0] if len(clauses) == 1 else f"{clauses[0]} and {clauses[1]}"
    return f"{_SINGLE_MOTION_PHRASES[lead]}, {joined}."


def _shot_move_label(shot) -> str:
    """Read the move from either an in-memory Shot or a stored timeline row."""
    move = getattr(shot, "move", None)
    if isinstance(move, str):
        return move
    label = getattr(move, "label", None)
    if label:
        return str(label)
    return str(getattr(shot, "move_label", None) or "")


def _with_note(shot, prompt: str) -> str:
    """Append the customer's own words.

    Appended rather than substituted: `prompt_override` is the tool for
    replacing the whole direction and it belongs to development, not to a
    listing agent who wants one more thing mentioned.
    """
    note = (getattr(shot, "prompt_note", None) or "").strip()
    return f"{prompt} {note}" if note else prompt


def _effective_move_label(
    shot,
    cls: str,
    *,
    is_first: bool,
    is_last: bool,
    room_has_pair: bool = False,
) -> str:
    """The compound move this shot will actually be generated with.

    Position decides whether the camera pushes or pulls -- that is a fact about
    where the shot sits in the film, not about its geometry. The pan and tilt
    the shot was measured with survive it, because they are what makes the move
    read as a camera rather than a zoom control, and 87% of real singles carry
    at least one of them.

    A customer's pick outranks all of it. The positional defaults describe what
    listing films usually do, which is a reason to start there and not a reason
    to overrule someone who looked at the shot.
    """
    if getattr(shot, "move_locked", False):
        # The composed move is stored separately, so the measured one below
        # stays intact and a reset has something to go back to.
        chosen = getattr(shot, "move_chosen", None)
        if chosen:
            return compose_label(axes_of(str(chosen).lower()))

    axes = axes_of(_shot_move_label(shot).lower())

    # Apply the same evidence policy to persisted timelines as to newly built
    # plans. Existing rows can carry an older random pan/tilt label, but a
    # threshold single still has no evidence for that path.
    if getattr(shot, "kind", "single_image") != "pair_keyframes":
        room = (getattr(shot, "room_type", None) or "").strip().lower()
        if room in STRAIGHT_SINGLE_ROOM_TYPES:
            axes["pan"] = None
            axes["tilt"] = None
        elif room_has_pair:
            axes["tilt"] = None

    if getattr(shot, "move_locked", False):
        return compose_label(axes)

    if cls == "detail":
        axes["zoom"] = "push in"
    elif cls in {"aerial", "closing"} or is_last:
        axes["zoom"] = "pull out"
    elif is_first or cls in {"approach", "threshold"}:
        axes["zoom"] = "push in"

    return compose_label(axes)


def effective_move_label(shot, *, is_first: bool = False,
                         is_last: bool = False,
                         room_has_pair: bool = False) -> str:
    """The move this shot's prompt will actually ask the provider for.

    The editor must show this, not the raw stored label: a control that
    disagrees with the prompt it drives is worse than no control.
    """
    cls = shot_class(
        getattr(shot, "room_type", None),
        is_close_up=bool(getattr(shot, "is_close_up", False)),
        act=getattr(shot, "act", None),
        is_last=is_last,
    )
    return _effective_move_label(
        shot,
        cls,
        is_first=is_first,
        is_last=is_last,
        room_has_pair=room_has_pair,
    )


def build_shot_prompt(shot, *, squeeze_factor: float = 1.0,
                      is_first: bool = False, is_last: bool = False,
                      room_has_pair: bool = False) -> str:
    """Write the motion direction for one generated clip.

    `shot` needs `room_type`, `kind`, `is_close_up`, `act` and `move`; both the
    in-memory `Shot` and a stored `TimelineClip` satisfy that.
    """
    override = getattr(shot, "prompt_override", None)
    if override and override.strip():
        # Custom motion text may replace the default direction, but never the
        # provider-safety contract shared by every shot.
        return _with_note(shot, f"{override.strip()} {_INVARIANTS}")

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
        motion = _motion_sentence(
            _effective_move_label(
                shot,
                cls,
                is_first=is_first,
                is_last=is_last,
                room_has_pair=room_has_pair,
            ))

    subject = getattr(shot, "detail_subject", None)
    if cls == "detail" and subject:
        motion = f"Slow, straight push in toward the {subject.lower()}."

    return _with_note(shot, f"{motion} {_INVARIANTS}")
