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
