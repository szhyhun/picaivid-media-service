"""Camera movement, measured from professionally produced listing films.

Derived from 139 shots across three films (6880 Wallace Dr, 2738 Whitehead Pl,
2490 Anthony Pl) by tracking features frame to frame and fitting a similarity
transform, sampling only the stable middle of each shot so the dissolves at the
boundaries do not contaminate the estimate.

What the measurements say, in order of how strongly they hold:

  pan       87% of shots   median 4.6% of frame width per second
  tilt      69%            median 5.0% of frame height per second
  push in   66%            median 6.7% per second
  roll      40%            median 1.36 deg/s
  pull out   9%            median 2.4% per second
  static     4%

Three things follow, and they are what a still-photo pipeline must imitate:

1. **Almost nothing is static.** Only 4% of shots hold still. A frozen photo
   reads as a slideshow, which is the exact failure the product exists to avoid.
2. **Movement compounds.** The canonical shot is a slow push combined with a
   pan and a slight tilt -- not one axis at a time.
3. **Push in beats pull out roughly 7:1**, and pans run rightward about 71% of
   the time. Pulling out is reserved: in the films it appears mostly on
   establishing exteriors, where revealing context is the point.

All four primitives are reproducible as 2D transforms on a still image, so they
apply to single-photo beats as much as to verified pairs.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

# Frequencies observed across the three reference films.
FREQ_PAN = 0.87
FREQ_TILT = 0.69
FREQ_PUSH_IN = 0.66
FREQ_ROLL = 0.40
FREQ_PULL_OUT = 0.09
FREQ_STATIC = 0.04

# Median magnitudes, expressed as fractions per second so they are resolution
# independent. Scale is a fraction of the current size; pan and tilt are
# fractions of frame width and height.
ZOOM_IN_RATE = 0.067
ZOOM_OUT_RATE = 0.024
PAN_RATE = 0.046
TILT_RATE = 0.050
ROLL_RATE_DEG = 1.36

# Rightward pans outnumbered leftward 86 to 35.
PAN_RIGHT_SHARE = 0.71


@dataclass(frozen=True)
class CameraMove:
    """A move to apply to one shot, as fractions per second."""

    zoom: float          # >0 pushes in, <0 pulls out
    pan: float           # >0 moves the frame right across the image
    tilt: float          # >0 moves the frame up across the image
    roll_deg: float      # slow rotational drift
    label: str

    def is_static(self) -> bool:
        return self.label == "static hold"


def _unit(seed: str, salt: str) -> float:
    """Deterministic value in [0,1) for a photo, so a plan is reproducible.

    Random moves would make two runs of the same listing differ, which makes
    review and regression testing impossible.
    """
    digest = hashlib.sha256(f"{seed}:{salt}".encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def choose_move(seed: str, *, is_establishing: bool = False) -> CameraMove:
    """Pick a move for one shot, matching the measured distribution.

    `is_establishing` flips the zoom preference: in the reference films the pull
    out is reserved for exteriors and aerials, where opening the frame reveals
    the setting rather than drawing attention into it.
    """
    r_static = _unit(seed, "static")
    if r_static < FREQ_STATIC:
        return CameraMove(0.0, 0.0, 0.0, 0.0, "static hold")

    if is_establishing:
        zoom = -ZOOM_OUT_RATE
        zoom_label = "pull out"
    elif _unit(seed, "zoomdir") < FREQ_PUSH_IN / (FREQ_PUSH_IN + FREQ_PULL_OUT):
        zoom = ZOOM_IN_RATE
        zoom_label = "push in"
    else:
        zoom = -ZOOM_OUT_RATE
        zoom_label = "pull out"

    pan = 0.0
    if _unit(seed, "pan") < FREQ_PAN:
        rightward = _unit(seed, "pandir") < PAN_RIGHT_SHARE
        pan = PAN_RATE if rightward else -PAN_RATE

    tilt = 0.0
    if _unit(seed, "tilt") < FREQ_TILT:
        tilt = TILT_RATE if _unit(seed, "tiltdir") < 0.5 else -TILT_RATE

    roll = 0.0
    if _unit(seed, "roll") < FREQ_ROLL:
        roll = ROLL_RATE_DEG if _unit(seed, "rolldir") < 0.5 else -ROLL_RATE_DEG

    parts = [zoom_label]
    if pan:
        parts.append("pan right" if pan > 0 else "pan left")
    if tilt:
        parts.append("tilt up" if tilt > 0 else "tilt down")
    return CameraMove(zoom, pan, tilt, roll, " + ".join(parts))


def required_headroom(move: CameraMove, shot_seconds: float) -> float:
    """How much larger than the output frame the source photo must be.

    A push of 6.7%/s over a 1.2s shot needs about 8% of extra pixels, and a pan
    needs its own margin. Cropping tighter than this crops into the edge of the
    photo and the move reveals blank frame.
    """
    zoom_need = abs(move.zoom) * shot_seconds
    pan_need = abs(move.pan) * shot_seconds
    tilt_need = abs(move.tilt) * shot_seconds
    return 1.0 + zoom_need + max(pan_need, tilt_need)


# Reversing which photograph is the start frame reverses the move. That is
# physics, not a UI preference: if the camera travelled A -> B it now travels
# B -> A, so a push in becomes a pull back and a rightward pan becomes leftward.
#
# It has to be applied to the stored move, not just the display. A pair carries
# no direction into its prompt -- the two frames decide it -- but the moment a
# pair is downgraded to a single image that label *does* drive the prompt, and a
# stale one would direct the camera the wrong way.
_OPPOSITE_ATOM: dict[str, str] = {
    "push in": "pull out",
    "pull out": "push in",
    "pan right": "pan left",
    "pan left": "pan right",
    "tilt up": "tilt down",
    "tilt down": "tilt up",
    "static hold": "static hold",
}


def reverse_label(label: str | None) -> str:
    """Invert every atom of a compound move label."""
    if not label:
        return ""
    atoms = [a.strip() for a in label.split("+") if a.strip()]
    return " + ".join(_OPPOSITE_ATOM.get(a, a) for a in atoms)


def reverse_vector(vector: dict | None) -> dict:
    """Negate a move vector. Zero stays zero, so a static hold is unaffected."""
    v = vector or {}
    return {
        "zoom": -float(v.get("zoom") or 0.0),
        "pan": -float(v.get("pan") or 0.0),
        "tilt": -float(v.get("tilt") or 0.0),
        "roll_deg": -float(v.get("roll_deg") or 0.0),
    }
