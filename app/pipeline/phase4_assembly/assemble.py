"""Cut the generated clips into the finished film.

Two things happen to every clip, and both come from measurement rather than
taste.

**Middle window.** We buy three seconds of source and show one beat-length
window from the stable centre -- about 1.2s at 100 BPM. Input anchoring and
generation artefacts are most visible at the ends, so accelerating the full
source would put the weakest material into the final cut.

**Shutter.** That surplus is the point. Generated frames are individually
perfect and perfectly sharp, and a run of sharp frames reads as a slideshow no
matter how many of them there are -- which is why adding frames by interpolation
did not help. A real camera's shutter stays open and smears whatever moves.
Blending pairs of *real* generated frames synthesises that smear, and blending
real frames beats blending invented ones.

Cuts are hard, never dissolved: peak/trough frame differencing across all three
reference films puts elevated change at ~4% of a shot, one or two frames.
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_FPS = 30
# 2 blended frames is a 360-degree shutter -- the most motion blur an honest
# frame budget allows. Cinema's 180-degree look is 1; the reference films sit
# nearer 360 because the camera never stops.
BLEND_FRAMES = 2


class AssemblyError(RuntimeError):
    """ffmpeg could not produce the film."""


def _run(args: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssemblyError(f"{' '.join(args[:3])}... failed: {result.stderr[-500:]}")
    return result


def probe(path: str | Path) -> dict:
    """Frame rate, frame count and duration of a video file."""
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate,nb_frames,width,height",
                "-show_entries", "format=duration",
                "-of", "json", str(path)]).stdout
    data = json.loads(out)
    stream = (data.get("streams") or [{}])[0]
    rate = stream.get("r_frame_rate") or "0/1"
    num, _, den = rate.partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return {
        "fps": round(fps, 3),
        "frames": int(stream.get("nb_frames") or 0),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "duration": float((data.get("format") or {}).get("duration") or 0.0),
    }


def shot_filter(*, source_seconds: float, trim_start: float,
                duration: float, blend: int = BLEND_FRAMES,
                fps: int = OUTPUT_FPS) -> str:
    """The filter chain that takes a native-speed source window into one slot."""
    usable = max(0.05, source_seconds - max(0.0, trim_start))
    window = min(duration, usable)
    chain = [f"trim=start={trim_start:.3f}:duration={window:.3f}",
             "setpts=PTS-STARTPTS"]
    # An explicit trim too close to the source end still fills the beat slot.
    # Normal centre windows retain native speed.
    if window < duration:
        chain.append(f"setpts={duration / window:.4f}*PTS")
    if blend > 1:
        chain.append(f"tmix=frames={blend}:weights='{' '.join(['1'] * blend)}'")
    chain.append(f"fps={fps}")
    return ",".join(chain)


def render_shot(src: str | Path, dest: str | Path, *, source_seconds: float,
                trim_start: float, duration: float,
                blend: int = BLEND_FRAMES, fps: int = OUTPUT_FPS) -> Path:
    """Trim and blur one clip into its slot."""
    vf = shot_filter(source_seconds=source_seconds, trim_start=trim_start,
                     duration=duration, blend=blend, fps=fps)
    _run(["ffmpeg", "-y", "-i", str(src), "-vf", vf, "-an",
          "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(dest)])
    return Path(dest)


def concat(parts: list[Path], dest: str | Path,
           audio_path: str | Path | None = None) -> Path:
    """Join the slots with hard cuts, and lay the music under them."""
    if not parts:
        raise AssemblyError("Nothing to assemble.")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        for part in parts:
            fh.write(f"file '{Path(part).resolve()}'\n")
        listing = fh.name

    args = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listing]
    if audio_path:
        # -shortest so a long track is cut to the film, never the other way.
        args += ["-i", str(audio_path), "-map", "0:v", "-map", "1:a",
                 "-c:a", "aac", "-shortest"]
    args += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(dest)]
    try:
        _run(args)
    finally:
        Path(listing).unlink(missing_ok=True)
    return Path(dest)


def assemble_film(shots: list[dict], dest: str | Path, *,
                  source_seconds: float = 3.0,
                  audio_path: str | Path | None = None,
                  blend: int = BLEND_FRAMES, fps: int = OUTPUT_FPS) -> dict:
    """Render every shot into its slot and join them.

    `shots` are dicts of `{path, trim_start, duration}` in screen order.
    """
    if not shots:
        raise AssemblyError("Nothing to assemble.")

    with tempfile.TemporaryDirectory() as work:
        parts = []
        for index, shot in enumerate(shots):
            part = Path(work) / f"{index:04d}.mp4"
            render_shot(shot["path"], part, source_seconds=source_seconds,
                        trim_start=float(shot.get("trim_start") or 0.0),
                        duration=float(shot["duration"]), blend=blend, fps=fps)
            parts.append(part)
        concat(parts, dest, audio_path=audio_path)

    info = probe(dest)
    logger.info("FILM_ASSEMBLED shots=%s duration=%.2fs fps=%s",
                len(shots), info["duration"], info["fps"])
    return info
