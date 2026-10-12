"""Cut the generated clips into the finished film.

Two things happen to every clip, and both come from measurement rather than
taste.

**Middle window.** We buy five seconds of source and show one beat-length
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
from functools import lru_cache
from pathlib import Path

from app.db.models.timeline import SOURCE_SECONDS

logger = logging.getLogger(__name__)

OUTPUT_FPS = 30
# 2 blended frames is a 360-degree shutter -- the most motion blur an honest
# frame budget allows. Cinema's 180-degree look is 1; the reference films sit
# nearer 360 because the camera never stops.
BLEND_FRAMES = 2


# Transfer functions that mean the source carries high dynamic range. Ray 3.2
# generates HDR natively, and we pay double for it.
HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67", "smpte428", "bt2020-10", "bt2020-12"})


class AssemblyError(RuntimeError):
    """ffmpeg could not produce the film."""


@lru_cache(maxsize=None)
def _has_filter(name: str) -> bool:
    """Whether this ffmpeg build offers a filter.

    Debian's ffmpeg carries zscale; several Homebrew builds do not. The chain
    has to be chosen from what is actually installed rather than from what the
    production image happens to have, or the film renders correctly on one
    machine and wrongly on another.
    """
    try:
        listing = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                                 capture_output=True, text=True).stdout
    except OSError:
        return False
    return any(line.split()[1:2] == [name] for line in listing.splitlines() if line.strip())


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


def tonemap_chain() -> list[str]:
    """Map HDR down to Rec.709 without crushing it.

    We buy HDR because it is visibly better material, and the film ships as
    8-bit H.264 because that is what listing sites and social players accept.
    Those are only compatible if the conversion is a real tone-map. Handing a
    PQ source straight to `format=yuv420p` is not a conversion at all -- the
    values are reinterpreted as if they were already Rec.709, which clips the
    highlights and leaves the picture flat and desaturated. That is the washed
    out look, and it is worse than never having generated HDR at all.

    `zscale` is what linearises PQ, and there is no acceptable substitute.
    `tonemap` on its own assumes linear input; measured on a real PQ clip it
    darkened mean luma from 126 to 43, which is not an approximation of the
    right answer but a different and worse picture. So this requires zscale
    and the caller fails loudly without it -- a film that is visibly wrong is
    harder to notice, and much harder to explain, than one that did not build.
    """
    return [
        "zscale=t=linear:npl=100",
        "format=gbrpf32le",
        "zscale=p=bt709",
        # Hable keeps highlight detail -- windows and skies are exactly what a
        # property film must not blow out.
        "tonemap=tonemap=hable:desat=0",
        "zscale=t=bt709:m=bt709:r=tv",
    ]


def shot_filter(*, source_seconds: float, trim_start: float,
                duration: float, blend: int = BLEND_FRAMES,
                fps: int = OUTPUT_FPS, tonemap: list[str] | None = None) -> str:
    """The filter chain that takes a native-speed source window into one slot."""
    usable = max(0.05, source_seconds - max(0.0, trim_start))
    window = min(duration, usable)
    chain = [f"trim=start={trim_start:.3f}:duration={window:.3f}",
             "setpts=PTS-STARTPTS"]
    # An explicit trim too close to the source end still fills the beat slot.
    # Normal centre windows retain native speed.
    if window < duration:
        chain.append(f"setpts={duration / window:.4f}*PTS")
    # Tone-map before blending. tmix averages whatever it is handed, and
    # averaging PQ-coded values is averaging the wrong numbers -- the shutter
    # smear would be computed in a space where equal steps are not equal light.
    if tonemap:
        chain.extend(tonemap)
    if blend > 1:
        chain.append(f"tmix=frames={blend}:weights='{' '.join(['1'] * blend)}'")
    chain.append(f"fps={fps}")
    chain.append("format=yuv420p")
    if tonemap:
        # Relabel the frames as Rec.709. Encoder flags alone do not do this:
        # ffmpeg propagates the source's BT.2020/PQ tags through the filter
        # graph, so the file would claim to be HDR while holding tone-mapped
        # SDR pixels, and every player would tone-map it a second time.
        chain.append("setparams=color_primaries=bt709:color_trc=bt709:"
                     "colorspace=bt709:range=tv")
    return ",".join(chain)


def is_hdr(path: str | Path) -> bool:
    """Whether this clip carries a high-dynamic-range transfer function."""
    try:
        out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=color_transfer,color_primaries",
                    "-of", "json", str(path)]).stdout
    except AssemblyError:
        return False
    stream = (json.loads(out).get("streams") or [{}])[0]
    return str(stream.get("color_transfer") or "").lower() in HDR_TRANSFERS


def render_shot(src: str | Path, dest: str | Path, *, source_seconds: float,
                trim_start: float, duration: float,
                blend: int = BLEND_FRAMES, fps: int = OUTPUT_FPS) -> Path:
    """Trim and blur one clip into its slot, tone-mapping it if it is HDR."""
    tonemap = None
    if is_hdr(src):
        if not _has_filter("zscale"):
            raise AssemblyError(
                f"{src} is HDR and this ffmpeg has no zscale filter, so it "
                "cannot be tone-mapped correctly. Install an ffmpeg built with "
                "libzimg (Debian's ffmpeg package has it), or generate with "
                "RUNWARE_HDR=false.")
        tonemap = tonemap_chain()
    vf = shot_filter(source_seconds=source_seconds, trim_start=trim_start,
                     duration=duration, blend=blend, fps=fps, tonemap=tonemap)
    _run(["ffmpeg", "-y", "-i", str(src), "-vf", vf, "-an",
          "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
          # Say what the output actually is. Without this the file inherits the
          # source's BT.2020/PQ tags while holding Rec.709 pixels, and players
          # tone-map it a second time.
          "-colorspace", "bt709", "-color_primaries", "bt709",
          "-color_trc", "bt709", "-color_range", "tv", str(dest)])
    return Path(dest)


def concat(parts: list[Path], dest: str | Path,
           audio_path: str | Path | None = None,
           audio_offset: float = 0.0) -> Path:
    """Join the slots with hard cuts, and lay the music under them."""
    if not parts:
        raise AssemblyError("Nothing to assemble.")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        for part in parts:
            fh.write(f"file '{Path(part).resolve()}'\n")
        listing = fh.name

    args = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listing]
    if audio_path:
        # Loop a short track so it cannot truncate the film; -shortest then
        # ends at the video, never at the audio file's duration.
        args += ["-stream_loop", "-1"]
        if audio_offset > 0:
            # The first detected beat becomes time zero for the first shot.
            args += ["-ss", f"{audio_offset:.3f}"]
        args += ["-i", str(audio_path),
                 "-map", "0:v", "-map", "1:a",
                 "-c:a", "aac", "-shortest"]
    args += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(dest)]
    try:
        _run(args)
    finally:
        Path(listing).unlink(missing_ok=True)
    return Path(dest)


def assemble_film(shots: list[dict], dest: str | Path, *,
                  source_seconds: float = SOURCE_SECONDS,
                  audio_path: str | Path | None = None,
                  audio_offset: float = 0.0,
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
            # Pairs and singles are bought at different lengths, so the
            # squeeze factor is per shot. One film-wide value would stretch
            # every single by 5/3 and wreck the beat grid.
            render_shot(shot["path"], part,
                        source_seconds=float(
                            shot.get("source_seconds") or source_seconds),
                        trim_start=float(shot.get("trim_start") or 0.0),
                        duration=float(shot["duration"]), blend=blend, fps=fps)
            parts.append(part)
        concat(parts, dest, audio_path=audio_path, audio_offset=audio_offset)

    info = probe(dest)
    logger.info("FILM_ASSEMBLED shots=%s duration=%.2fs fps=%s",
                len(shots), info["duration"], info["fps"])
    return info
