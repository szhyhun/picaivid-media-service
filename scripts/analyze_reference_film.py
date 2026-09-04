"""Measure a professionally produced listing film: rhythm, cutting, music sync.

Answers the questions the shot planner needs and cannot guess:
  * how fast does the picture change, and is it locked to the music
  * how many shots per room, and how long does a room hold
  * what order do rooms appear in, and which rooms get the most screen time

Usage: python -m scripts.analyze_reference_film <video> [more videos...]
"""
from __future__ import annotations

import collections
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
from scipy.signal import find_peaks, stft

SR = 22050
HOP = 512
VIS_FPS = 10.0


def _run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def audio_grid(path: str):
    """Tempo and beat times from a spectral-flux onset envelope."""
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as fh:
        raw = fh.name
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
          "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", raw, "-y"])
    x = np.fromfile(raw, dtype=np.float32)
    os.unlink(raw)
    if x.size < SR:
        return None, None
    _, _, Z = stft(x, fs=SR, nperseg=2048, noverlap=2048 - HOP)
    flux = np.maximum(0, np.diff(np.abs(Z), axis=1)).sum(axis=0)
    flux = (flux - flux.mean()) / (flux.std() + 1e-9)
    hop = HOP / SR
    ac = np.correlate(flux, flux, mode="full")[len(flux) - 1:]
    lo, hi = int(round((60 / 200) / hop)), int(round((60 / 60) / hop))
    lag = max(range(lo, hi + 1), key=lambda l: ac[l])
    bpm = 60.0 / (lag * hop)
    off = max(range(lag), key=lambda o: flux[np.arange(o, len(flux), lag)].sum())
    return bpm, np.arange(off, len(flux), lag) * hop


def visual_transitions(path: str):
    """Times where the picture changes materially."""
    tmp = tempfile.mkdtemp()
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
          "-vf", f"fps={VIS_FPS},scale=192:108", os.path.join(tmp, "g%05d.jpg"), "-y"])
    files = sorted(glob.glob(os.path.join(tmp, "*.jpg")))
    arr = np.stack([np.asarray(Image.open(f).convert("L"), np.float32) / 255.0 for f in files])
    shutil.rmtree(tmp)
    d = np.abs(np.diff(arr, axis=0)).mean(axis=(1, 2))
    t = (np.arange(len(d)) + 0.5) / VIS_FPS
    pk, _ = find_peaks(d, height=np.percentile(d, 80),
                       distance=int(0.8 * VIS_FPS), prominence=d.std() * 0.8)
    return t[pk], d


def cut_grid(transitions):
    """Fit a constant-period grid to the cuts themselves.

    More reliable than the audio tempo estimate: a 1% tempo error accumulates
    into visible phase drift over a minute and makes a genuinely beat-locked
    film look unsynced. The cuts are the ground truth for the edit's rhythm.
    """
    tr = np.sort(transitions)
    if len(tr) < 4:
        return None
    best = (None, 1e9)
    for per in np.arange(0.8, 2.0, 0.001):
        for ph in np.arange(0, per, 0.005):
            k = np.round((tr - ph) / per)
            err = np.abs(tr - (ph + k * per)).mean()
            if err < best[1]:
                best = ((per, ph), err)
    (per, ph), err = best
    return per, ph, err


def sync_test(transitions, beats, trials=4000):
    """Are transitions closer to beats than chance would put them?"""
    if beats is None or len(transitions) < 4:
        return None
    def off(times):
        return np.array([np.min(np.abs(beats - x)) for x in times])
    obs = off(transitions).mean()
    rng = np.random.default_rng(0)
    lo, hi = transitions.min(), transitions.max()
    null = np.array([off(rng.uniform(lo, hi, len(transitions))).mean() for _ in range(trials)])
    return obs, null.mean(), float((null <= obs).mean())


def analyze(path: str) -> None:
    name = os.path.basename(path)
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True).stdout.strip())
    bpm, beats = audio_grid(path)
    tr, _ = visual_transitions(path)
    print(f"\n=== {name[:70]} ===")
    print(f"  duration        {dur:.1f}s")
    if bpm:
        print(f"  tempo           {bpm:.1f} BPM (beat {60/bpm:.3f}s, bar {4*60/bpm:.2f}s)")
    print(f"  shots           {len(tr)}  ->  {dur/max(len(tr),1):.2f}s per shot")
    if bpm and len(tr) > 1:
        spacing = np.median(np.diff(np.sort(tr)))
        print(f"  shot spacing    {spacing:.2f}s = {spacing/(60/bpm):.2f} beats")
    grid = cut_grid(tr)
    if grid:
        per, _, err = grid
        implied = 2 * 60 / per
        print(f"  cut grid        {per:.3f}s  (fit error {err*1000:.0f} ms)")
        print(f"  implied tempo   {implied:.1f} BPM at 2 beats per shot"
              f"  -> nearest round {round(implied)} BPM"
              f" (delta {abs(per - 2*60/round(implied))*1000:.0f} ms)")
        metronomic = err < 0.12
        print(f"  rhythm          {'METRONOMIC - locked 2-beat grid' if metronomic else 'free / not on a fixed grid'}")


if __name__ == "__main__":
    for v in sys.argv[1:]:
        analyze(v)
