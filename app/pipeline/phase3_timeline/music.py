"""Music download and beat-grid analysis for the editable timeline."""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np
from scipy.signal import fftconvolve, stft

from app.core.config import settings
from app.services.storage.s3_client import s3_client

SR = 22_050
HOP = 512


class MusicAnalysisError(RuntimeError):
    """The selected track could not be downloaded or analysed."""


@dataclass(frozen=True)
class MusicAnalysis:
    bpm: float
    beat_offset: float
    duration: float


def _suffix(uri: str) -> str:
    suffix = Path(urlparse(uri).path).suffix.lower()
    return suffix if suffix in {".mp3", ".m4a", ".mp4", ".wav", ".ogg", ".aac"} else ".audio"


def download_music(uri: str, destination: str | Path) -> Path:
    """Stream an S3 or HTTP(S) track to a local temporary file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()

    if scheme == "s3" or not scheme:
        try:
            data = s3_client.download_bytes(uri)
            destination.write_bytes(data)
        except Exception as exc:  # noqa: BLE001 - normalize storage failures
            raise MusicAnalysisError(f"Could not download music from storage: {exc}") from exc
        return destination

    if scheme not in {"http", "https"}:
        raise MusicAnalysisError("Music URI must use s3://, https://, or local http://.")
    if scheme == "http" and settings.ENVIRONMENT != "development":
        raise MusicAnalysisError("Plain HTTP music URLs are only allowed in development.")

    try:
        with httpx.stream("GET", uri, follow_redirects=True, timeout=120.0) as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    handle.write(chunk)
    except Exception as exc:  # noqa: BLE001 - normalize provider/storage failures
        destination.unlink(missing_ok=True)
        raise MusicAnalysisError(f"Could not download music URL: {exc}") from exc
    return destination


def analyze_audio(path: str | Path) -> MusicAnalysis:
    """Estimate tempo and first beat from spectral flux.

    The autocorrelation uses an FFT convolution so a several-minute catalog
    track does not turn tempo analysis into an accidental quadratic operation.
    """
    raw_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as handle:
            raw_path = handle.name
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
             "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", raw_path, "-y"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise MusicAnalysisError(f"ffmpeg could not read the music: {result.stderr[-300:]}")
        x = np.fromfile(raw_path, dtype=np.float32)
    finally:
        if raw_path:
            os.unlink(raw_path)

    if x.size < SR:
        raise MusicAnalysisError("Music must be at least one second long.")

    _, _, spectrum = stft(x, fs=SR, nperseg=2048, noverlap=2048 - HOP)
    flux = np.maximum(0, np.diff(np.abs(spectrum), axis=1)).sum(axis=0)
    if flux.size < 4 or not np.isfinite(flux).all():
        raise MusicAnalysisError("Music has no usable beat signal.")
    flux = (flux - flux.mean()) / (flux.std() + 1e-9)
    hop_seconds = HOP / SR
    autocorrelation = fftconvolve(flux, flux[::-1], mode="full")[len(flux) - 1:]
    low = max(1, int(round((60 / 200) / hop_seconds)))
    high = min(len(autocorrelation) - 1, int(round((60 / 60) / hop_seconds)))
    if high < low:
        raise MusicAnalysisError("Music is too short for tempo estimation.")
    lag = max(range(low, high + 1), key=lambda index: autocorrelation[index])
    if lag <= 0:
        raise MusicAnalysisError("Music tempo could not be estimated.")

    offset_index = max(
        range(lag),
        key=lambda offset: float(flux[np.arange(offset, len(flux), lag)].sum()),
    )
    return MusicAnalysis(
        bpm=60.0 / (lag * hop_seconds),
        beat_offset=offset_index * hop_seconds,
        duration=x.size / SR,
    )


def analyze_music_uri(uri: str) -> MusicAnalysis:
    """Download and analyse a track without retaining a local cache."""
    with tempfile.TemporaryDirectory() as work:
        path = download_music(uri, Path(work) / f"music{_suffix(uri)}")
        return analyze_audio(path)
