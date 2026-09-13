"""Fetch the generated clips, cut them, and store the finished film."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from sqlalchemy.orm import Session

from app.db.models import Job
from app.db.models.timeline import SOURCE_SECONDS
from app.pipeline.phase4_assembly.assemble import assemble_film
from app.pipeline.phase3_timeline.music import download_music
from app.services.storage.s3_client import s3_client

logger = logging.getLogger(__name__)


def assemble_timeline_film(db: Session, job_id: int, plan: list[dict],
                          *, music_uri: str | None = None,
                          music_offset: float = 0.0) -> dict:
    """Download each clip, render it into its slot, join, and upload.

    The plan carries `trim_start` and `duration` per shot rather than letting
    this module recompute them, so what gets rendered is exactly what the
    editor showed.
    """
    with tempfile.TemporaryDirectory() as work:
        shots = []
        for index, item in enumerate(plan):
            local = Path(work) / f"src{index:04d}.mp4"
            local.write_bytes(s3_client.download_bytes(item["s3_uri"]))
            shots.append({"path": str(local), "trim_start": item["trim_start"],
                          "duration": item["duration"]})

        music_path = None
        if music_uri:
            music_path = download_music(music_uri, Path(work) / "music.audio")
        out = Path(work) / "film.mp4"
        info = assemble_film(shots, out, source_seconds=SOURCE_SECONDS,
                             audio_path=music_path, audio_offset=music_offset)
        uri = s3_client.upload_bytes(
            f"jobs/{job_id}/film.mp4", out.read_bytes(), "video/mp4")

    job = db.query(Job).filter(Job.id == job_id).first()
    if job is not None:
        job.final_video_uri = uri
        db.commit()

    logger.info("FILM_STORED job_id=%s uri=%s duration=%.2fs",
                job_id, uri, info["duration"])
    return {**info, "s3_uri": uri}
