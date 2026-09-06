"""Generate a video for every planned shot on a timeline.

This runs on the CPU worker, not the GPU one. Runware is a remote API, so the
work here is waiting on the network, not on a graphics card.

Three things shape the design, and all three are about not wasting money:

* **The task id is ours and is written before we await.** A worker that dies
  mid-generation restarts against the same `provider_task_uuid` and collects the
  clip it already paid for, instead of ordering a second one.
* **Concurrency is capped.** Runware queues dynamically rather than rate-limiting,
  and their guidance is 2-4 in flight; pushing past that saturates the queue and
  slows everything down without generating anything sooner.
* **One failure does not stop the film.** A failed shot keeps its slot on the
  timeline and can be retried alone. A partial film is useful; an aborted batch
  is not.
"""
from __future__ import annotations

import asyncio
import logging
import uuid

import httpx
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Clip, JobPhoto
from app.db.models.timeline import SOURCE_SECONDS, Timeline, TimelineClip
from app.pipeline.phase1_analyze.directorial_prompt import build_shot_prompt
from app.pipeline.progress import NullProgress
from app.services.runware.client import RunwareError, generate_video
from app.services.storage.s3_client import s3_client

logger = logging.getLogger(__name__)


def _photo_urls(db: Session, photo_ids: list[int]) -> list[str]:
    """Presigned source images, valid long enough to survive a queue.

    The UI's one-hour URLs are too short here: a submitted task can sit in
    Runware's queue for minutes before it fetches anything.
    """
    rows = {int(p.id): p for p in
            db.query(JobPhoto).filter(JobPhoto.id.in_(photo_ids)).all()}
    urls = []
    for pid in photo_ids:
        row = rows.get(int(pid))
        if row is None or not row.s3_uri:
            raise RunwareError(f"Photo {pid} has no stored image.")
        key = row.s3_uri[5:].split("/", 1)[1] if row.s3_uri.startswith("s3://") else row.s3_uri
        url = s3_client.generate_presigned_url(
            key, expires_in=settings.RUNWARE_SUBMIT_URL_EXPIRY_SECONDS)
        if not url:
            raise RunwareError(f"Could not sign photo {pid}.")
        urls.append(url)
    return urls


def pending_shots(timeline: Timeline) -> list[TimelineClip]:
    """Shots with no usable video yet: never generated, or generated and failed."""
    return [c for c in timeline.ordered_clips() if c.state in {"planned", "failed"}]


def _prepare(db: Session, timeline: Timeline, shot: TimelineClip) -> Clip:
    """Create or reuse the Clip row, with its task id, before any network call."""
    if shot.clip_id and shot.clip and shot.clip.provider_task_uuid \
            and (shot.clip.status or "") not in {"failed"}:
        return shot.clip

    squeeze = SOURCE_SECONDS / max(0.1, shot.duration_for(timeline.grid_seconds))
    ordered = timeline.ordered_clips()
    prompt = build_shot_prompt(
        shot, squeeze_factor=squeeze,
        is_last=bool(ordered) and shot.shot_uid == ordered[-1].shot_uid)

    clip = shot.clip if shot.clip_id else None
    if clip is None:
        clip = Clip(job_id=timeline.job_id)
        db.add(clip)
        db.flush()
        shot.clip_id = clip.id

    clip.shot_uid = shot.shot_uid
    clip.provider = "runware"
    clip.provider_task_uuid = str(uuid.uuid4())
    clip.model_used = settings.RUNWARE_MODEL
    clip.prompt_used = prompt
    clip.duration = float(settings.RUNWARE_DURATION_SECONDS)
    clip.source_photo_ids = list(shot.photo_ids or [])
    clip.motion_type = shot.move_label
    clip.status = "rendering"
    db.commit()
    return clip


async def _run_one(client: httpx.AsyncClient, semaphore: asyncio.Semaphore,
                   *, prompt: str, urls: list[str], task_uuid: str, on_tick):
    async with semaphore:
        return await generate_video(client, prompt=prompt, photo_urls=urls,
                                    task_uuid=task_uuid, on_tick=on_tick)


async def _generate_all(jobs: list[dict], on_tick) -> dict[str, object]:
    semaphore = asyncio.Semaphore(max(1, settings.RUNWARE_MAX_CONCURRENCY))
    async with httpx.AsyncClient(timeout=60.0) as client:
        results = await asyncio.gather(*[
            _run_one(client, semaphore, prompt=j["prompt"], urls=j["urls"],
                     task_uuid=j["task_uuid"], on_tick=on_tick)
            for j in jobs
        ], return_exceptions=True)
    return {j["task_uuid"]: r for j, r in zip(jobs, results)}


def generate_timeline_clips(db: Session, timeline: Timeline, *,
                            progress=None, shot_uids: list[str] | None = None) -> dict:
    """Generate every pending shot. Returns a summary of what happened."""
    progress = progress or NullProgress()

    shots = pending_shots(timeline)
    if shot_uids:
        wanted = set(shot_uids)
        shots = [s for s in shots if s.shot_uid in wanted]
    if not shots:
        return {"requested": 0, "succeeded": 0, "failed": 0}

    progress.start_phase("Generating clips", len(shots))

    jobs, by_task = [], {}
    for shot in shots:
        progress.raise_if_cancelled()
        try:
            clip = _prepare(db, timeline, shot)
            jobs.append({"prompt": clip.prompt_used,
                         "urls": _photo_urls(db, list(shot.photo_ids or [])),
                         "task_uuid": clip.provider_task_uuid})
            by_task[clip.provider_task_uuid] = clip
        except RunwareError as exc:
            logger.warning("SHOT_PREPARE_FAILED shot=%s error=%s", shot.shot_uid, exc)
            if shot.clip:
                shot.clip.status = "failed"
                shot.clip.render_metadata = {"error": str(exc)}
                db.commit()

    if not jobs:
        return {"requested": len(shots), "succeeded": 0, "failed": len(shots)}

    logger.info("RUNWARE_BATCH_START count=%s concurrency=%s resolution=%s",
                len(jobs), settings.RUNWARE_MAX_CONCURRENCY, settings.RUNWARE_RESOLUTION)
    results = asyncio.run(_generate_all(jobs, progress.heartbeat))

    done = failed = 0
    for task_uuid, result in results.items():
        clip = by_task[task_uuid]
        if isinstance(result, Exception) or not getattr(result, "ok", False):
            reason = str(result) if isinstance(result, Exception) else (result.error or "unknown")
            clip.status = "failed"
            clip.render_metadata = {**(clip.render_metadata or {}), "error": reason}
            failed += 1
            logger.warning("SHOT_FAILED task=%s error=%s", task_uuid, reason)
        else:
            try:
                # Runware's URLs expire; the clip has to live in our bucket.
                data = httpx.get(result.video_url, timeout=120.0).content
                key = f"jobs/{timeline.job_id}/clips/{clip.shot_uid}.mp4"
                clip.s3_uri = s3_client.upload_bytes(key, data, "video/mp4")
                clip.status = "complete"
                clip.seed = result.seed
                done += 1
            except Exception as exc:               # noqa: BLE001 - store and move on
                clip.status = "failed"
                clip.render_metadata = {**(clip.render_metadata or {}),
                                        "error": f"download failed: {exc}"}
                failed += 1
        db.commit()
        progress.update(done + failed, len(jobs), "Generating clips")

    logger.info("RUNWARE_BATCH_DONE succeeded=%s failed=%s", done, failed)
    return {"requested": len(shots), "succeeded": done, "failed": failed}
