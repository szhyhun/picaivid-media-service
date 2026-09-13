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
import base64
import logging
import mimetypes
import uuid
from urllib.parse import urlparse

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


def _uses_inline_media() -> bool:
    """Whether Runware must receive the image bytes instead of fetching a URL."""
    mode = (settings.RUNWARE_INPUT_MODE or "auto").strip().lower()
    if mode == "data_uri":
        return True
    if mode == "url":
        return False
    if mode != "auto":
        raise RunwareError("RUNWARE_INPUT_MODE must be auto, url, or data_uri.")

    endpoint = settings.S3_ENDPOINT or ""
    if not endpoint:
        # Production AWS S3 has no custom endpoint. Its presigned URLs are
        # internet-reachable by design, so no base64 fallback is necessary.
        return False
    host = urlparse(endpoint).hostname or ""
    # A Docker service name and localhost variants are resolvable only inside
    # this machine's network. Runware needs bytes in the request in that case.
    return host in {"localhost", "127.0.0.1", "::1", "minio"} or "." not in host


def _photo_inputs(db: Session, photo_ids: list[int]) -> list[str]:
    """Provider-readable source images, valid long enough to survive a queue.

    The UI's one-hour URLs are too short here: a submitted task can sit in
    Runware's queue for minutes before it fetches anything. Local MinIO is not
    internet-reachable, so local runs transmit a documented data URI instead.
    """
    rows = {int(p.id): p for p in
            db.query(JobPhoto).filter(JobPhoto.id.in_(photo_ids)).all()}
    inputs = []
    inline = _uses_inline_media()
    for pid in photo_ids:
        row = rows.get(int(pid))
        if row is None or not row.s3_uri:
            raise RunwareError(f"Photo {pid} has no stored image.")
        if inline:
            mime_type = mimetypes.guess_type(row.filename or "")[0] or "image/jpeg"
            encoded = base64.b64encode(s3_client.download_bytes(row.s3_uri)).decode("ascii")
            inputs.append(f"data:{mime_type};base64,{encoded}")
            continue
        key = row.s3_uri[5:].split("/", 1)[1] if row.s3_uri.startswith("s3://") else row.s3_uri
        url = s3_client.generate_presigned_url(
            key, expires_in=settings.RUNWARE_SUBMIT_URL_EXPIRY_SECONDS)
        if not url:
            raise RunwareError(f"Could not sign photo {pid}.")
        inputs.append(url)
    return inputs


def pending_shots(timeline: Timeline) -> list[TimelineClip]:
    """Shots with no usable video yet: never generated, or generated and failed."""
    return [c for c in timeline.ordered_clips() if c.state in {"planned", "failed"}]


def queued_shots(timeline: Timeline) -> list[TimelineClip]:
    """Shots persisted by the API and ready for the background generator."""
    return [
        c for c in timeline.ordered_clips()
        if c.clip is not None and (c.clip.status or "").lower() == "queued"
    ]


def _prepare(db: Session, timeline: Timeline, shot: TimelineClip) -> Clip:
    """Create or reuse the Clip row, with its task id, before any network call."""
    if shot.clip_id and shot.clip and shot.clip.provider_task_uuid \
            and (shot.clip.status or "") not in {"failed"}:
        return shot.clip

    ordered = timeline.ordered_clips()
    first_uid = ordered[0].shot_uid if ordered else None
    last_uid = ordered[-1].shot_uid if ordered else None
    prompt = build_shot_prompt(
        shot, squeeze_factor=1.0,
        is_first=bool(first_uid) and shot.shot_uid == first_uid,
        is_last=bool(last_uid) and shot.shot_uid == last_uid)

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
    # Queued is visible to the studio before the HTTP response returns. The
    # background runner turns it into an active Runware task immediately after.
    clip.status = "queued"
    db.commit()
    return clip


async def _run_one(client: httpx.AsyncClient, semaphore: asyncio.Semaphore,
                   *, prompt: str, inputs: list[str], task_uuid: str,
                   resolution: str | None, on_tick):
    async with semaphore:
        return await generate_video(client, prompt=prompt, photo_urls=inputs,
                                    task_uuid=task_uuid, resolution=resolution,
                                    on_tick=on_tick)


async def _generate_all(jobs: list[dict], resolution: str | None, on_tick) -> dict[str, object]:
    semaphore = asyncio.Semaphore(max(1, settings.RUNWARE_MAX_CONCURRENCY))
    async with httpx.AsyncClient(timeout=60.0) as client:
        results = await asyncio.gather(*[
            _run_one(client, semaphore, prompt=j["prompt"], inputs=j["inputs"],
                     task_uuid=j["task_uuid"], resolution=resolution, on_tick=on_tick)
            for j in jobs
        ], return_exceptions=True)
    return {j["task_uuid"]: r for j, r in zip(jobs, results)}


def queue_timeline_clips(db: Session, timeline: Timeline, *,
                         shot_uids: list[str] | None = None,
                         resolution: str | None = None) -> list[str]:
    """Persist generation intent before the browser receives success.

    A background task does the network work, but the timeline must immediately
    show that its shots are in flight. It also means a second click cannot order
    the same generation while the first request is still being scheduled.
    """
    wanted = set(shot_uids or [])
    shots = [
        shot for shot in pending_shots(timeline)
        if not wanted or shot.shot_uid in wanted
    ]
    for shot in shots:
        clip = _prepare(db, timeline, shot)
        clip.render_metadata = {
            **(clip.render_metadata or {}),
            "resolution": resolution or settings.RUNWARE_RESOLUTION,
        }
        clip.status = "queued"
    if shots:
        db.commit()
    return [str(shot.shot_uid) for shot in shots]


def generate_timeline_clips(db: Session, timeline: Timeline, *,
                            progress=None, shot_uids: list[str] | None = None,
                            resolution: str | None = None) -> dict:
    """Generate persisted queued shots. Returns a summary of what happened."""
    progress = progress or NullProgress()

    shots = queued_shots(timeline)
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
                         "inputs": _photo_inputs(db, list(shot.photo_ids or [])),
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
                len(jobs), settings.RUNWARE_MAX_CONCURRENCY,
                resolution or settings.RUNWARE_RESOLUTION)
    results = asyncio.run(_generate_all(jobs, resolution, progress.heartbeat))

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
                clip.render_metadata = {
                    **(clip.render_metadata or {}),
                    "provider_video_uuid": result.video_uuid,
                    "provider_cost_usd": result.cost,
                    "resolution": resolution or settings.RUNWARE_RESOLUTION,
                }
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
