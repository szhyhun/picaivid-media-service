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
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Clip, JobPhoto
from app.db.models.timeline import SOURCE_SECONDS, Timeline, TimelineClip
from app.pipeline.phase1_analyze.directorial_prompt import build_shot_prompt
from app.pipeline.progress import NullProgress
from app.services.runware.billing_guard import already_submitted, claim_for_submit
from app.services.runware.client import (
    RunwareError,
    duration_for_kind,
    generate_video,
    poll_video,
    resolve_video_options,
)
from app.services.storage.s3_client import s3_client

logger = logging.getLogger(__name__)


class BatchDeadlineExceeded(RunwareError):
    """The whole run ran out of time, not just one shot."""


@dataclass(frozen=True)
class ShotOutcome:
    """What one shot produced, and where it was stored.

    Storage happens inside the task rather than in a pass afterwards, so the
    result carries its own bucket URI by the time the batch is collected.
    """

    result: object
    s3_uri: str | None = None
    store_error: str | None = None



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


def _download_video(url: str) -> bytes:
    """Download one provider result and reject HTTP or empty responses."""
    response = httpx.get(url, timeout=120.0)
    response.raise_for_status()
    if not response.content:
        raise RunwareError("Runware returned an empty video file.")
    return response.content


def pending_shots(timeline: Timeline) -> list[TimelineClip]:
    """Shots with no usable video yet: never generated, or generated and failed."""
    return [c for c in timeline.ordered_clips() if c.state in {"planned", "failed"}]


def queued_shots(timeline: Timeline) -> list[TimelineClip]:
    """Shots persisted by the API and ready for the background generator."""
    return [
        c for c in timeline.ordered_clips()
        if c.clip is not None and (c.clip.status or "").lower() == "queued"
    ]


def _clip_matches_request(clip: Clip, shot: TimelineClip, *, prompt: str,
                          options) -> bool:
    """Return whether a completed clip is byte-for-byte reusable for a shot.

    A generated result is reusable only when the source photos, exact prompt,
    model, and paid output settings all match. Position-dependent prompt rules
    are included in ``prompt``, so moving a shot to a different editorial role
    will correctly require a new generation.
    """
    metadata = clip.render_metadata or {}
    return (
        clip.status == "complete"
        and bool(clip.s3_uri)
        and clip.model_used == options.model
        and clip.prompt_used == prompt
        and list(clip.source_photo_ids or []) == list(shot.photo_ids or [])
        and abs(float(clip.duration or 0.0) - float(options.duration)) < 1e-6
        and metadata.get("resolution") == options.resolution
        and bool(metadata.get("hdr")) == bool(options.hdr)
        and bool(metadata.get("exr_export")) == bool(options.exr_export)
        and int(metadata.get("output_quality", options.output_quality))
        == int(options.output_quality)
    )


def _find_reusable_clip(db: Session, timeline: Timeline, shot: TimelineClip,
                        *, prompt: str, options) -> Clip | None:
    """Find an identical completed result from another video cut."""
    candidates = (
        db.query(Clip)
        .filter(Clip.job_id == timeline.job_id,
                Clip.status == "complete",
                Clip.model_used == options.model)
        .order_by(Clip.id.asc())
        .all()
    )
    return next(
        (candidate for candidate in candidates
         if _clip_matches_request(candidate, shot, prompt=prompt, options=options)),
        None,
    )


def _prepare(db: Session, timeline: Timeline, shot: TimelineClip, *,
             resolution: str | None = None) -> Clip:
    """Create or reuse the Clip row, with its task id, before any network call."""
    options = resolve_video_options(resolution=resolution)
    if shot.clip_id and shot.clip and shot.clip.provider_task_uuid \
            and (shot.clip.status or "") not in {"failed"}:
        if shot.clip.model_used != options.model:
            raise RunwareError(
                f"Cannot resume shot {shot.shot_uid} with model "
                f"{shot.clip.model_used or '<unknown>'}; expected {options.model}.")
        return shot.clip

    ordered = timeline.ordered_clips()
    first_uid = ordered[0].shot_uid if ordered else None
    last_uid = ordered[-1].shot_uid if ordered else None
    paired_room_ids = {
        candidate.room_id for candidate in ordered
        if candidate.room_id and candidate.kind == "pair_keyframes"
    }
    prompt = build_shot_prompt(
        shot, squeeze_factor=1.0,
        is_first=bool(first_uid) and shot.shot_uid == first_uid,
        is_last=bool(last_uid) and shot.shot_uid == last_uid,
        room_has_pair=shot.room_id in paired_room_ids)

    reusable = _find_reusable_clip(
        db, timeline, shot, prompt=prompt, options=options)
    if reusable is not None:
        shot.clip_id = reusable.id
        db.flush()
        logger.info("RUNWARE_REUSE clip=%s shot=%s source_shot=%s",
                    reusable.id, shot.shot_uid, reusable.shot_uid)
        return reusable

    clip = shot.clip if shot.clip_id else None
    if clip is None:
        clip = Clip(job_id=timeline.job_id)
        db.add(clip)
        db.flush()
        shot.clip_id = clip.id

    clip.shot_uid = shot.shot_uid
    clip.provider = "runware"
    # A fresh task id is a fresh purchase, so the submission mark has to go
    # with it. This is the one place a clip becomes buyable again, and it is
    # only reached for a shot that is planned or has already failed -- never
    # for one still in flight.
    clip.provider_task_uuid = str(uuid.uuid4())
    clip.provider_submitted_at = None
    clip.model_used = options.model
    clip.prompt_used = prompt
    clip.duration = float(duration_for_kind(shot.kind))
    clip.source_photo_ids = list(shot.photo_ids or [])
    clip.motion_type = shot.move_label
    # Queued is visible to the studio before the HTTP response returns. The
    # background runner turns it into an active Runware task immediately after.
    clip.status = "queued"
    db.commit()
    return clip


async def _run_one(client: httpx.AsyncClient, semaphore: asyncio.Semaphore,
                   *, prompt: str, inputs: list[str], task_uuid: str,
                   resolution: str | None, duration: int, clip_id: int,
                   s3_key: str, resume: bool, on_tick) -> ShotOutcome:
    async with semaphore:
        if resume:
            # This clip carries a submission timestamp, so it has already been
            # charged for. Collect it; never buy it twice.
            logger.info("RUNWARE_RESUME task=%s -- already submitted, polling only",
                        task_uuid)
            result = await poll_video(client, task_uuid, on_tick=on_tick)
        else:
            result = await generate_video(
                client, prompt=prompt, photo_urls=inputs, task_uuid=task_uuid,
                resolution=resolution, duration=duration, on_tick=on_tick,
                claim=lambda: claim_for_submit(clip_id))

        if not getattr(result, "ok", False):
            return ShotOutcome(result=result)

        # Fetch and store here, while the URL is minutes old rather than hours.
        # Collecting every clip in a pass after the batch meant the first shot's
        # provider URL sat unused for the length of the entire run, and a
        # provider URL that expires before we reach it loses a clip we paid for.
        #
        # Both calls are blocking (httpx.get, then boto3), so they go to a
        # worker thread: running them on the event loop would stall every other
        # shot's polling for the length of a download.
        try:
            data = await asyncio.to_thread(_download_video, result.video_url)
            uri = await asyncio.to_thread(
                s3_client.upload_bytes, s3_key, data, "video/mp4")
            return ShotOutcome(result=result, s3_uri=uri)
        except Exception as exc:   # noqa: BLE001 - reported per shot, never fatal
            logger.warning("SHOT_STORE_FAILED task=%s error=%s", task_uuid, exc)
            return ShotOutcome(result=result, store_error=str(exc))


async def _generate_all(jobs: list[dict], resolution: str | None, on_tick,
                        *, deadline_seconds: float | None = None) -> dict[str, object]:
    semaphore = asyncio.Semaphore(max(1, settings.RUNWARE_MAX_CONCURRENCY))
    async with httpx.AsyncClient(timeout=60.0) as client:
        pending = asyncio.gather(*[
            _run_one(client, semaphore, prompt=j["prompt"], inputs=j["inputs"],
                     task_uuid=j["task_uuid"], resolution=resolution,
                     duration=j["duration"], clip_id=j["clip_id"],
                     s3_key=j["s3_key"], resume=j["resume"], on_tick=on_tick)
            for j in jobs
        ], return_exceptions=True)
        if not deadline_seconds:
            results = await pending
        else:
            try:
                # A per-shot cap is not a cap on the run: enough slow shots in
                # sequence keep a worker (and its SQS message) alive for hours.
                results = await asyncio.wait_for(pending, timeout=deadline_seconds)
            except asyncio.TimeoutError:
                pending.cancel()
                results = [BatchDeadlineExceeded(
                    f"The run exceeded its {deadline_seconds:g}s budget.")
                    for _ in jobs]
    return {j["task_uuid"]: r for j, r in zip(jobs, results)}


def queue_timeline_clips(db: Session, timeline: Timeline, *,
                         shot_uids: list[str] | None = None,
                         resolution: str | None = None) -> list[str]:
    """Persist generation intent before the browser receives success.

    A background task does the network work, but the timeline must immediately
    show that its shots are in flight. It also means a second click cannot order
    the same generation while the first request is still being scheduled.
    """
    options = resolve_video_options(resolution=resolution)
    wanted = set(shot_uids or [])
    shots = [
        shot for shot in pending_shots(timeline)
        if not wanted or shot.shot_uid in wanted
    ]
    queued_uids = []
    for shot in shots:
        clip = _prepare(db, timeline, shot, resolution=resolution)
        # A matching result may already be complete and shared by another
        # studio. Never move that shared Clip back to queued; doing so would
        # make the other studio look like it needs generation again.
        if clip.status == "complete" and clip.s3_uri:
            continue
        clip.render_metadata = {
            **(clip.render_metadata or {}),
            "model": options.model,
            "duration": options.duration,
            "resolution": options.resolution,
            "hdr": options.hdr,
            "exr_export": options.exr_export,
            "output_quality": options.output_quality,
        }
        clip.status = "queued"
        queued_uids.append(str(shot.shot_uid))
    if shots:
        db.commit()
    return queued_uids


def generate_timeline_clips(db: Session, timeline: Timeline, *,
                            progress=None, shot_uids: list[str] | None = None,
                            resolution: str | None = None) -> dict:
    """Generate persisted queued shots. Returns a summary of what happened."""
    progress = progress or NullProgress()
    options = resolve_video_options(resolution=resolution)

    shots = queued_shots(timeline)
    if shot_uids:
        wanted = set(shot_uids)
        shots = [s for s in shots if s.shot_uid in wanted]
    if not shots:
        return {"requested": 0, "succeeded": 0, "failed": 0}

    total_requested = len(shots)
    progress.start_phase("Generating clips", total_requested)

    jobs, by_task = [], {}
    failed = 0
    for shot in shots:
        progress.raise_if_cancelled()
        try:
            clip = _prepare(db, timeline, shot, resolution=resolution)
            jobs.append({"prompt": clip.prompt_used,
                         "inputs": _photo_inputs(db, list(shot.photo_ids or [])),
                         "task_uuid": clip.provider_task_uuid,
                         "duration": duration_for_kind(shot.kind),
                         "clip_id": int(clip.id),
                         "s3_key": f"jobs/{timeline.job_id}/clips/{shot.shot_uid}.mp4",
                         # Already submitted means already billed. Such a clip
                         # is only ever collected, never ordered again.
                         "resume": already_submitted(clip)})
            by_task[clip.provider_task_uuid] = clip
        except RunwareError as exc:
            logger.warning("SHOT_PREPARE_FAILED shot=%s error=%s", shot.shot_uid, exc)
            if shot.clip:
                shot.clip.status = "failed"
                shot.clip.render_metadata = {
                    **(shot.clip.render_metadata or {}),
                    "error": f"input preparation failed: {exc}",
                }
                db.commit()
            failed += 1

    if not jobs:
        progress.update(failed, total_requested, "Generating clips")
        return {"requested": total_requested, "succeeded": 0, "failed": failed}

    logger.info("RUNWARE_BATCH_START model=%s count=%s concurrency=%s "
                "duration=%ss resolution=%s hdr=%s",
                options.model,
                len(jobs), settings.RUNWARE_MAX_CONCURRENCY,
                options.duration, options.resolution, options.hdr)
    try:
        results = asyncio.run(_generate_all(
            jobs, options.resolution, progress.heartbeat,
            deadline_seconds=settings.RUNWARE_BATCH_DEADLINE_SECONDS))
    except Exception as exc:  # noqa: BLE001 - persist every shot as failed
        reason = f"generation batch failed: {exc}"
        logger.exception("RUNWARE_BATCH_FAILED count=%s", len(jobs))
        for clip in by_task.values():
            clip.status = "failed"
            clip.render_metadata = {**(clip.render_metadata or {}), "error": reason}
        db.commit()
        failed += len(jobs)
        progress.update(failed, total_requested, "Generating clips")
        return {"requested": total_requested, "succeeded": 0, "failed": failed}

    done = 0
    # Nothing here touches the network any more: every clip was downloaded and
    # stored by its own task. This loop only writes what happened, which is why
    # it can safely hold the one database session.
    for task_uuid, outcome in results.items():
        clip = by_task[task_uuid]
        result = getattr(outcome, "result", None)
        store_error = getattr(outcome, "store_error", None)
        if isinstance(outcome, Exception) or not getattr(result, "ok", False):
            reason = (str(outcome) if isinstance(outcome, Exception)
                      else (getattr(result, "error", None) or "unknown"))
            clip.status = "failed"
            clip.render_metadata = {**(clip.render_metadata or {}), "error": reason}
            failed += 1
            logger.warning("SHOT_FAILED task=%s error=%s", task_uuid, reason)
        elif store_error:
            # The clip exists and was paid for; only our copy of it failed.
            clip.status = "failed"
            clip.render_metadata = {**(clip.render_metadata or {}),
                                    "error": f"result download failed: {store_error}"}
            failed += 1
        else:
            try:
                clip.s3_uri = outcome.s3_uri
                clip.status = "complete"
                clip.seed = result.seed
                clip.render_metadata = {
                    **(clip.render_metadata or {}),
                    "model": options.model,
                    "duration": options.duration,
                    "provider_video_uuid": result.video_uuid,
                    "provider_cost_usd": result.cost,
                    "resolution": options.resolution,
                    "hdr": options.hdr,
                    "exr_export": options.exr_export,
                    "output_quality": options.output_quality,
                }
                done += 1
            except Exception as exc:               # noqa: BLE001 - store and move on
                clip.status = "failed"
                clip.render_metadata = {**(clip.render_metadata or {}),
                                        "error": f"result download failed: {exc}"}
                failed += 1
        db.commit()
        progress.update(done + failed, total_requested, "Generating clips")

    logger.info("RUNWARE_BATCH_DONE succeeded=%s failed=%s", done, failed)
    return {"requested": total_requested, "succeeded": done, "failed": failed}
