"""Runware client for Luma Ray3.2 image-to-video generation.

Runware takes one HTTP endpoint and a list of task objects. Video inference is
asynchronous: you submit a task with a `taskUUID` you choose, then poll
`getResponse` with that same id until a `videoURL` appears.

Choosing the id ourselves is what makes retries safe. We write it to the `clips`
row *before* awaiting, so a worker that dies mid-generation resumes the existing
task on restart instead of submitting a second one and paying twice.

Ray3.2's frame-to-frame mode is deliberately validated here instead of relying
on an environment value being correct: it requires a 5-second duration, and
HDR is available only at 720p or 1080p. A malformed combination must fail
before a paid request is sent.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# 16:9 presets accepted by Runware. Cost scales with pixels, so the endpoint
# validates the selected tier before a paid request.
RESOLUTIONS: dict[str, tuple[int, int]] = {
    "540p": (960, 540),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}

# Transient by Runware's own documentation: the queue is dynamic rather than
# rate-limited, so these mean "busy", not "wrong".
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
BACKOFF_SECONDS = (1.0, 2.0, 4.0)


class RunwareError(RuntimeError):
    """A generation failed in a way retrying will not fix."""


@dataclass(frozen=True)
class VideoResult:
    task_uuid: str
    video_url: str | None
    video_uuid: str | None = None
    seed: int | None = None
    cost: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.video_url)


@dataclass(frozen=True)
class VideoOptions:
    """Validated options for the one supported Runware video model."""

    model: str
    duration: int
    resolution: str
    hdr: bool
    exr_export: bool
    output_quality: int


RAY_MODEL_ID = "luma:ray@3.2"
RAY_FRAME_DURATION_SECONDS = 5


def allowed_durations() -> frozenset[int]:
    """The only duration this deployment is allowed to buy."""
    return frozenset({RAY_FRAME_DURATION_SECONDS})


def duration_for_kind(kind: str | None) -> int:
    """Seconds of source video to buy for any shot kind."""
    duration = int(settings.RUNWARE_DURATION_SECONDS)
    if duration != RAY_FRAME_DURATION_SECONDS:
        raise RunwareError(
            f"Ray3.2 generation must use duration={RAY_FRAME_DURATION_SECONDS}; "
            f"got {duration}.")
    return duration
HDR_RESOLUTIONS = frozenset({"720p", "1080p"})


def resolution_size(name: str | None = None) -> tuple[int, int]:
    key = (name or settings.RUNWARE_RESOLUTION or "720p").lower()
    if key not in RESOLUTIONS:
        raise RunwareError(
            f"Unknown resolution {key!r}; expected one of {sorted(RESOLUTIONS)}")
    return RESOLUTIONS[key]


def runware_model_id(model: str | None = None) -> str:
    """Return the configured model, refusing silent fallback to another model."""
    configured = (model or settings.RUNWARE_MODEL or "").strip()
    if configured != RAY_MODEL_ID:
        raise RunwareError(
            f"Runware model must be {RAY_MODEL_ID!r}; got {configured or '<empty>'!r}")
    return configured


def resolve_video_options(*, model: str | None = None,
                          duration: int | None = None,
                          resolution: str | None = None,
                          hdr: bool | None = None,
                          exr_export: bool | None = None,
                          output_quality: int | None = None) -> VideoOptions:
    """Resolve and validate the exact Ray3.2 options before submission."""
    model_id = runware_model_id(model)
    resolution_name = (resolution or settings.RUNWARE_RESOLUTION or "720p").lower()
    resolution_size(resolution_name)
    duration_seconds = int(
        settings.RUNWARE_DURATION_SECONDS if duration is None else duration)
    hdr_enabled = settings.RUNWARE_HDR if hdr is None else bool(hdr)
    exr_enabled = (
        settings.RUNWARE_EXR_EXPORT if exr_export is None else bool(exr_export))
    quality = int(
        settings.RUNWARE_OUTPUT_QUALITY
        if output_quality is None else output_quality)

    if duration_seconds not in allowed_durations():
        raise RunwareError(
            f"Ray3.2 generation must use duration={RAY_FRAME_DURATION_SECONDS}; "
            f"got {duration_seconds}.")
    if not hdr_enabled:
        raise RunwareError("This deployment requires HDR to be enabled for Ray3.2.")
    if hdr_enabled and resolution_name not in HDR_RESOLUTIONS:
        raise RunwareError(
            "Ray3.2 HDR requires resolution=720p or resolution=1080p.")
    if exr_enabled and not hdr_enabled:
        raise RunwareError("Ray3.2 exrExport requires HDR to be enabled.")
    if not 20 <= quality <= 99:
        raise RunwareError("Runware outputQuality must be between 20 and 99.")

    return VideoOptions(model_id, duration_seconds, resolution_name,
                        hdr_enabled, exr_enabled, quality)


def build_frame_images(photo_urls: list[str]) -> list[dict[str, str]]:
    """Map our one- and two-photo shots onto Ray3.2's `frameImages`.

    One photograph is a start frame and the model invents everything the camera
    reveals. Two photographs pin both ends, so it only has to invent the middle
    -- which is the whole reason the pairing geometry exists.
    """
    if not photo_urls:
        raise RunwareError("A shot needs at least one photograph.")
    if len(photo_urls) == 1:
        return [{"image": photo_urls[0], "frame": "first"}]
    if len(photo_urls) > 2:
        raise RunwareError("A Ray3.2 shot accepts at most two photographs.")
    return [
        {"image": photo_urls[0], "frame": "first"},
        {"image": photo_urls[1], "frame": "last"},
    ]


def build_video_task(*, prompt: str, photo_urls: list[str],
                     task_uuid: str | None = None,
                     duration: int | None = None,
                     resolution: str | None = None,
                     model: str | None = None,
                     hdr: bool | None = None,
                     exr_export: bool | None = None,
                     output_quality: int | None = None) -> dict[str, Any]:
    """The exact request body Runware expects. Pure, so it can be tested."""
    options = resolve_video_options(
        model=model,
        duration=duration,
        resolution=resolution,
        hdr=hdr,
        exr_export=exr_export,
        output_quality=output_quality,
    )
    return {
        "model": options.model,
        "inputs": {"frameImages": build_frame_images(photo_urls)},
        "positivePrompt": prompt,
        "duration": options.duration,
        "settings": {
            "hdr": options.hdr,
            "exrExport": options.exr_export,
        },
        "taskType": "videoInference",
        "numberResults": 1,
        "outputType": "URL",
        "outputFormat": "MP4",
        "outputQuality": options.output_quality,
        "deliveryMethod": "async",
        "includeCost": settings.RUNWARE_INCLUDE_COST,
        "resolution": options.resolution,
        "taskUUID": task_uuid or str(uuid.uuid4()),
    }


def _headers() -> dict[str, str]:
    if not settings.RUNWARE_API_KEY:
        raise RunwareError("RUNWARE_API_KEY is not set.")
    return {
        "Authorization": f"Bearer {settings.RUNWARE_API_KEY}",
        "Content-Type": "application/json",
    }


def _first_result(payload: Any) -> dict[str, Any] | None:
    """Runware answers with {"data": [...]}; errors arrive under "errors"."""
    if not isinstance(payload, dict):
        return None
    for key in ("data", "errors"):
        rows = payload.get(key)
        if isinstance(rows, list) and rows:
            return rows[0]
    return None


def _error_message(row: dict[str, Any]) -> str | None:
    """Normalise Runware's documented top-level and nested error shapes."""
    message = row.get("message")
    if message:
        return str(message)
    error = row.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or "Runware error")
    if error:
        return str(error)
    code = row.get("code")
    return str(code) if code else None


async def _post(client: httpx.AsyncClient, body: list[dict[str, Any]]) -> Any:
    """POST with backoff. Runware asks for this explicitly for resilience."""
    last: Exception | None = None
    for attempt, delay in enumerate((*BACKOFF_SECONDS, None)):
        try:
            response = await client.post(settings.RUNWARE_API_URL,
                                         json=body, headers=_headers())
            if response.status_code in RETRY_STATUS and delay is not None:
                logger.warning("RUNWARE_BUSY status=%s retry_in=%ss",
                               response.status_code, delay)
                await asyncio.sleep(delay)
                continue
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise RunwareError(
                f"Runware rejected the request ({exc.response.status_code}): "
                f"{exc.response.text[:300]}") from exc
        except httpx.HTTPError as exc:
            last = exc
            if delay is None:
                break
            await asyncio.sleep(delay)
    raise RunwareError(f"Runware unreachable: {last}")


async def submit_video(client: httpx.AsyncClient, task: dict[str, Any]) -> str:
    """Start a generation. Returns the taskUUID to poll."""
    payload = await _post(client, [task])
    row = _first_result(payload) or {}
    if row.get("error") or row.get("message") and not row.get("taskUUID"):
        raise RunwareError(str(row.get("message") or row.get("error")))
    return str(row.get("taskUUID") or task["taskUUID"])


async def poll_video(client: httpx.AsyncClient, task_uuid: str, *,
                     interval: float | None = None,
                     max_attempts: int | None = None,
                     max_wait_seconds: float | None = None,
                     on_tick=None) -> VideoResult:
    """Wait for one video without allowing a provider task to hang forever."""
    interval = (settings.RUNWARE_POLL_INTERVAL_SECONDS
                if interval is None else interval)
    attempts = (settings.RUNWARE_MAX_POLL_ATTEMPTS
                if max_attempts is None else max_attempts)
    max_wait = (settings.RUNWARE_MAX_WAIT_SECONDS
                if max_wait_seconds is None else max_wait_seconds)

    delay = max(1.0, float(interval))
    deadline = time.monotonic() + max(1.0, float(max_wait))
    polls = 0
    while polls < max(1, int(attempts)) and time.monotonic() < deadline:
        polls += 1
        payload = await _post(client, [{"taskType": "getResponse",
                                        "taskUUID": task_uuid}])
        row = _first_result(payload) or {}
        if row.get("videoURL"):
            cost = row.get("cost")
            return VideoResult(
                task_uuid=task_uuid,
                video_url=str(row["videoURL"]),
                video_uuid=str(row["videoUUID"]) if row.get("videoUUID") else None,
                seed=row.get("seed"),
                cost=float(cost) if cost is not None else None,
            )
        # getResponse reports terminal failures through `errors` with
        # status=error. Do not mistake a normal processing acknowledgement for
        # a failed generation merely because it has an informational message.
        if str(row.get("status") or "").lower() in {"error", "failed"}:
            return VideoResult(task_uuid=task_uuid, video_url=None,
                               error=_error_message(row) or "Runware rejected the task.")
        if on_tick:
            on_tick()
        remaining = deadline - time.monotonic()
        if polls >= max(1, int(attempts)) or remaining <= 0:
            break
        await asyncio.sleep(min(delay, remaining))
        delay = min(delay * 1.5, 20.0)

    return VideoResult(task_uuid=task_uuid, video_url=None,
                       error=(f"Timed out after {max_wait:g}s and {polls} polls."))


async def generate_video(client: httpx.AsyncClient, *, prompt: str,
                         photo_urls: list[str], task_uuid: str,
                         on_tick=None, claim=None, **kwargs) -> VideoResult:
    """Submit one shot and wait for it.

    `claim` is called immediately before the submit and must return True for
    the submit to happen. It is how "this clip has not been paid for yet" is
    established atomically; returning False means another worker already owns
    this task, so we poll its result instead of ordering a second one.

    The claim is taken *before* the network call and never after. A submit that
    is not preceded by a successful claim cannot happen -- that is the whole
    guarantee, and it is why this is the only function that submits.
    """
    if claim is not None and not claim():
        logger.info("RUNWARE_ALREADY_CLAIMED task=%s -- polling instead of submitting",
                    task_uuid)
        return await poll_video(client, task_uuid, on_tick=on_tick)

    task = build_video_task(prompt=prompt, photo_urls=photo_urls,
                            task_uuid=task_uuid, **kwargs)
    await submit_video(client, task)
    return await poll_video(client, task_uuid, on_tick=on_tick)
