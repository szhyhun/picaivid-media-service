"""Runware client for Vidu Q3 Turbo image-to-video generation.

Runware takes one HTTP endpoint and a list of task objects. Video inference is
asynchronous: you submit a task with a `taskUUID` you choose, then poll
`getResponse` with that same id until a `videoURL` appears.

Choosing the id ourselves is what makes retries safe. We write it to the `clips`
row *before* awaiting, so a worker that dies mid-generation resumes the existing
task on restart instead of submitting a second one and paying twice.

Two things about Vidu that cost money if missed:

* **Audio defaults to on.** We lay our own music on a beat grid and discard the
  generated track entirely, so leaving it on is paying for output that goes
  straight in the bin.
* **Duration is billed from a 1-second floor**, not a 5-second one. That is why
  this model is worth four times less per shot than the alternatives, and why
  we can afford to buy 3s of source for a 1.2s cut.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# 16:9 at each tier Vidu offers. Cost scales with pixels, so development runs
# at the bottom of this table and production at the top.
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
    seed: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.video_url)


def resolution_size(name: str | None = None) -> tuple[int, int]:
    key = (name or settings.RUNWARE_RESOLUTION or "540p").lower()
    if key not in RESOLUTIONS:
        raise RunwareError(
            f"Unknown resolution {key!r}; expected one of {sorted(RESOLUTIONS)}")
    return RESOLUTIONS[key]


def build_frame_images(photo_urls: list[str]) -> list[Any]:
    """Map our two shot kinds onto Vidu's `frameImages`.

    One photograph is a start frame and the model invents everything the camera
    reveals. Two photographs pin both ends, so it only has to invent the middle
    -- which is the whole reason the pairing geometry exists.
    """
    if not photo_urls:
        raise RunwareError("A shot needs at least one photograph.")
    if len(photo_urls) == 1:
        return [photo_urls[0]]
    return [photo_urls[0], {"image": photo_urls[1], "frame": "last"}]


def build_video_task(*, prompt: str, photo_urls: list[str],
                     task_uuid: str | None = None,
                     duration: int | None = None,
                     resolution: str | None = None,
                     model: str | None = None) -> dict[str, Any]:
    """The exact request body Runware expects. Pure, so it can be tested."""
    width, height = resolution_size(resolution)
    return {
        "taskType": "videoInference",
        "taskUUID": task_uuid or str(uuid.uuid4()),
        "model": model or settings.RUNWARE_MODEL,
        "positivePrompt": prompt,
        "width": width,
        "height": height,
        "duration": int(duration or settings.RUNWARE_DURATION_SECONDS),
        "outputType": "URL",
        "outputFormat": "MP4",
        "deliveryMethod": "async",
        # Never on. See the module docstring.
        "providerSettings": {"vidu": {"audio": False}},
        "inputs": {"frameImages": build_frame_images(photo_urls)},
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
                     on_tick=None) -> VideoResult:
    """Wait for one video. `on_tick` keeps the job heartbeat fresh while we wait."""
    interval = interval or settings.RUNWARE_POLL_INTERVAL_SECONDS
    attempts = max_attempts or settings.RUNWARE_MAX_POLL_ATTEMPTS

    for _ in range(attempts):
        payload = await _post(client, [{"taskType": "getResponse",
                                        "taskUUID": task_uuid}])
        row = _first_result(payload) or {}
        if row.get("videoURL"):
            return VideoResult(task_uuid=task_uuid, video_url=str(row["videoURL"]),
                               seed=row.get("seed"))
        message = row.get("message") or row.get("error")
        # A "still processing" message is not a failure; anything else is.
        if message and "process" not in str(message).lower():
            return VideoResult(task_uuid=task_uuid, video_url=None,
                               error=str(message))
        if on_tick:
            on_tick()
        await asyncio.sleep(interval)

    return VideoResult(task_uuid=task_uuid, video_url=None,
                       error=f"Timed out after {attempts} polls.")


async def generate_video(client: httpx.AsyncClient, *, prompt: str,
                         photo_urls: list[str], task_uuid: str,
                         on_tick=None, **kwargs) -> VideoResult:
    """Submit one shot and wait for it."""
    task = build_video_task(prompt=prompt, photo_urls=photo_urls,
                            task_uuid=task_uuid, **kwargs)
    await submit_video(client, task)
    return await poll_video(client, task_uuid, on_tick=on_tick)
