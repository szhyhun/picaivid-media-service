"""Job progress and cancellation payloads.

`state` is the single field the UI should branch on. It is derived here rather
than in the client so that "stalled" -- a running job whose worker died -- is
computed once, from the heartbeat, instead of every consumer inventing its own
timeout. Before this existed, a crashed worker and a slow one were
indistinguishable and both rendered as an endless spinner.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel

JobState = Literal[
    "idle",       # no job has ever run for this project
    "queued",     # job row exists, worker has not started it
    "running",    # actively working, heartbeat fresh
    "stalled",    # claims to be running but the heartbeat is stale
    "complete",
    "failed",
    "cancelled",
]


class JobProgressResponse(BaseModel):
    project_id: str
    job_id: Optional[int] = None
    state: JobState = "idle"
    status: Optional[str] = None
    phase: Optional[int] = None
    phase_label: Optional[str] = None

    current: Optional[int] = None
    total: Optional[int] = None
    percent: Optional[float] = None

    elapsed_seconds: Optional[float] = None
    seconds_since_heartbeat: Optional[float] = None
    stale_after_seconds: int = 90

    cancel_requested: bool = False
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None


class CancelResponse(BaseModel):
    project_id: str
    job_id: Optional[int] = None
    cancelled: bool
    detail: str
