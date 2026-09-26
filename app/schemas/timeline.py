"""Studio timeline payloads.

`in_time`, `duration` and `state` are all **derived** server-side and sent as
read-only facts. The client never posts timing: it posts order, trims and
overrides, and gets the resulting schedule back. Letting a browser own the
clock is how edits drift out of sync with the beat grid they are supposed to
sit on.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

ShotState = Literal["planned", "generating", "ready", "failed"]
ShotKind = Literal["pair_keyframes", "single_image"]


class TimelinePhoto(BaseModel):
    id: int
    thumbnail_url: Optional[str] = None
    position: Optional[int] = None
    room_label: Optional[str] = None


class TimelineShot(BaseModel):
    shot_uid: str
    order_index: int
    state: ShotState

    room_id: Optional[str] = None
    room_type: Optional[str] = None
    act: Optional[int] = None
    kind: Optional[ShotKind] = None
    is_close_up: bool = False

    # Timing. All derived; `max_duration` is what the generated source allows,
    # and the UI should stop a trim handle there rather than fail on save.
    in_time: float = 0.0
    duration: float = 0.0
    trim_start: float = 0.0
    max_duration: float = 5.0
    duration_override: Optional[float] = None

    move: Optional[str] = None
    move_vector: Optional[dict[str, float]] = None
    crop_box: Optional[list[int]] = None
    subject_box: Optional[list[int]] = None
    source_size: Optional[list[int]] = None
    detail_subject: Optional[str] = None
    prompt_override: Optional[str] = None
    # The customer's optional note, appended to our prompt -- never replacing it.
    prompt_note: Optional[str] = None
    # True once a customer composed this move, which keeps the positional
    # default from overruling it.
    move_locked: bool = False
    # Exact prompt previewed before generation or sent for a ready clip.
    generation_prompt: Optional[str] = None

    video_url: Optional[str] = None
    # Why this shot failed, in the provider's own words. A bare "failed" makes
    # every retry a guess: a timeout and a rejected key need different actions.
    error: Optional[str] = None
    # What this shot has actually cost so far, when the provider reports it.
    cost_usd: Optional[float] = None
    photos: list[TimelinePhoto] = []


MusicSource = Literal["catalog", "upload"]


class TimelineMusic(BaseModel):
    """Public music metadata; storage/provider URIs never leave the service."""

    source: MusicSource
    title: str
    bpm: float
    beat_offset: float = 0.0
    duration: Optional[float] = None


class TimelineMusicRequest(BaseModel):
    """A catalog track or an already-uploaded customer track."""

    music_uri: str = Field(min_length=1, max_length=1000)
    source: MusicSource = "catalog"
    title: str = Field(default="Custom track", min_length=1, max_length=255)
    bpm: Optional[float] = Field(default=None, gt=20, lt=300)
    beat_offset: Optional[float] = Field(default=None, ge=0)
    duration: Optional[float] = Field(default=None, gt=0)


class TimelineResponse(BaseModel):
    project_id: str
    timeline_id: Optional[int] = None
    analysis_status: Optional[str] = None
    analysis_ready: bool = False
    version: int = 0
    status: str = "draft"
    bpm: float = 100.0
    beats_per_shot: int = 2
    shot_seconds: float = 1.2
    source_seconds: float = 5.0
    duration: float = 0.0
    shots: list[TimelineShot] = []
    music: Optional[TimelineMusic] = None
    detail: str = ""


class TimelineSummary(BaseModel):
    id: int
    version: int = 1
    status: str = "draft"
    name: str = ""
    bpm: float = 100.0
    duration: float = 0.0
    target_seconds: Optional[float] = None


class TimelineListResponse(BaseModel):
    project_id: str
    timelines: list[TimelineSummary] = []


class TimelineShotUpdate(BaseModel):
    """One shot as the editor wants it. Unknown uids are created as planned."""

    shot_uid: Optional[str] = None
    photo_ids: list[int] = []
    kind: ShotKind = "single_image"
    room_id: Optional[str] = None
    room_type: Optional[str] = None
    act: Optional[int] = None
    trim_start: float = 0.0
    duration_override: Optional[float] = None
    prompt_override: Optional[str] = None
    prompt_note: Optional[str] = None
    # The composed move, e.g. "push in + pan right + tilt up". Locks the shot
    # only when it differs from the move the shot would have generated anyway.
    move: Optional[str] = None


class TimelineUpdateRequest(BaseModel):
    """Full replacement of the shot list, in the order given.

    Sending the whole list rather than a patch keeps reordering, insertion and
    deletion as one atomic operation -- a drag is all three at once, and a
    half-applied drag is worse than a rejected one.
    """

    shots: list[TimelineShotUpdate]
    bpm: Optional[float] = None
    beats_per_shot: Optional[int] = None


class CandidatePhoto(BaseModel):
    """A photo with no shot on the timeline yet."""

    id: int
    thumbnail_url: Optional[str] = None
    position: Optional[int] = None
    room_label: Optional[str] = None
    room_id: Optional[str] = None
    # The verified partner this photo could anchor a pair shot with, if any.
    # Pair shots hold both ends of the generated move to real photographs, so
    # offering this at the point of use is the geometry work paying off.
    pair_with: Optional[int] = None


class CandidateListResponse(BaseModel):
    project_id: str
    photos: list[CandidatePhoto] = []
