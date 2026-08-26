"""Schemas for human-labeled scene ground truth.

Photo references are `rails_photo_id` strings ("photo keys"), never job_photo
ids, so labels survive analysis re-runs.
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class RoomInstancePayload(BaseModel):
    instance: str
    photo_keys: List[str] = Field(default_factory=list)


class SceneTruthSetPayload(BaseModel):
    listing_slug: str = ""
    split: str = "calibration"
    status: str = "draft"
    room_instances: List[RoomInstancePayload] = Field(default_factory=list)
    open_plan_groups: List[List[str]] = Field(default_factory=list)
    duplicates: List[List[str]] = Field(default_factory=list)
    must_not_group: List[List[str]] = Field(default_factory=list)
    preferred_pairs: List[List[str]] = Field(default_factory=list)
    story_bridges: List[List[str]] = Field(default_factory=list)
    notes: str = ""
    labeled_by: str = ""


class TruthPhoto(BaseModel):
    photo_key: str
    photo_id: int
    position: int
    filename: Optional[str] = None
    thumbnail_url: Optional[str] = None
    room_label: Optional[str] = None
    predicted_component_id: Optional[int] = None


class SceneTruthSetResponse(BaseModel):
    project_id: str
    job_id: Optional[int] = None
    listing_slug: str = ""
    split: str = "calibration"
    status: str = "draft"
    room_instances: List[RoomInstancePayload] = Field(default_factory=list)
    open_plan_groups: List[List[str]] = Field(default_factory=list)
    duplicates: List[List[str]] = Field(default_factory=list)
    must_not_group: List[List[str]] = Field(default_factory=list)
    preferred_pairs: List[List[str]] = Field(default_factory=list)
    story_bridges: List[List[str]] = Field(default_factory=list)
    notes: str = ""
    labeled_by: str = ""
    photos: List[TruthPhoto] = Field(default_factory=list)
    photo_count: int = 0
    labeled_count: int = 0
    predicted_component_count: int = 0
    warnings: List[str] = Field(default_factory=list)
    stale_photo_keys: List[str] = Field(default_factory=list)
    revision: int = 1
    updated_at: Optional[str] = None


class TruthSetSummary(BaseModel):
    project_id: str
    last_job_id: Optional[int] = None
    listing_slug: str = ""
    split: str = "calibration"
    status: str = "draft"
    photo_count: int = 0
    labeled_count: int = 0
    room_count: int = 0
    revision: int = 1
    updated_at: Optional[str] = None


class TruthSetListResponse(BaseModel):
    sets: List[TruthSetSummary] = Field(default_factory=list)
    total_labeled_photos: int = 0
    complete_count: int = 0
