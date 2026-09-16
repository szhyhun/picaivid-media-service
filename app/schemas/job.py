"""Pydantic schemas for job messages from SQS."""
from typing import Optional
from pydantic import BaseModel, Field


class JobMessage(BaseModel):
    """Job message received from SQS (sent by Rails).

    Rails sends project_id. Python reads photos from Rails DB.
    """
    action: str = "run"  # "run", "timeline_generate", or "timeline_assemble"
    project_id: str  # UUID from Rails projects table
    template_type: Optional[str] = None
    target_length: Optional[float] = None  # seconds
    music_uri: Optional[str] = None
    enable_beat_sync: bool = True

    # Timeline commands are dispatched by Rails after the customer edits the
    # authenticated timeline. They contain no provider credentials.
    shot_uids: list[str] = Field(default_factory=list)
    resolution: Optional[str] = None

    # Optional: resume from specific phase
    start_phase: Optional[int] = None


class JobStatusResponse(BaseModel):
    """Job status response for API."""
    job_id: int
    project_id: str
    status: str
    current_phase: int
    error_message: Optional[str] = None
