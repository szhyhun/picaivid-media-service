"""Pydantic schemas for job messages from SQS."""
from typing import Optional
from pydantic import BaseModel


class JobMessage(BaseModel):
    """Job message received from SQS (sent by Rails).

    Rails sends project_id. Python reads photos from Rails DB.
    """
    action: str = "run"  # "run" or "cancel"
    project_id: str  # UUID from Rails projects table
    template_type: Optional[str] = None
    target_length: Optional[float] = None  # seconds
    music_uri: Optional[str] = None
    enable_beat_sync: bool = True

    # Optional: resume from specific phase
    start_phase: Optional[int] = None


class JobStatusResponse(BaseModel):
    """Job status response for API."""
    job_id: int
    project_id: str
    status: str
    current_phase: int
    error_message: Optional[str] = None
