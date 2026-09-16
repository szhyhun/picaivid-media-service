"""Commands executed by the private CPU media worker from Rails SQS jobs."""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.db.models import Job
from app.pipeline.phase3_timeline.generate import (
    generate_timeline_clips,
    pending_shots,
    queued_shots,
    queue_timeline_clips,
)
from app.pipeline.phase3_timeline.materialize import latest_timeline
from app.pipeline.phase4_assembly.render import assemble_timeline_film
from app.pipeline.progress import JobProgress
from app.services.rails_webhook import (
    notify_render_complete,
    notify_render_failed,
    notify_timeline_complete,
)

logger = logging.getLogger(__name__)


def _latest_job(db: Session, project_id: str) -> Job:
    job = (
        db.query(Job)
        .filter(Job.project_id == project_id)
        .order_by(Job.id.desc())
        .first()
    )
    if job is None:
        raise RuntimeError(f"No media job exists for project {project_id}.")
    return job


def run_timeline_generation(
    db: Session,
    project_id: str,
    *,
    shot_uids: list[str] | None = None,
    resolution: str | None = None,
) -> dict:
    """Generate queued timeline shots without an HTTP/background-task path."""
    job = _latest_job(db, project_id)
    timeline = latest_timeline(db, int(job.id))
    if timeline is None:
        raise RuntimeError(f"No timeline exists for project {project_id}.")

    wanted = {str(uid) for uid in (shot_uids or [])}
    pending = [shot for shot in pending_shots(timeline)
               if not wanted or str(shot.shot_uid) in wanted]
    queued = [shot for shot in queued_shots(timeline)
              if not wanted or str(shot.shot_uid) in wanted]
    if pending:
        queue_timeline_clips(
            db,
            timeline,
            shot_uids=[str(shot.shot_uid) for shot in pending],
            resolution=resolution,
        )
    if not pending and not queued:
        return {"requested": 0, "succeeded": 0, "failed": 0}

    job.status = "timeline_generating"
    job.error_message = None
    db.commit()

    try:
        result = generate_timeline_clips(
            db,
            timeline,
            progress=JobProgress(int(job.id)),
            shot_uids=list(wanted) if wanted else None,
            resolution=resolution,
        )
    except Exception as exc:
        job.status = "failed"
        job.error_message = str(exc)[:1000]
        db.commit()
        notify_render_failed(project_id, str(exc))
        raise

    failed = int(result.get("failed", 0))
    job.status = "failed" if failed else "timeline_complete"
    job.error_message = "One or more shots failed." if failed else None
    db.commit()
    notify_timeline_complete(
        project_id,
        requested=int(result.get("requested", 0)),
        succeeded=int(result.get("succeeded", 0)),
        failed=failed,
    )
    logger.info(
        "TIMELINE_GENERATION_DONE project_id=%s requested=%s succeeded=%s failed=%s",
        project_id,
        result.get("requested", 0),
        result.get("succeeded", 0),
        failed,
    )
    return result


def run_timeline_assembly(db: Session, project_id: str) -> dict:
    """Assemble the editor-approved timeline on the CPU worker."""
    job = _latest_job(db, project_id)
    timeline = latest_timeline(db, int(job.id))
    if timeline is None:
        raise RuntimeError(f"No timeline exists for project {project_id}.")

    clips = timeline.ordered_clips()
    if not clips:
        raise RuntimeError("The timeline is empty.")
    outstanding = [clip.shot_uid for clip in clips if clip.state != "ready"]
    if outstanding:
        raise RuntimeError(f"{len(outstanding)} shot(s) have no video yet.")

    grid = timeline.grid_seconds
    plan = [
        {
            "shot_uid": clip.shot_uid,
            "s3_uri": clip.clip.s3_uri,
            "trim_start": clip.trim_start or 0.0,
            "duration": clip.duration_for(grid),
        }
        for clip in clips
    ]
    job.status = "assembling"
    job.error_message = None
    db.commit()

    try:
        info = assemble_timeline_film(
            db,
            int(job.id),
            plan,
            music_uri=job.music_uri,
            music_offset=float(job.beat_offset or 0.0),
        )
    except Exception as exc:
        job.status = "failed"
        job.error_message = str(exc)[:1000]
        db.commit()
        notify_render_failed(project_id, str(exc))
        raise

    job.status = "complete"
    job.error_message = None
    db.commit()
    notify_render_complete(project_id)
    logger.info("TIMELINE_ASSEMBLY_DONE project_id=%s", project_id)
    return {
        "project_id": project_id,
        "shots": len(plan),
        "duration": float(info.get("duration", 0.0)),
        "s3_uri": info.get("s3_uri"),
    }
