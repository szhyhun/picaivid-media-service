"""Commands executed by the private CPU media worker from Rails SQS jobs."""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.db.models import Job
from app.db.models.timeline import Timeline
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


def _job_and_timeline(db: Session, project_id: str, timeline_id: int | None):
    if timeline_id is not None:
        timeline = (db.query(Timeline)
                    .join(Job, Timeline.job_id == Job.id)
                    .filter(Timeline.id == int(timeline_id),
                            Job.project_id == project_id)
                    .first())
        if timeline is None:
            raise RuntimeError(f"Timeline {timeline_id} does not belong to project {project_id}.")
        return timeline.job, timeline

    job = _latest_job(db, project_id)
    return job, latest_timeline(db, int(job.id))


def run_timeline_generation(
    db: Session,
    project_id: str,
    *,
    shot_uids: list[str] | None = None,
    resolution: str | None = None,
    timeline_id: int | None = None,
    handoff_to_assembly: bool = False,
) -> dict:
    """Generate queued timeline shots without an HTTP/background-task path."""
    job, timeline = _job_and_timeline(db, project_id, timeline_id)
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
    job.status = "failed" if failed else (
        "assembling" if handoff_to_assembly else "timeline_complete"
    )
    job.error_message = "One or more shots failed." if failed else None
    db.commit()
    if not handoff_to_assembly:
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


def run_timeline_film(
    db: Session,
    project_id: str,
    *,
    shot_uids: list[str] | None = None,
    resolution: str | None = None,
    timeline_id: int | None = None,
) -> dict:
    """Generate the requested shots and assemble the film in one CPU job."""
    generation = run_timeline_generation(
        db,
        project_id,
        shot_uids=shot_uids,
        resolution=resolution,
        timeline_id=timeline_id,
        handoff_to_assembly=True,
    )
    if int(generation.get("failed", 0)):
        return {"generation": generation, "assembled": False}

    # A timeline with no pending shots still needs an explicit assembling
    # status before the final render begins, so the single UI action has no
    # misleading complete gap between its two internal phases.
    job, _timeline = _job_and_timeline(db, project_id, timeline_id)
    job.status = "assembling"
    db.commit()
    assembly = run_timeline_assembly(db, project_id, timeline_id=timeline_id)
    return {"generation": generation, "assembly": assembly, "assembled": True}


def run_timeline_assembly(db: Session, project_id: str, *, timeline_id: int | None = None) -> dict:
    """Assemble the editor-approved timeline on the CPU worker."""
    job, timeline = _job_and_timeline(db, project_id, timeline_id)
    if timeline is None:
        raise RuntimeError(f"No timeline exists for project {project_id}.")

    clips = timeline.ordered_clips()
    if not clips:
        raise RuntimeError("The timeline is empty.")
    # A shot still in flight has to be waited for -- its video is coming and
    # cutting without it would be assembling the wrong film.
    unfinished = [c.shot_uid for c in clips if c.state in {"planned", "generating"}]
    if unfinished:
        raise RuntimeError(f"{len(unfinished)} shot(s) have no video yet.")

    # A shot that has *failed* is a different case. Waiting for it achieves
    # nothing, and holding the whole film hostage to one bad generation is
    # worse than delivering the rest. It is skipped, and said out loud.
    usable = [c for c in clips if c.state == "ready"]
    skipped = [c.shot_uid for c in clips if c.state == "failed"]
    if not usable:
        raise RuntimeError("Every shot failed to generate; there is nothing to assemble.")
    if skipped:
        logger.warning("ASSEMBLY_SKIPPING_FAILED count=%s shots=%s",
                       len(skipped), ", ".join(skipped))

    grid = timeline.grid_seconds
    plan = [
        {
            "shot_uid": clip.shot_uid,
            "s3_uri": clip.clip.s3_uri,
            "trim_start": clip.trim_start or 0.0,
            "duration": clip.duration_for(grid),
            "source_seconds": clip.source_seconds(),
        }
        for clip in usable
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
