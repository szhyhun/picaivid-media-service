"""CPU worker for Runware polling, timeline generation, and assembly."""
from __future__ import annotations

import logging

from app.db.session import get_db_context
from app.pipeline.phase3_timeline.worker_commands import (
    run_timeline_assembly,
    run_timeline_generation,
)
from app.schemas.job import JobMessage
from app.workers.runtime import CPU_PHASES, configured_phases, run_worker

logger = logging.getLogger(__name__)


def process_message(message: dict) -> None:
    """Process a CPU queue command from Rails."""
    action = message.get("action", "run")
    if action in {"timeline_generate", "timeline_assemble"}:
        job_message = JobMessage(**message)
        with get_db_context() as db:
            if action == "timeline_generate":
                kwargs = {
                    "shot_uids": job_message.shot_uids,
                    "resolution": job_message.resolution,
                }
                if job_message.timeline_id is not None:
                    kwargs["timeline_id"] = job_message.timeline_id
                run_timeline_generation(db, job_message.project_id, **kwargs)
            else:
                if job_message.timeline_id is None:
                    run_timeline_assembly(db, job_message.project_id)
                else:
                    run_timeline_assembly(
                        db, job_message.project_id, timeline_id=job_message.timeline_id)
        return

    if action != "run":
        logger.warning("Unknown CPU worker action: %s", action)
        return

    # Legacy phase messages are accepted only when explicitly addressed to a
    # CPU phase. Timeline commands above remain the normal Rails path.
    from app.pipeline.orchestrator import PipelineOrchestrator

    job_message = JobMessage(**message)
    allowed_phases = configured_phases(CPU_PHASES)
    requested_phase = int(job_message.start_phase or 1)
    if requested_phase not in allowed_phases:
        raise RuntimeError(
            f"cpu worker cannot run phase {requested_phase} "
            f"(handles {allowed_phases}); returning message to the queue"
        )

    with get_db_context() as db:
        orchestrator = PipelineOrchestrator(db)
        job = orchestrator.create_job_from_message(job_message)
        orchestrator.execute(
            job.id,
            start_phase=job_message.start_phase,
            allowed_phases=allowed_phases,
        )


def main() -> None:
    run_worker("cpu", process_message, include_vggt=False)


if __name__ == "__main__":
    main()
