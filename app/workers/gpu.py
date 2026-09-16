"""GPU worker for VGGT analysis and GPU clip rendering."""
from __future__ import annotations

import logging

from app.db.session import get_db_context
from app.pipeline.orchestrator import PipelineOrchestrator
from app.schemas.job import JobMessage
from app.workers.runtime import GPU_PHASES, configured_phases, run_worker

logger = logging.getLogger(__name__)


def process_message(message: dict) -> None:
    """Process a GPU pipeline message from Rails."""
    action = message.get("action", "run")
    if action in {"timeline_generate", "timeline_assemble"}:
        raise RuntimeError(f"gpu worker cannot run timeline command {action}")
    if action != "run":
        logger.warning("Unknown GPU worker action: %s", action)
        return

    job_message = JobMessage(**message)
    allowed_phases = configured_phases(GPU_PHASES)
    requested_phase = int(job_message.start_phase or 1)
    if requested_phase not in allowed_phases:
        raise RuntimeError(
            f"gpu worker cannot run phase {requested_phase} "
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
    run_worker("gpu", process_message, include_vggt=True)


if __name__ == "__main__":
    main()
