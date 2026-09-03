"""SQS Worker for phased pipeline execution."""
import json
import logging
import signal
import sys

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.session import get_db_context
from app.services.sqs.consumer import SQSConsumer
from app.pipeline.orchestrator import PipelineOrchestrator
from app.schemas.job import JobMessage
from app.models.warmup import warmup_core_models

setup_logging()
logger = logging.getLogger(__name__)

# Phases each worker type handles
CPU_PHASES = [3, 4]   # Timeline, Assembly
GPU_PHASES = [1, 2]   # VGGT analyze, Render clips


def _configured_phases(default_phases: list[int]) -> list[int]:
    if not settings.WORKER_PHASES:
        return default_phases
    try:
        phases = [int(value.strip()) for value in settings.WORKER_PHASES.split(",") if value.strip()]
    except ValueError as error:
        raise RuntimeError("WORKER_PHASES must be a comma-separated list of phase numbers") from error
    if not phases or any(phase not in {1, 2, 3, 4} for phase in phases):
        raise RuntimeError("WORKER_PHASES must contain phase numbers from 1 through 4")
    return phases


def process_message(message: dict) -> None:
    """Process a single SQS message."""
    action = message.get("action", "run")

    if action != "run":
        logger.warning(f"Unknown action: {action}")
        return

    # Parse message
    job_message = JobMessage(**message)

    # Geometry reconstruction phase 1 belongs to GPU workers in all environments.
    if settings.ENVIRONMENT == "development":
        allowed_phases = _configured_phases(GPU_PHASES if settings.WORKER_TYPE == "gpu" else CPU_PHASES)
    elif settings.WORKER_TYPE == "gpu":
        allowed_phases = _configured_phases(GPU_PHASES)
    else:
        allowed_phases = _configured_phases(CPU_PHASES)

    # A worker that cannot run the requested phase must not consume the message.
    # Returning here would let the consumer delete it as "successfully
    # processed", stranding the job at `pending` forever with the work
    # unrecoverable -- observed on jobs 47 and 48 when a stray WORKER_TYPE=cpu
    # worker won phase-1 messages it could only skip. Raising instead lets the
    # visibility timeout return the message to the queue for a capable worker.
    requested_phase = int(job_message.start_phase or 1)
    if requested_phase not in allowed_phases:
        raise RuntimeError(
            f"{settings.WORKER_TYPE} worker cannot run phase {requested_phase} "
            f"(handles {allowed_phases}); returning message to the queue"
        )

    with get_db_context() as db:
        orchestrator = PipelineOrchestrator(db)

        # Create or find job
        job = orchestrator.create_job_from_message(job_message)

        # Execute phases
        orchestrator.execute(
            job.id,
            start_phase=job_message.start_phase,
            allowed_phases=allowed_phases,
        )


def main():
    """Main worker entry point."""
    logger.info(f"Starting {settings.WORKER_TYPE.upper()} worker...")
    logger.info(f"SQS Queue: {settings.SQS_QUEUE_URL}")
    warmup_core_models(
        context=f"worker:{settings.WORKER_TYPE}",
        include_vggt=settings.WORKER_TYPE == "gpu",
        include_legacy=False,
    )

    # Handle shutdown gracefully
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal, exiting...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    consumer = SQSConsumer(
        queue_url=settings.SQS_QUEUE_URL,
        handler=process_message,
        visibility_timeout=3600,
    )

    try:
        consumer.start()
    except KeyboardInterrupt:
        logger.info("Worker stopped")


if __name__ == "__main__":
    main()
