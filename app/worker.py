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

setup_logging()
logger = logging.getLogger(__name__)

# Phases each worker type handles
CPU_PHASES = [1, 3, 4]  # Analyze, Timeline, Assembly
GPU_PHASES = [2]         # Render clips


def process_message(message: dict) -> None:
    """Process a single SQS message."""
    action = message.get("action", "run")

    if action != "run":
        logger.warning(f"Unknown action: {action}")
        return

    # Parse message
    job_message = JobMessage(**message)
    allowed_phases = GPU_PHASES if settings.WORKER_TYPE == "gpu" else CPU_PHASES

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
