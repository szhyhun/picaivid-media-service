"""Shared process runtime for the CPU and GPU SQS workers."""
from __future__ import annotations

import logging
import signal
import sys
from collections.abc import Callable

from app.core.config import settings
from app.core.logging import setup_logging
from app.models.warmup import warmup_core_models
from app.services.sqs.consumer import SQSConsumer

CPU_PHASES = [3, 4]
GPU_PHASES = [1, 2]


def queue_url(role: str) -> str:
    if role == "cpu" and settings.SQS_CPU_QUEUE_URL:
        return settings.SQS_CPU_QUEUE_URL
    return settings.SQS_QUEUE_URL


def configured_phases(default_phases: list[int]) -> list[int]:
    if not settings.WORKER_PHASES:
        return default_phases
    try:
        phases = [int(value.strip()) for value in settings.WORKER_PHASES.split(",") if value.strip()]
    except ValueError as error:
        raise RuntimeError("WORKER_PHASES must be a comma-separated list of phase numbers") from error
    if not phases or any(phase not in {1, 2, 3, 4} for phase in phases):
        raise RuntimeError("WORKER_PHASES must contain phase numbers from 1 through 4")
    return phases


def run_worker(role: str, handler: Callable[[dict], None], *, include_vggt: bool) -> None:
    """Run one role's SQS consumer with shared shutdown and warmup behavior."""
    setup_logging()
    logger = logging.getLogger(__name__)
    selected_queue = queue_url(role)
    logger.info("Starting %s worker...", role.upper())
    logger.info("SQS Queue: %s", selected_queue)
    warmup_core_models(
        context=f"worker:{role}",
        include_vggt=include_vggt,
        include_legacy=False,
    )

    def signal_handler(sig, frame):
        logger.info("Received shutdown signal, exiting...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    consumer = SQSConsumer(
        queue_url=selected_queue,
        handler=handler,
        visibility_timeout=3600,
    )
    try:
        consumer.start()
    except KeyboardInterrupt:
        logger.info("Worker stopped")
