"""Backward-compatible dispatcher for the role-specific worker entrypoints.

Deployments should invoke ``app.workers.cpu`` or ``app.workers.gpu`` directly.
This module remains for existing local commands and scripts.
"""
from __future__ import annotations

from app.core.config import settings
from app.workers.runtime import queue_url


def _queue_url() -> str:
    return queue_url(settings.WORKER_TYPE)


def process_message(message: dict) -> None:
    if settings.WORKER_TYPE == "cpu":
        from app.workers.cpu import process_message as handler
    else:
        from app.workers.gpu import process_message as handler
    handler(message)


def main():
    if settings.WORKER_TYPE == "cpu":
        from app.workers.cpu import main as run
    else:
        from app.workers.gpu import main as run
    run()


if __name__ == "__main__":
    main()
