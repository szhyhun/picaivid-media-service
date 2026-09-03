"""Live job progress, heartbeat, and cooperative cancellation.

The pipeline runs for minutes inside a single transaction. Anything it writes
through that session is invisible to other connections until it commits, which
is precisely the wrong shape for progress: the UI needs to see movement *while*
the work happens. So the reporter opens its own short-lived session per update
and commits immediately.

The same applies in reverse for cancellation. A cancel flag set by the API is
only visible to the pipeline if we read it on a fresh connection -- reading
through the pipeline's own session would return the row as it looked when the
transaction began, and the cancel would never be noticed.

Cancellation is cooperative by design. `raise_if_cancelled()` is called at loop
boundaries, never mid-write, so a cancelled run unwinds between pairs rather
than leaving a half-written scene graph behind.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Protocol

logger = logging.getLogger(__name__)

# How often to touch the database while a phase runs. The pair loop reports
# every 25 pairs (~19s at the measured 0.75 s/pair), which is a sensible write
# rate; this guard keeps a faster caller from writing on every iteration.
MIN_WRITE_INTERVAL_SECONDS = 2.0

# A heartbeat older than this means the worker died mid-job. Chosen well above
# the slowest observed gap between pair-progress writes so a slow run is never
# mistaken for a dead one.
STALE_AFTER_SECONDS = 90


class JobCancelled(Exception):
    """Raised inside the pipeline when a cancel has been requested."""


class ProgressReporter(Protocol):
    """What the pipeline needs; keeps the pipeline free of DB concerns."""

    def update(self, current: int, total: int, label: str) -> None: ...

    def raise_if_cancelled(self) -> None: ...


class NullProgress:
    """Default reporter. Lets the pipeline run untouched in tests and scripts."""

    def update(self, current: int, total: int, label: str) -> None:
        return None

    def raise_if_cancelled(self) -> None:
        return None


class JobProgress:
    """Database-backed reporter for one job."""

    def __init__(self, job_id: int, *, session_factory=None) -> None:
        self._job_id = int(job_id)
        if session_factory is None:
            from app.db.session import SessionLocal

            session_factory = SessionLocal
        self._session_factory = session_factory
        self._last_write = 0.0

    def _write(self, **values) -> None:
        from app.db.models import Job

        session = self._session_factory()
        try:
            session.query(Job).filter(Job.id == self._job_id).update(
                values, synchronize_session=False
            )
            session.commit()
        except Exception as error:
            # Progress reporting must never take down the job it is reporting on.
            session.rollback()
            logger.warning("Progress write failed for job %s: %s", self._job_id, error)
        finally:
            session.close()

    def start_phase(self, label: str, total: int | None = None) -> None:
        now = datetime.utcnow()
        self._write(
            progress_label=label[:120],
            progress_current=0,
            progress_total=total,
            phase_started_at=now,
            heartbeat_at=now,
        )
        self._last_write = time.monotonic()

    def update(self, current: int, total: int, label: str) -> None:
        now = time.monotonic()
        # Always write the final tick so the bar lands on 100% rather than
        # stopping wherever the throttle happened to fall.
        if now - self._last_write < MIN_WRITE_INTERVAL_SECONDS and current != total:
            return
        self._last_write = now
        self._write(
            progress_current=int(current),
            progress_total=int(total),
            progress_label=label[:120],
            heartbeat_at=datetime.utcnow(),
        )

    def heartbeat(self) -> None:
        self._write(heartbeat_at=datetime.utcnow())

    def cancelled(self) -> bool:
        from app.db.models import Job

        session = self._session_factory()
        try:
            flag = (
                session.query(Job.cancel_requested)
                .filter(Job.id == self._job_id)
                .scalar()
            )
            return bool(flag)
        except Exception as error:
            # If we cannot read the flag, assume no cancel and keep working.
            # Aborting a healthy job because of a transient read error would be
            # far worse than finishing one the user asked to stop.
            logger.warning("Cancel check failed for job %s: %s", self._job_id, error)
            return False
        finally:
            session.close()

    def raise_if_cancelled(self) -> None:
        if self.cancelled():
            raise JobCancelled(f"Job {self._job_id} cancelled by request")
