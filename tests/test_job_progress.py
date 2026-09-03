"""Progress reporting, stall detection, and cooperative cancellation."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from app.pipeline.progress import (
    STALE_AFTER_SECONDS,
    JobCancelled,
    JobProgress,
    NullProgress,
)


class _FakeSession:
    """Records writes without a database."""

    def __init__(self, store: dict, cancel_flag: bool = False) -> None:
        self._store = store
        self._cancel_flag = cancel_flag
        self.committed = 0

    def query(self, *args):
        return self

    def filter(self, *args):
        return self

    def update(self, values, synchronize_session=False):
        self._store.update(values)
        return 1

    def scalar(self):
        return self._cancel_flag

    def commit(self):
        self.committed += 1

    def rollback(self):
        pass

    def close(self):
        pass


class NullProgressTests(unittest.TestCase):
    def test_null_reporter_is_inert(self):
        """The pipeline must run untouched when nobody is watching."""
        reporter = NullProgress()
        reporter.update(1, 10, "x")
        self.assertIsNone(reporter.raise_if_cancelled())


class JobProgressWriteTests(unittest.TestCase):
    def test_start_phase_records_label_and_resets_counter(self):
        store: dict = {}
        reporter = JobProgress(1, session_factory=lambda: _FakeSession(store))
        reporter.start_phase("Analyzing scene geometry", total=200)
        self.assertEqual(store["progress_label"], "Analyzing scene geometry")
        self.assertEqual(store["progress_current"], 0)
        self.assertEqual(store["progress_total"], 200)
        self.assertIsNotNone(store["phase_started_at"])
        self.assertIsNotNone(store["heartbeat_at"])

    def test_update_throttles_but_always_writes_the_final_tick(self):
        """A throttled last write would leave the bar stuck below 100%."""
        store: dict = {}
        reporter = JobProgress(1, session_factory=lambda: _FakeSession(store))
        reporter.start_phase("phase", total=100)

        reporter.update(25, 100, "phase")   # throttled: too soon after start
        self.assertEqual(store["progress_current"], 0)

        reporter.update(100, 100, "phase")  # final tick bypasses the throttle
        self.assertEqual(store["progress_current"], 100)

    def test_write_failure_never_propagates(self):
        """Progress reporting must not be able to kill the job it reports on."""
        def exploding_factory():
            session = MagicMock()
            session.query.side_effect = RuntimeError("db down")
            return session

        reporter = JobProgress(1, session_factory=exploding_factory)
        reporter.start_phase("phase")  # must not raise
        reporter.update(5, 10, "phase")

    def test_label_is_truncated_to_column_width(self):
        store: dict = {}
        reporter = JobProgress(1, session_factory=lambda: _FakeSession(store))
        reporter.start_phase("x" * 500)
        self.assertLessEqual(len(store["progress_label"]), 120)


class CancellationTests(unittest.TestCase):
    def test_raise_if_cancelled_raises_when_flag_set(self):
        reporter = JobProgress(
            1, session_factory=lambda: _FakeSession({}, cancel_flag=True)
        )
        with self.assertRaises(JobCancelled):
            reporter.raise_if_cancelled()

    def test_raise_if_cancelled_is_quiet_when_flag_clear(self):
        reporter = JobProgress(
            1, session_factory=lambda: _FakeSession({}, cancel_flag=False)
        )
        self.assertIsNone(reporter.raise_if_cancelled())

    def test_unreadable_cancel_flag_does_not_abort_a_healthy_job(self):
        """Aborting a good run over a transient read error is worse than
        finishing one the user asked to stop."""
        def exploding_factory():
            session = MagicMock()
            session.query.side_effect = RuntimeError("db blip")
            return session

        reporter = JobProgress(1, session_factory=exploding_factory)
        self.assertFalse(reporter.cancelled())
        self.assertIsNone(reporter.raise_if_cancelled())


class StallDerivationTests(unittest.TestCase):
    """The API derives `stalled` from the heartbeat so a dead worker is
    distinguishable from a slow one -- the failure mode that made three
    separate outages look like an endless spinner."""

    def _payload(self, **overrides):
        from app.main import _job_progress_payload

        job = MagicMock()
        job.id = 1
        job.status = overrides.get("status", "analyzing")
        job.current_phase = 1
        job.progress_current = overrides.get("current", 10)
        job.progress_total = overrides.get("total", 100)
        job.progress_label = "Analyzing scene geometry"
        job.created_at = overrides.get("created_at", datetime.utcnow())
        job.heartbeat_at = overrides.get("heartbeat_at", datetime.utcnow())
        job.canceled_at = overrides.get("canceled_at", None)
        job.cancel_requested = overrides.get("cancel_requested", False)
        job.error_message = None
        return _job_progress_payload("p", job)

    def test_no_job_is_idle(self):
        from app.main import _job_progress_payload

        self.assertEqual(_job_progress_payload("p", None).state, "idle")

    def test_fresh_heartbeat_is_running(self):
        self.assertEqual(self._payload().state, "running")

    def test_stale_heartbeat_is_stalled(self):
        stale = datetime.utcnow() - timedelta(seconds=STALE_AFTER_SECONDS + 30)
        self.assertEqual(self._payload(heartbeat_at=stale).state, "stalled")

    def test_claimed_but_never_reported_is_stalled(self):
        """A worker that took the job and never wrote a heartbeat -- exactly what
        a worker that cannot run the requested phase looks like."""
        old = datetime.utcnow() - timedelta(seconds=STALE_AFTER_SECONDS + 30)
        payload = self._payload(heartbeat_at=None, created_at=old)
        self.assertEqual(payload.state, "stalled")

    def test_pending_is_queued_not_stalled(self):
        old = datetime.utcnow() - timedelta(seconds=STALE_AFTER_SECONDS + 30)
        payload = self._payload(status="pending", heartbeat_at=None, created_at=old)
        self.assertEqual(payload.state, "queued")

    def test_complete_reports_full_percent(self):
        payload = self._payload(status="render_complete", current=5, total=100)
        self.assertEqual(payload.state, "complete")
        self.assertEqual(payload.percent, 100.0)

    def test_cancelled_wins_over_status(self):
        payload = self._payload(canceled_at=datetime.utcnow())
        self.assertEqual(payload.state, "cancelled")

    def test_percent_is_clamped(self):
        self.assertEqual(self._payload(current=150, total=100).percent, 100.0)


if __name__ == "__main__":
    unittest.main()
