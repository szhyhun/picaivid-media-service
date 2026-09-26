"""A clip must never be submitted to a paid provider more than once."""
from __future__ import annotations

import asyncio
import unittest
from datetime import datetime
from types import SimpleNamespace

from app.services.runware import client as rw
from app.services.runware.billing_guard import already_submitted, claim_for_submit


class _Row:
    """One clip row, with just enough of the query API to be claimed."""

    def __init__(self):
        self.provider_submitted_at = None


class _FakeSession:
    """A session whose conditional UPDATE behaves like the real one."""

    def __init__(self, row, *, explode=False):
        self.row = row
        self.explode = explode
        self.committed = 0
        self.closed = False
        self._only_unsubmitted = False

    def query(self, *_):
        return self

    def filter(self, criterion):
        # The second filter in claim_for_submit is the IS NULL guard.
        if "IS NULL" in str(criterion):
            self._only_unsubmitted = True
        return self

    def update(self, values, **_):
        if self.explode:
            raise RuntimeError("database is down")
        if self._only_unsubmitted and self.row.provider_submitted_at is not None:
            return 0
        self.row.provider_submitted_at = values["provider_submitted_at"]
        return 1

    def commit(self):
        self.committed += 1

    def rollback(self):
        pass

    def close(self):
        self.closed = True


class ClaimTests(unittest.TestCase):
    def test_the_first_claim_wins_and_the_second_loses(self):
        row = _Row()
        factory = lambda: _FakeSession(row)          # noqa: E731
        self.assertTrue(claim_for_submit(1, session_factory=factory))
        self.assertFalse(claim_for_submit(1, session_factory=factory))

    def test_a_lost_claim_leaves_the_winners_timestamp_alone(self):
        row = _Row()
        factory = lambda: _FakeSession(row)          # noqa: E731
        claim_for_submit(1, session_factory=factory)
        first = row.provider_submitted_at
        claim_for_submit(1, session_factory=factory)
        self.assertEqual(row.provider_submitted_at, first)

    def test_a_broken_database_refuses_the_claim_rather_than_allowing_a_spend(self):
        # Fail closed. An unusable database is not permission to spend money.
        row = _Row()
        self.assertFalse(
            claim_for_submit(1, session_factory=lambda: _FakeSession(row, explode=True)))

    def test_the_session_is_always_closed(self):
        row = _Row()
        sessions = []

        def factory():
            session = _FakeSession(row)
            sessions.append(session)
            return session

        claim_for_submit(1, session_factory=factory)
        claim_for_submit(1, session_factory=factory)
        self.assertTrue(all(s.closed for s in sessions))

    def test_already_submitted_reads_the_timestamp_not_the_status(self):
        # A worker killed after submitting leaves status untouched; only the
        # timestamp can tell us the money is already gone.
        mid_flight = SimpleNamespace(status="queued",
                                     provider_submitted_at=datetime.utcnow())
        untouched = SimpleNamespace(status="queued", provider_submitted_at=None)
        self.assertTrue(already_submitted(mid_flight))
        self.assertFalse(already_submitted(untouched))


class SubmitGateTests(unittest.TestCase):
    """generate_video is the only function that submits, so it is the gate."""

    def _run(self, claim):
        submitted = []
        polled = []

        async def fake_post(_client, body):
            kind = body[0].get("taskType")
            if kind == "videoInference":
                submitted.append(body[0]["taskUUID"])
                return {"data": [{"taskUUID": body[0]["taskUUID"]}]}
            polled.append(body[0]["taskUUID"])
            return {"data": [{"videoURL": "https://example.test/v.mp4"}]}

        original = rw._post
        rw._post = fake_post
        try:
            result = asyncio.run(rw.generate_video(
                None, prompt="p", photo_urls=["a.jpg"], task_uuid="t-1", claim=claim))
        finally:
            rw._post = original
        return result, submitted, polled

    def test_a_granted_claim_submits_once(self):
        result, submitted, polled = self._run(lambda: True)
        self.assertEqual(submitted, ["t-1"])
        self.assertTrue(result.ok)

    def test_a_refused_claim_polls_instead_of_submitting(self):
        # This is the redelivered-SQS-message case: the work is already in
        # flight and already paid for, so collect it rather than buy it twice.
        result, submitted, polled = self._run(lambda: False)
        self.assertEqual(submitted, [])
        self.assertEqual(polled, ["t-1"])
        self.assertTrue(result.ok)


if __name__ == "__main__":
    unittest.main()
