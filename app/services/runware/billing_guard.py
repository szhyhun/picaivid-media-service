"""The single place that decides whether a clip may be submitted to a provider.

Submission is the moment money is committed, so "has this clip been submitted?"
must be a durable, atomic fact. It cannot be inferred from `Clip.status`: a
worker killed between the submit call and the status write leaves a clip that
looks untouched but has already been paid for, and the next worker to see it
would order the same video again.

The rule is therefore: **claim, then submit -- never the reverse.** The claim is
a conditional UPDATE, so if two workers race (an SQS message redelivered while
the first worker is still running is the realistic case), exactly one wins and
the loser polls the task the winner started instead of opening a second one.

Losing the claim is not an error. It means the work is already in flight and
this worker should collect the result rather than pay for it again.
"""
from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def claim_for_submit(clip_id: int, *, session_factory=None) -> bool:
    """Take the exclusive right to submit this clip. True if we may submit.

    Writes through its own short-lived session, deliberately: the caller is
    inside an async batch holding a session that may not be committed for
    minutes, and a claim that is not visible to other workers immediately is
    not a claim at all.
    """
    from app.db.models import Clip

    if session_factory is None:
        from app.db.session import SessionLocal

        session_factory = SessionLocal

    session = session_factory()
    try:
        claimed = (
            session.query(Clip)
            .filter(Clip.id == int(clip_id))
            .filter(Clip.provider_submitted_at.is_(None))
            .update({"provider_submitted_at": datetime.utcnow()},
                    synchronize_session=False)
        )
        session.commit()
        return bool(claimed)
    except Exception:
        # Fail closed. An unusable database must not become permission to
        # spend money; the shot fails visibly and a human retries it.
        session.rollback()
        logger.exception("CLAIM_FAILED clip=%s", clip_id)
        return False
    finally:
        session.close()


def already_submitted(clip) -> bool:
    """Whether this clip has ever been handed to the provider."""
    return getattr(clip, "provider_submitted_at", None) is not None
