"""Timeline model for video timeline data.

One `TimelineClip` row is one shot, and it serves both editing stages. In the
storyboard a shot is a *plan* -- photos, kind, camera move, and no duration at
all, because a photograph has none. In the timeline it is a *clip*: five
generated seconds you can cut anywhere inside.

Keeping both in one row is deliberate. A shot must hold one identity across the
two stages or "regenerate this shot" has nothing to name, and the storyboard
cannot show which shots already have video.
"""
from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base

# What we buy from the video provider per shot. Bigger than any beat length we
# will ever cut, which is what makes trimming and retiming free.
SOURCE_SECONDS = 5.0
# Both kinds are bought at five seconds, because Ray 3.2 sells no other
# length that keeps start/end keyframes. Kept as two names so the trim budget
# follows the purchase automatically if a single ever moves to a model with
# finer duration control.
PAIR_SOURCE_SECONDS = 5.0
SINGLE_SOURCE_SECONDS = 5.0


def source_seconds_for(kind: str | None) -> float:
    """How many seconds of video this shot kind is generated with."""
    return PAIR_SOURCE_SECONDS if kind == "pair_keyframes" else SINGLE_SOURCE_SECONDS



class Timeline(Base):
    """Timeline table - stores timeline data from Phase 3."""

    __tablename__ = "timelines"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    version = Column(Integer, default=1)
    status = Column(String(50), default="draft")
    beat_grid = Column(JSONB)
    total_duration = Column(Float)

    # Relationships
    job = relationship("Job", back_populates="timelines")
    timeline_clips = relationship("TimelineClip", back_populates="timeline", cascade="all, delete-orphan")

    @property
    def bpm(self) -> float:
        return float((self.beat_grid or {}).get("bpm") or 100.0)

    @property
    def beats_per_shot(self) -> int:
        return int((self.beat_grid or {}).get("beats_per_shot") or 2)

    @property
    def grid_seconds(self) -> float:
        """Default shot length: whole beats at the film's tempo.

        Measured across three reference films, shot length is exactly two beats
        and never varies within a film -- an editor snapping to a grid, not
        cutting by feel.
        """
        return self.beats_per_shot * 60.0 / self.bpm

    def ordered_clips(self) -> list["TimelineClip"]:
        return sorted(self.timeline_clips, key=lambda c: c.order_index)

    def retime(self) -> float:
        """Recompute every in/out time from the grid. Returns total duration.

        `in_time` is derived, never authoritative: storing it would let the
        stored value drift from the order it is supposed to describe.
        """
        cursor = 0.0
        for clip in self.ordered_clips():
            clip.in_time = round(cursor, 4)
            cursor += clip.duration_for(self.grid_seconds)
            clip.out_time = round(cursor, 4)
        self.total_duration = round(cursor, 4)
        return self.total_duration

    def __repr__(self) -> str:
        return f"<Timeline id={self.id} version={self.version} status={self.status}>"


class TimelineClip(Base):
    """One shot: a storyboard plan and, once generated, a trimmable clip."""

    __tablename__ = "timeline_clips"

    id = Column(Integer, primary_key=True)
    timeline_id = Column(Integer, ForeignKey("timelines.id", ondelete="CASCADE"), nullable=False, index=True)
    # Null until the shot has been generated. A planned shot is a real row.
    clip_id = Column(Integer, ForeignKey("clips.id"), nullable=True)
    order_index = Column(Integer, nullable=False)
    in_time = Column(Float)
    out_time = Column(Float)
    transition_type = Column(String(50))
    audio_policy = Column(String(50))

    # Stable identity across reordering, regeneration and both stages.
    shot_uid = Column(String(36), index=True)

    # --- Storyboard: what the shot is, independent of any generated video ---
    room_id = Column(String(120))
    room_type = Column(String(60))
    act = Column(Integer)
    kind = Column(String(30))               # pair_keyframes | single_image
    photo_ids = Column(JSONB)               # one photo, or two used as keyframes
    move_label = Column(String(120))
    move_vector = Column(JSONB)             # fractions per second
    is_close_up = Column(Boolean, default=False)
    crop_box = Column(JSONB)
    subject_box = Column(JSONB)
    source_size = Column(JSONB)
    detail_subject = Column(String(120))
    prompt_override = Column(Text)
    # The customer's own words, appended to the prompt we build rather than
    # replacing it: their direction is an addition to the measured decisions,
    # not a reason to discard them.
    prompt_note = Column(Text)
    # True once a customer picks a move. The positional defaults are a grammar,
    # not a rule: they should stop applying the moment someone disagrees.
    move_locked = Column(Boolean, default=False)
    # The customer's composed move. `move_label` stays as measured, so the
    # planner's own choice survives an override and reset has a target.
    move_chosen = Column(String(120))

    # --- Edit: how the generated source is cut into the film ---
    trim_start = Column(Float, default=0.0)
    # NULL means "follow the beat grid", so changing the music retimes every
    # unpinned shot without regenerating a single clip.
    duration_override = Column(Float)

    timeline = relationship("Timeline", back_populates="timeline_clips")
    clip = relationship("Clip", back_populates="timeline_clips")

    @property
    def uses_keyframes(self) -> bool:
        return self.kind == "pair_keyframes"

    @property
    def state(self) -> str:
        """Derived, not stored -- a second copy of this would only drift.

        The Clip row is created at submission time because it has to hold the
        provider task id before we await, so its presence and status already
        say everything about where the shot has got to.
        """
        if self.clip_id is None or self.clip is None:
            return "planned"
        status = (self.clip.status or "").lower()
        if status in {"complete", "rendered"}:
            return "ready"
        if status == "failed":
            return "failed"
        return "generating"

    def source_seconds(self) -> float:
        """Length of the generated source behind this shot, by kind."""
        return source_seconds_for(self.kind)

    def duration_for(self, grid_seconds: float) -> float:
        """How long this shot holds on screen, clamped to what we actually bought."""
        wanted = self.duration_override if self.duration_override else grid_seconds
        start = self.trim_start or 0.0
        return max(0.1, min(float(wanted), self.source_seconds() - start))

    def max_duration(self) -> float:
        """The trim ceiling. Passing it means paying to regenerate."""
        return max(0.1, self.source_seconds() - (self.trim_start or 0.0))

    def __repr__(self) -> str:
        return f"<TimelineClip id={self.id} order={self.order_index} state={self.state}>"
