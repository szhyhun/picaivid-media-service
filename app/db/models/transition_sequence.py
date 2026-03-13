"""Transition sequence models for certified multi-image transitions."""
from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base


class TransitionSequence(Base):
    __tablename__ = "transition_sequences"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    sequence_rank = Column(Integer, nullable=False, default=0)
    sequence_score = Column(Float, nullable=False, default=0.0)
    certification_status = Column(String(20), nullable=False, default="usable")
    room_type_hint = Column(String(100), nullable=True)
    source_cluster_ids = Column(JSONB, nullable=True)
    motion_hint = Column(String(100), nullable=True)

    job = relationship("Job")
    steps = relationship(
        "TransitionSequenceStep",
        back_populates="sequence",
        cascade="all, delete-orphan",
        order_by="TransitionSequenceStep.step_index",
    )


class TransitionSequenceStep(Base):
    __tablename__ = "transition_sequence_steps"

    id = Column(Integer, primary_key=True)
    transition_sequence_id = Column(
        Integer,
        ForeignKey("transition_sequences.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    step_index = Column(Integer, nullable=False)
    photo_id = Column(Integer, ForeignKey("job_photos.id", ondelete="CASCADE"), nullable=False)
    photo_similarity_id = Column(
        Integer,
        ForeignKey("photo_similarities.id", ondelete="SET NULL"),
        nullable=True,
    )

    sequence = relationship("TransitionSequence", back_populates="steps")
    photo = relationship("JobPhoto")
    photo_similarity = relationship("PhotoSimilarity")
