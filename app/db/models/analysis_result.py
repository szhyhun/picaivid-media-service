"""AnalysisResult model for Phase 1 analysis outputs."""
from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base


class AnalysisResult(Base):
    """Analysis result table - stores Phase 1 analysis outputs per room cluster.

    LTX-2 Model Recommendations:
    - LTX-2 single: Single image Ken Burns (low tier)
    - LTX-2 interpolation: Two image interpolation (medium tier)
    - LTX-2 parallax: Multi-frame parallax (medium/high + SFM)
    - LTX-2 multi-view: Full multi-view synthesis (high + SFM)
    - SEVA: Premium geometry-driven (high + SFM + premium tier)
    """

    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    room_cluster_id = Column(Integer, ForeignKey("room_clusters.id", ondelete="CASCADE"), nullable=True)

    # Motion recommendations
    recommended_motion = Column(String(50))
    allowed_motion_types = Column(JSONB)  # Array of allowed motions
    recommended_duration = Column(Float)
    tier = Column(String(20))  # low, medium, high
    model_recommendation = Column(String(100))  # LTX-2 single/interp/parallax/multi-view, SEVA

    # LTX-2 prompt and parameters
    prompt_template = Column(Text)  # Generated prompt for LTX-2
    cfg_scale = Column(Float, default=4.0)  # CFG scale 3-6
    inference_steps = Column(Integer, default=40)  # 30-50 steps

    # Debug metrics (scene geometry, component confidence, motion decisions)
    debug_metrics = Column(JSONB, default={})

    # Relationships
    room_cluster = relationship("RoomCluster", back_populates="analysis_results")

    def __repr__(self) -> str:
        return f"<AnalysisResult id={self.id} tier={self.tier} motion={self.recommended_motion} model={self.model_recommendation}>"
