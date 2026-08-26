"""Regression coverage for persisting plans with autoflush disabled."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.db.models import AnalysisResult, JobPhoto, PhotoRelation
from app.pipeline.phase1_analyze.shot_planner import build_and_persist_shot_plan


class _Query:
    def __init__(self, session: "_AutoflushDisabledSession", model: type) -> None:
        self.session = session
        self.model = model

    def filter(self, *args: object) -> "_Query":
        del args
        return self

    def all(self) -> list[object]:
        if self.model is JobPhoto:
            return self.session.photos
        if self.model is AnalysisResult:
            # Simulate SessionLocal(autoflush=False): pending rows are invisible to
            # a query until the application explicitly flushes them.
            return self.session.analyses if self.session.flushed else []
        if self.model is PhotoRelation:
            return []
        raise AssertionError(f"unexpected query model: {self.model}")


class _AutoflushDisabledSession:
    def __init__(self, photos: list[object], analyses: list[object]) -> None:
        self.photos = photos
        self.analyses = analyses
        self.flushed = False
        self.flush_calls = 0

    def flush(self) -> None:
        self.flushed = True
        self.flush_calls += 1

    def query(self, model: type) -> _Query:
        return _Query(self, model)


class ShotPlanPersistenceTests(unittest.TestCase):
    def test_pending_analysis_rows_are_flushed_before_plan_queries(self) -> None:
        photo = SimpleNamespace(
            id=100,
            room_cluster_id=10,
            cluster_order=0,
            position=0,
            final_score=0.8,
            manual_metadata={},
        )
        analysis = SimpleNamespace(
            room_cluster_id=10,
            recommended_motion="micro_push_in",
            recommended_duration=3.0,
            debug_metrics={},
        )
        cluster = SimpleNamespace(
            id=10,
            scene_component_id=None,
            scene_component=None,
            sequence_order=0,
            room_type="bedroom",
            hero_photo_id=100,
            sfm_eligible=False,
            geometry_confidence=0.7,
            overlap_score=0.0,
            recommended_motion="micro_push_in",
            recommended_duration=3.0,
        )
        job = SimpleNamespace(id=1, project_id="project-1", target_length=30)
        session = _AutoflushDisabledSession([photo], [analysis])

        plan = build_and_persist_shot_plan(session, job, [cluster])

        self.assertGreaterEqual(session.flush_calls, 2)
        self.assertEqual(len(plan["ordered_shots"]), 1)
        self.assertIn("shot", analysis.debug_metrics)
        self.assertIn("shot_plan", analysis.debug_metrics)


if __name__ == "__main__":
    unittest.main()
