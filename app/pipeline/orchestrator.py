"""Pipeline orchestrator for phased execution."""
import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from app.db.models import Job
from app.pipeline.phase1_analyze.analyzer import Phase1Analyzer
from app.pipeline.phase2_render.renderer import Phase2Renderer
from app.schemas.job import JobMessage
from app.services.rails_webhook import notify_render_complete, notify_render_failed

from app.pipeline.progress import JobCancelled

logger = logging.getLogger(__name__)

# Phases each worker type handles
CPU_PHASES = [3, 4]   # Timeline, Assembly
GPU_PHASES = [1, 2]   # VGGT analyze, Render clips


class PipelineOrchestrator:
    """Orchestrates phased pipeline execution.

    Phase 1: Analyze and Plan (GPU / VGGT)
    Phase 2: Render Clips (GPU)
    Phase 3: Timeline and Beat Sync (CPU)
    Phase 4: Final Assembly (CPU)
    """

    def __init__(self, db: Session):
        self.db = db

    def create_job_from_message(self, message: JobMessage) -> Job:
        """Create a new job from SQS message.

        Args:
            message: Parsed SQS message

        Returns:
            Created Job instance
        """
        job = Job(
            project_id=message.project_id,
            template_type=message.template_type,
            target_length=message.target_length,
            music_uri=message.music_uri,
            enable_beat_sync=message.enable_beat_sync,
            status="pending",
            current_phase=0,
        )
        self.db.add(job)
        self.db.commit()

        logger.info(f"Created job {job.id} for project {message.project_id}")
        return job

    def execute(
        self,
        job_id: int,
        start_phase: Optional[int] = None,
        allowed_phases: Optional[List[int]] = None,
    ) -> None:
        """Execute pipeline phases for a job.

        Args:
            job_id: Job ID to process
            start_phase: Optional phase to start from (for resume)
            allowed_phases: Optional list of phases this worker can run
        """
        job = self.db.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        # Determine which phase to run
        current = start_phase if start_phase else job.current_phase
        if current == 0:
            current = 1  # Start with Phase 1

        # Default to CPU phases if not specified
        if allowed_phases is None:
            allowed_phases = CPU_PHASES

        logger.info(f"Executing job {job_id} from phase {current}, allowed: {allowed_phases}")

        # Run phases sequentially
        while current <= 4:
            if current not in allowed_phases:
                logger.info(f"Phase {current} not in allowed phases, stopping")
                # If we just finished Phase 2, notify Rails
                if current == 3 and 2 in allowed_phases:
                    notify_render_complete(job.project_id)
                break

            try:
                success = self._run_phase(job, current)
            except JobCancelled:
                # Cooperative cancel: the pipeline unwound at a loop boundary,
                # so nothing is half-written in memory. Discard whatever partial
                # scene graph reached the database so a re-run starts clean.
                logger.info("Job %s cancelled during phase %s", job_id, current)
                self.db.rollback()
                self._discard_partial_analysis(job_id)
                job = self.db.query(Job).filter(Job.id == job_id).first()
                if job is not None:
                    job.status = "cancelled"
                    job.canceled_at = datetime.utcnow()
                    job.cancel_requested = False
                    job.progress_label = "Cancelled"
                    self.db.commit()
                    notify_render_failed(job.project_id, "Cancelled by request")
                return

            if not success:
                logger.error(f"Phase {current} failed for job {job_id}")
                notify_render_failed(job.project_id, job.error_message or "Unknown error")
                break

            current += 1

            # Check if next phase is allowed
            if current > 4:
                logger.info(f"Job {job_id} complete")
                job.status = "complete"
                self.db.commit()
                notify_render_complete(job.project_id)
                break

    def _discard_partial_analysis(self, job_id: int) -> None:
        """Delete the partial scene graph left by a cancelled run.

        Keeps job_photos (and therefore the uploaded photos and their
        embeddings) so re-running does not re-download or re-encode anything.
        """
        from app.db.models import (
            AnalysisResult,
            PhotoRelation,
            PhotoSceneGeometry,
            RoomCluster,
            SceneComponent,
            SceneComponentMembership,
        )

        for model in (
            SceneComponentMembership,
            PhotoSceneGeometry,
            PhotoRelation,
            AnalysisResult,
            RoomCluster,
            SceneComponent,
        ):
            try:
                self.db.query(model).filter(model.job_id == job_id).delete(
                    synchronize_session=False
                )
            except Exception as error:
                logger.warning(
                    "Could not clear %s for cancelled job %s: %s",
                    model.__name__, job_id, error,
                )
        self.db.commit()

    def _run_phase(self, job: Job, phase: int) -> bool:
        """Run a single phase.

        Args:
            job: Job instance
            phase: Phase number (1-4)

        Returns:
            True if successful
        """
        logger.info(f"Running phase {phase} for job {job.id}")

        if phase == 1:
            analyzer = Phase1Analyzer(self.db)
            return analyzer.run(job.id)

        elif phase == 2:
            # Phase 2: Render clips (GPU preferred, but mock mode works on CPU)
            renderer = Phase2Renderer(self.db)
            return renderer.run(job.id)

        elif phase == 3:
            # Phase 3: Timeline (not implemented yet)
            logger.info("Phase 3 (timeline) not yet implemented")
            job.status = "phase3_pending"
            self.db.commit()
            return False

        elif phase == 4:
            # Phase 4: Assembly (not implemented yet)
            logger.info("Phase 4 (assembly) not yet implemented")
            job.status = "phase4_pending"
            self.db.commit()
            return False

        return False
