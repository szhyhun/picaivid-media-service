"""Pipeline orchestrator for phased execution."""
import logging
from typing import List, Optional

from sqlalchemy.orm import Session

from app.db.models import Job
from app.pipeline.phase1_analyze.analyzer import Phase1Analyzer
from app.schemas.job import JobMessage

logger = logging.getLogger(__name__)

# Phases each worker type handles
CPU_PHASES = [1, 3, 4]  # Analyze, Timeline, Assembly
GPU_PHASES = [2]         # Render clips


class PipelineOrchestrator:
    """Orchestrates phased pipeline execution.

    Phase 1: Analyze and Plan (CPU)
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
                break

            success = self._run_phase(job, current)
            if not success:
                logger.error(f"Phase {current} failed for job {job_id}")
                break

            current += 1

            # Check if next phase is allowed
            if current > 4:
                logger.info(f"Job {job_id} complete")
                job.status = "complete"
                self.db.commit()
                break

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
            # Phase 2: Render clips (GPU only)
            logger.info("Phase 2 (render) requires GPU worker")
            job.status = "waiting_for_gpu"
            self.db.commit()
            return False  # Stop here, GPU worker will pick up

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
