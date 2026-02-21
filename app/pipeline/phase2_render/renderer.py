"""Phase 2 Renderer - Generate video clips for all room clusters.

This phase:
1. Gets all room clusters from Phase 1
2. For each cluster, generates a video clip using LTX-2
3. Uploads clips to S3
4. Creates Clip records in database
"""
import logging
from typing import List

from sqlalchemy.orm import Session

from app.db.models import Job, RoomCluster, AnalysisResult, Clip
from app.services.storage.s3_client import s3_client
from app.pipeline.phase2_render.clip_generator import generate_clip_for_cluster

logger = logging.getLogger(__name__)


class Phase2Renderer:
    """Phase 2: Render video clips for each room cluster using LTX-2.

    Follows the same patterns as Phase1Analyzer:
    - Status updates before/after processing
    - Error handling with job state
    - Database commits after each step
    """

    def __init__(self, db: Session):
        self.db = db

    def run(self, job_id: int) -> bool:
        """Run Phase 2 rendering for a job.

        Args:
            job_id: Job ID to process

        Returns:
            True if successful
        """
        logger.info(f"Starting Phase 2 rendering for job {job_id}")

        job = self.db.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found")
            return False

        # Verify job is in correct state
        if job.status not in ("analysis_complete", "waiting_for_gpu"):
            logger.error(
                f"Job {job_id} in invalid state for Phase 2: {job.status}"
            )
            return False

        try:
            # Update job status
            job.status = "rendering"
            job.current_phase = 2
            self.db.commit()

            # Get all clusters with analysis results
            clusters = self._get_clusters_with_analysis(job.id)
            if not clusters:
                logger.warning(f"No clusters found for job {job_id}")
                job.status = "render_complete"
                self.db.commit()
                return True

            logger.info(f"Rendering clips for {len(clusters)} clusters")

            # Generate clip for each cluster
            clips_generated = 0
            clips_failed = 0

            for cluster, analysis in clusters:
                clip = generate_clip_for_cluster(
                    db=self.db,
                    job=job,
                    cluster=cluster,
                    analysis=analysis,
                    s3_client=s3_client,
                )

                if clip:
                    clips_generated += 1
                    self.db.commit()
                else:
                    clips_failed += 1
                    logger.warning(
                        f"Failed to generate clip for cluster {cluster.id}"
                    )

            # Update job status
            if clips_generated > 0:
                job.status = "render_complete"
            else:
                job.status = "failed"
                job.error_message = "No clips generated"

            self.db.commit()

            logger.info(
                f"Phase 2 complete for job {job_id}: "
                f"{clips_generated} clips generated, {clips_failed} failed"
            )

            return clips_generated > 0

        except Exception as e:
            logger.error(f"Phase 2 failed for job {job_id}: {e}")
            self.db.rollback()
            failed_job = self.db.query(Job).filter(Job.id == job_id).first()
            if failed_job:
                failed_job.status = "failed"
                failed_job.error_message = str(e)[:1000]
                self.db.commit()
            raise

    def _get_clusters_with_analysis(
        self, job_id: int
    ) -> List[tuple[RoomCluster, AnalysisResult]]:
        """Get all clusters and their analysis results.

        Args:
            job_id: Job ID

        Returns:
            List of (RoomCluster, AnalysisResult) tuples
        """
        results = (
            self.db.query(RoomCluster, AnalysisResult)
            .join(
                AnalysisResult,
                AnalysisResult.room_cluster_id == RoomCluster.id,
            )
            .filter(RoomCluster.job_id == job_id)
            .order_by(RoomCluster.sequence_order.asc().nulls_last(), RoomCluster.id.asc())
            .all()
        )

        return results

    def render_single_cluster(
        self, job_id: int, cluster_id: int
    ) -> Clip | None:
        """Render a single cluster (for re-rendering or testing).

        Args:
            job_id: Job ID
            cluster_id: Cluster ID to render

        Returns:
            Created Clip, or None if failed
        """
        job = self.db.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found")
            return None

        cluster = (
            self.db.query(RoomCluster)
            .filter(
                RoomCluster.id == cluster_id,
                RoomCluster.job_id == job_id,
            )
            .first()
        )

        if not cluster:
            logger.error(f"Cluster {cluster_id} not found for job {job_id}")
            return None

        analysis = (
            self.db.query(AnalysisResult)
            .filter(AnalysisResult.room_cluster_id == cluster_id)
            .first()
        )

        if not analysis:
            logger.error(f"No analysis result for cluster {cluster_id}")
            return None

        clip = generate_clip_for_cluster(
            db=self.db,
            job=job,
            cluster=cluster,
            analysis=analysis,
            s3_client=s3_client,
        )

        if clip:
            self.db.commit()

        return clip
