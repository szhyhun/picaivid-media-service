"""Phase 1 Analyzer - Main coordinator for photo analysis and planning."""
import logging
from typing import List

from sqlalchemy.orm import Session

from app.db.models import Job, JobPhoto, RoomCluster
from app.models.openclip import openclip_model
from app.models.midas import midas_model
from app.services.storage.s3_client import s3_client
from app.services.rails.photo_reader import rails_photo_reader, RailsPhoto
from app.pipeline.phase1_analyze.clustering import cluster_photos_by_room
from app.pipeline.phase1_analyze.scoring import compute_photo_scores
from app.pipeline.phase1_analyze.motion_planner import plan_motion_for_cluster

logger = logging.getLogger(__name__)


class Phase1Analyzer:
    """Phase 1: Analyze photos, cluster by room, score quality, plan motion.

    This phase:
    1. Reads photos from Rails DB (read-only)
    2. Creates JobPhoto records in Python DB
    3. Computes embeddings with OpenCLIP
    4. Classifies room types
    5. Analyzes depth with MiDaS
    6. Computes quality scores
    7. Clusters photos by room
    8. Plans motion strategy per cluster
    """

    def __init__(self, db: Session):
        self.db = db

    def run(self, job_id: int) -> bool:
        """Run Phase 1 analysis for a job.

        Args:
            job_id: Job ID to process

        Returns:
            True if successful
        """
        logger.info(f"Starting Phase 1 analysis for job {job_id}")

        job = self.db.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found")
            return False

        try:
            # Update job status
            job.status = "analyzing"
            job.current_phase = 1
            self.db.commit()

            # Step 1: Read photos from Rails DB
            rails_photos = rails_photo_reader.get_photos_by_project_id(job.project_id)
            if not rails_photos:
                logger.error(f"No photos found in Rails for project {job.project_id}")
                job.status = "failed"
                job.error_message = "No photos found in project"
                self.db.commit()
                return False

            logger.info(f"Found {len(rails_photos)} photos in Rails")

            # Step 2: Create JobPhoto records in Python DB
            job_photos = self._create_job_photos(job, rails_photos)
            self.db.commit()

            # Step 3: Compute embeddings
            self._compute_embeddings(job_photos)
            self.db.commit()

            # Step 4: Classify room types
            self._classify_rooms(job_photos)
            self.db.commit()

            # Step 5: Analyze depth
            self._analyze_depth(job_photos)
            self.db.commit()

            # Step 6: Compute quality scores
            compute_photo_scores(job_photos)
            self.db.commit()

            # Step 7: Cluster photos by room
            clusters = cluster_photos_by_room(self.db, job, job_photos)

            # Step 8: Plan motion for each cluster
            for cluster in clusters:
                plan_motion_for_cluster(self.db, cluster)

            # Update job status
            job.status = "analysis_complete"
            self.db.commit()

            logger.info(f"Phase 1 complete for job {job_id}: {len(clusters)} room clusters")
            return True

        except Exception as e:
            logger.error(f"Phase 1 failed for job {job_id}: {e}", exc_info=True)
            job.status = "failed"
            job.error_message = str(e)[:1000]
            self.db.commit()
            raise

    def _create_job_photos(self, job: Job, rails_photos: List[RailsPhoto]) -> List[JobPhoto]:
        """Create JobPhoto records from Rails photos.

        Args:
            job: Job instance
            rails_photos: Photos read from Rails DB

        Returns:
            List of created JobPhoto instances
        """
        job_photos = []
        for rp in rails_photos:
            # Use the s3_object_key directly (not as s3:// URI)
            # The S3 client handles bucket internally
            s3_uri = rp.s3_object_key

            job_photo = JobPhoto(
                job_id=job.id,
                rails_photo_id=rp.id,
                s3_uri=s3_uri,
                filename=rp.filename,
                width=rp.width,
                height=rp.height,
                position=rp.position,
                room_override=rp.room_type,  # Manual override from Rails
                manual_metadata=rp.metadata or {},
            )
            self.db.add(job_photo)
            job_photos.append(job_photo)

        self.db.flush()  # Get IDs
        logger.info(f"Created {len(job_photos)} JobPhoto records")
        return job_photos

    def _compute_embeddings(self, photos: List[JobPhoto]) -> None:
        """Compute OpenCLIP embeddings for all photos."""
        logger.info("Computing embeddings...")

        for i, photo in enumerate(photos):
            if photo.embedding is not None:
                continue

            try:
                image = s3_client.download_image(photo.s3_uri)
                embedding = openclip_model.get_embedding(image)
                photo.embedding = embedding.tolist()
                logger.info(f"Embedding computed for photo {i+1}/{len(photos)}")
            except Exception as e:
                logger.error(f"FAILED to compute embedding for photo {photo.id}: {e}", exc_info=True)
                photo.embedding = None

    def _classify_rooms(self, photos: List[JobPhoto]) -> None:
        """Classify room type for each photo."""
        logger.info("Classifying rooms...")

        for photo in photos:
            # Skip if manually overridden
            if photo.room_override:
                photo.room_label = photo.room_override
                continue

            if photo.room_label:
                continue

            try:
                image = s3_client.download_image(photo.s3_uri)
                room_type, confidence = openclip_model.classify_room(image)
                photo.room_label = room_type
                logger.info(f"Photo {photo.id}: {room_type} ({confidence:.2f})")
            except Exception as e:
                logger.error(f"FAILED to classify room for photo {photo.id}: {e}", exc_info=True)
                photo.room_label = "unknown"

    def _analyze_depth(self, photos: List[JobPhoto]) -> None:
        """Analyze depth for each photo."""
        logger.info("Analyzing depth...")

        for i, photo in enumerate(photos):
            if photo.depth_variance is not None:
                continue

            try:
                image = s3_client.download_image(photo.s3_uri)
                depth_metrics = midas_model.analyze_depth(image)
                photo.depth_variance = depth_metrics["variance"]
                photo.depth_layers = depth_metrics["depth_layers"]
                logger.info(f"Depth analyzed for photo {i+1}/{len(photos)}: var={photo.depth_variance:.3f}, layers={photo.depth_layers}")
            except Exception as e:
                logger.error(f"FAILED to analyze depth for photo {photo.id}: {e}", exc_info=True)
                photo.depth_variance = 0.0
                photo.depth_layers = 1
