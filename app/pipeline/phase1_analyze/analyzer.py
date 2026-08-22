"""Phase 1 Analyzer - Main coordinator for photo analysis and planning."""
import logging
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from sqlalchemy.orm import Session
from PIL import Image

from app.db.models import Job, JobPhoto, RoomCluster, PhotoRelation
from app.models.openclip import openclip_model
from app.services.storage.s3_client import s3_client
from app.services.rails.photo_reader import rails_photo_reader, RailsPhoto
from app.pipeline.phase1_analyze.clustering import cluster_photos_by_room
from app.pipeline.phase1_analyze.scoring import compute_photo_scores
from app.pipeline.phase1_analyze.motion_planner import plan_motion_for_cluster
from app.pipeline.phase1_analyze.shot_planner import build_and_persist_shot_plan

logger = logging.getLogger(__name__)


class Phase1Analyzer:
    """Phase 1: Analyze photos, cluster by room, score quality, plan motion.

    This phase:
    1. Reads photos from Rails DB (read-only)
    2. Creates JobPhoto records in Python DB
    3. Computes embeddings with OpenCLIP
    4. Classifies room types
    5. Computes quality scores
    6. Reconstructs cameras, depth, and scene geometry with VGGT
    7. Clusters photos by room
    8. Plans motion strategy per cluster
    """

    def __init__(self, db: Session):
        self.db = db
        self._image_cache_hits = 0
        self._image_cache_misses = 0
        self._image_download_seconds = 0.0

    def run(self, job_id: int) -> bool:
        """Run Phase 1 analysis for a job.

        Args:
            job_id: Job ID to process

        Returns:
            True if successful
        """
        logger.info(f"Starting Phase 1 analysis for job {job_id}")
        run_started_at = time.perf_counter()
        self._image_cache_hits = 0
        self._image_cache_misses = 0
        self._image_download_seconds = 0.0
        image_cache: Dict[int, Image.Image] = {}

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
            # Prefetch all images in parallel before the sequential pipeline steps.
            # This turns every subsequent _get_cached_image() call into a cache hit,
            # cutting total S3 download time by ~5-10x compared to sequential fetches.
            self._prefetch_images(job_photos, image_cache)

            # Step 3: Compute embeddings
            self._compute_embeddings(job_photos, image_cache)
            self.db.commit()

            # Step 4: Classify room types
            self._classify_rooms(job_photos, image_cache)
            self.db.commit()

            # Step 5: Correct common exterior label confusions using sequence context.
            self._postprocess_exterior_room_labels(job_photos)
            self.db.commit()

            # Step 5b: Correct common interior label confusions using
            # local sequence context (kitchen/living/dining).
            self._postprocess_interior_room_labels(job_photos, image_cache=image_cache)
            self.db.commit()

            # Step 6: Compute quality scores
            compute_photo_scores(job_photos, image_cache=image_cache)
            self.db.commit()

            # Step 7: Cluster photos by room (with visual overlap detection)
            clusters = cluster_photos_by_room(
                self.db,
                job,
                job_photos,
                s3_client=s3_client,
                preloaded_images=image_cache,
            )

            # Step 7b: Refine interior labels using verified geometric links
            # (e.g., living vs dining confusion in adjacent shots).
            self._postprocess_interior_room_labels(job_photos, job_id=job.id, image_cache=image_cache)
            self.db.commit()
            openclip_model.release()

            # Step 8: Plan motion for each cluster
            photos_by_cluster: Dict[int, List[JobPhoto]] = {}
            for photo in job_photos:
                if photo.room_cluster_id is None:
                    continue
                photos_by_cluster.setdefault(int(photo.room_cluster_id), []).append(photo)
            for cluster in clusters:
                plan_motion_for_cluster(
                    self.db,
                    cluster,
                    preloaded_photos=photos_by_cluster.get(int(cluster.id), []),
                )

            # Store an editorial plan independently of the later video renderer.
            build_and_persist_shot_plan(self.db, job, clusters)
            self.db.commit()

            # Update job status
            job.status = "analysis_complete"
            self.db.commit()

            logger.info(f"Phase 1 complete for job {job_id}: {len(clusters)} room clusters")
            total_run_seconds = time.perf_counter() - run_started_at
            logger.info(
                "PHASE1_RUNTIME job_id=%s total_time=%.3fs image_cache_hits=%s image_cache_misses=%s image_download_time=%.3fs",
                job_id,
                total_run_seconds,
                self._image_cache_hits,
                self._image_cache_misses,
                self._image_download_seconds,
            )
            return True

        except Exception as e:
            logger.error(f"Phase 1 failed for job {job_id}: {e}", exc_info=True)
            # SQLAlchemy transaction may be invalid after DB exceptions.
            # Roll back first, then persist failure state in a clean transaction.
            self.db.rollback()
            failed_job = self.db.query(Job).filter(Job.id == job_id).first()
            if failed_job:
                failed_job.status = "failed"
                failed_job.error_message = str(e)[:1000]
                self.db.commit()
            raise
        finally:
            # PIL keeps decoded pixel buffers outside Python's small-object heap.
            # Close them explicitly so repeated jobs do not retain a listing.
            for image in image_cache.values():
                try:
                    image.close()
                except Exception:
                    pass
            image_cache.clear()
            openclip_model.release()

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

    def _prefetch_images(self, photos: List[JobPhoto], image_cache: Dict[int, Image.Image], max_workers: int = 8) -> None:
        """Download all photos from S3 in parallel and populate image_cache.

        Runs before the sequential pipeline steps so that every subsequent
        _get_cached_image() call is a cache hit and pays no I/O latency.
        """
        missing = [p for p in photos if p.id not in image_cache]
        if not missing:
            return

        logger.info("Prefetching %s images from S3 (max_workers=%s)...", len(missing), max_workers)
        prefetch_started_at = time.perf_counter()

        def _download(photo: JobPhoto):
            return photo.id, s3_client.download_image(photo.s3_uri)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_download, p): p for p in missing}
            for future in as_completed(futures):
                photo = futures[future]
                try:
                    photo_id, image = future.result()
                    image_cache[photo_id] = image
                except Exception as e:
                    logger.error("FAILED to prefetch photo %s: %s", photo.id, e, exc_info=True)

        elapsed = time.perf_counter() - prefetch_started_at
        self._image_download_seconds += elapsed
        logger.info(
            "Prefetch complete: %s/%s images downloaded in %.3fs",
            sum(1 for p in missing if p.id in image_cache),
            len(missing),
            elapsed,
        )

    def _get_cached_image(self, photo: JobPhoto, image_cache: Dict[int, Image.Image]) -> Image.Image:
        cached = image_cache.get(photo.id)
        if cached is not None:
            self._image_cache_hits += 1
            return cached
        # Fallback for photos missed by prefetch (e.g. added after prefetch ran)
        self._image_cache_misses += 1
        started_at = time.perf_counter()
        image = s3_client.download_image(photo.s3_uri)
        self._image_download_seconds += (time.perf_counter() - started_at)
        image_cache[photo.id] = image
        return image

    def _compute_embeddings(self, photos: List[JobPhoto], image_cache: Dict[int, Image.Image]) -> None:
        """Compute OpenCLIP embeddings for all photos."""
        logger.info("Computing embeddings...")

        for i, photo in enumerate(photos):
            if photo.embedding is not None:
                continue

            try:
                image = self._get_cached_image(photo, image_cache)
                embedding = openclip_model.get_embedding(image)
                photo.embedding = embedding.tolist()
                logger.info(f"Embedding computed for photo {i+1}/{len(photos)}")
            except Exception as e:
                logger.error(f"FAILED to compute embedding for photo {photo.id}: {e}", exc_info=True)
                photo.embedding = None

    def _classify_rooms(self, photos: List[JobPhoto], image_cache: Dict[int, Image.Image]) -> None:
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
                image = self._get_cached_image(photo, image_cache)
                room_type, confidence = openclip_model.classify_room(image)

                # Lightweight metadata guardrail for a common confusion:
                # laundry rooms can be misclassified as kitchens.
                if (
                    room_type in {"kitchen", "bathroom", "unknown"}
                    and _looks_like_laundry(photo)
                ):
                    room_type = "laundry room"
                    logger.info(f"Photo {photo.id}: remapped to laundry room via metadata hint")

                photo.room_label = room_type
                logger.info(f"Photo {photo.id}: {room_type} ({confidence:.2f})")
            except Exception as e:
                logger.error(f"FAILED to classify room for photo {photo.id}: {e}", exc_info=True)
                photo.room_label = "unknown"

    def _postprocess_exterior_room_labels(self, photos: List[JobPhoto]) -> None:
        """Correct common exterior room label confusions.

        Targeted fixes:
        - mid-early front-yard frames bracketed by aerial frames -> aerial view
        - late-sequence front-yard frames near patio/backyard context -> backyard
        """
        if not photos:
            return

        ordered = sorted(photos, key=lambda p: p.position or 0)
        n = len(ordered)
        if n < 3:
            return

        def rel_pos(idx: int) -> float:
            return float(idx / max(1, n - 1))

        changed = 0
        for idx, photo in enumerate(ordered):
            if photo.room_override:
                continue

            current = _normalize_room_label(photo.room_label)
            if current not in {"front yard", "exterior front"}:
                continue

            window = ordered[max(0, idx - 8): min(n, idx + 9)]
            prev_close = ordered[max(0, idx - 2): idx]
            next_close = ordered[idx + 1: min(n, idx + 3)]

            prev_has_aerial = any(_is_aerial_label(_normalize_room_label(p.room_label)) for p in prev_close)
            next_has_aerial = any(_is_aerial_label(_normalize_room_label(p.room_label)) for p in next_close)
            near_back_context = any(_is_back_context_label(_normalize_room_label(p.room_label)) for p in window if p.id != photo.id)

            rp = rel_pos(idx)

            # 1) Early front-yard frame bracketed by aerial context -> aerial.
            if rp <= 0.20 and prev_has_aerial and next_has_aerial:
                photo.room_label = "aerial view"
                changed += 1
                logger.info(
                    "Photo %s room relabel: %s -> aerial view (early bracketed aerial context, pos=%.2f)",
                    photo.id,
                    current,
                    rp,
                )
                continue

            # 2) Late front-yard frame near patio/backyard context -> backyard.
            if current == "front yard" and rp >= 0.85 and near_back_context:
                photo.room_label = "backyard"
                changed += 1
                logger.info(
                    "Photo %s room relabel: %s -> backyard (late exterior context, pos=%.2f)",
                    photo.id,
                    current,
                    rp,
                )

        if changed:
            logger.info("Exterior room-label postprocess changed %s photos", changed)

    def _postprocess_interior_room_labels(
        self,
        photos: List[JobPhoto],
        job_id: int | None = None,
        image_cache: Dict[int, Image.Image] | None = None,
    ) -> None:
        """Correct common interior room label confusions.

        Sequence-context rules:
        - patio/exterior false-positive inside kitchen/living/dining runs -> kitchen

        Geometry-context rules (when job_id provided and similarities are available):
        - living room frame with strong adjacent dining geometric link -> dining room
        """
        if not photos:
            return

        ordered = sorted(photos, key=lambda p: p.position or 0)
        n = len(ordered)
        if n < 3:
            return

        exterior_like = {
            "patio",
            "front yard",
            "backyard",
            "exterior front",
            "exterior back",
            "pool",
        }
        interior_social = {"living room", "dining room", "kitchen", "entrance", "hallway"}
        service_like = {"laundry", "laundry room", "bathroom", "storage", "garage"}

        changed = 0

        # Rule 0: focused service-room relabel.
        # Uses richer CLIP prompts plus local context to separate:
        # bathroom vs laundry room vs storage vs garage.
        for idx, photo in enumerate(ordered):
            if photo.room_override:
                continue

            current = _normalize_room_label(photo.room_label)
            if current not in service_like:
                continue

            window = ordered[max(0, idx - 2): min(n, idx + 3)]
            neighbor_labels = [
                _normalize_room_label(p.room_label)
                for p in window
                if p.id != photo.id
            ]
            bathroom_votes = sum(label == "bathroom" for label in neighbor_labels)
            laundry_votes = sum(label in {"laundry", "laundry room"} for label in neighbor_labels)
            storage_votes = sum(label == "storage" for label in neighbor_labels)
            depth_var = float(photo.depth_variance or 0.0)
            has_laundry_hint = _looks_like_laundry(photo)
            image = self._get_cached_image(photo, image_cache or {})
            service_scores = _score_service_room_labels(image)
            bathroom_score = float(service_scores.get("bathroom", 0.0))
            laundry_score = float(service_scores.get("laundry room", 0.0))
            storage_score = float(service_scores.get("storage", 0.0))
            garage_score = float(service_scores.get("garage", 0.0))

            updated_label: str | None = None
            if garage_score >= max(storage_score, laundry_score, bathroom_score) + 0.02:
                updated_label = "garage"
            elif current in {"laundry", "laundry room"} and not has_laundry_hint:
                if bathroom_score >= laundry_score - 0.01 and (bathroom_votes >= 1 or bathroom_score >= storage_score + 0.01):
                    updated_label = "bathroom"
                elif storage_score >= laundry_score - 0.025:
                    updated_label = "storage"
            elif current == "bathroom" and not has_laundry_hint:
                if storage_score >= bathroom_score + 0.03:
                    updated_label = "storage"
            elif current == "storage" and has_laundry_hint and laundry_score >= storage_score - 0.01:
                updated_label = "laundry room"

            if updated_label and updated_label != photo.room_label:
                photo.room_label = updated_label
                changed += 1
                logger.info(
                    "Photo %s room relabel: %s -> %s (service-room focused relabel: scores bath=%.3f laundry=%.3f storage=%.3f garage=%.3f, votes bath=%s laundry=%s storage=%s, depth_var=%.3f, laundry_hint=%s)",
                    photo.id,
                    current,
                    updated_label,
                    bathroom_score,
                    laundry_score,
                    storage_score,
                    garage_score,
                    bathroom_votes,
                    laundry_votes,
                    storage_votes,
                    depth_var,
                    "yes" if has_laundry_hint else "no",
                )

        # Rule 1: recover interior kitchen shots mislabeled as patio/exterior.
        for idx, photo in enumerate(ordered):
            if photo.room_override:
                continue

            current = _normalize_room_label(photo.room_label)
            if current not in exterior_like:
                continue

            window = ordered[max(0, idx - 2): min(n, idx + 3)]
            neighbor_labels = [
                _normalize_room_label(p.room_label)
                for p in window
                if p.id != photo.id
            ]

            kitchen_votes = sum(label == "kitchen" for label in neighbor_labels)
            interior_votes = sum(label in interior_social for label in neighbor_labels)
            exterior_votes = sum(label in exterior_like for label in neighbor_labels)
            depth_var = float(photo.depth_variance or 0.0)

            if (
                kitchen_votes >= 2
                and interior_votes >= 3
                and exterior_votes == 0
                and depth_var <= 0.08
            ):
                photo.room_label = "kitchen"
                changed += 1
                logger.info(
                    "Photo %s room relabel: %s -> kitchen (interior sequence context: kitchen_votes=%s, interior_votes=%s, depth_var=%.3f)",
                    photo.id,
                    current,
                    kitchen_votes,
                    interior_votes,
                    depth_var,
                )

        # Rule 2: living->dining corrections with strong verified adjacent dining links.
        if job_id is not None and self.db is not None:
            strong_links = (
                self.db.query(PhotoRelation)
                .filter(PhotoRelation.job_id == job_id)
                .filter(PhotoRelation.is_connected.is_(True))
                .filter(PhotoRelation.relation_confidence.isnot(None))
                .filter(PhotoRelation.relation_confidence >= 0.52)
                .all()
            )

            photo_by_id = {p.id: p for p in photos}
            for sim in strong_links:
                left = photo_by_id.get(int(sim.photo_a_id))
                right = photo_by_id.get(int(sim.photo_b_id))
                if left is None or right is None:
                    continue

                for candidate, other in ((left, right), (right, left)):
                    if candidate.room_override:
                        continue
                    if _normalize_room_label(candidate.room_label) != "living room":
                        continue
                    if _normalize_room_label(other.room_label) != "dining room":
                        continue

                    cand_pos = int(candidate.position or 0)
                    other_pos = int(other.position or 0)
                    if abs(cand_pos - other_pos) > 1:
                        continue

                    photo_by_pos = {int(p.position or 0): p for p in ordered}
                    prev_photo = photo_by_pos.get(cand_pos - 1)
                    next_photo = photo_by_pos.get(cand_pos + 1)
                    adjacent_labels = {
                        _normalize_room_label(prev_photo.room_label) if prev_photo else "",
                        _normalize_room_label(next_photo.room_label) if next_photo else "",
                    }
                    if "kitchen" in adjacent_labels:
                        continue

                    candidate.room_label = "dining room"
                    changed += 1
                    logger.info(
                        "Photo %s room relabel: living room -> dining room (strong dining relation to %s, track_support=%.3f, confidence=%.3f)",
                        candidate.id,
                        other.id,
                        float(sim.track_support or 0.0),
                        float(sim.relation_confidence or 0.0),
                    )

        if changed:
            logger.info("Interior room-label postprocess changed %s photos", changed)


def _looks_like_laundry(photo: JobPhoto) -> bool:
    """Infer laundry room from filename/metadata hints."""
    hints = []
    if photo.filename:
        hints.append(photo.filename)
    if photo.manual_metadata:
        try:
            hints.append(json.dumps(photo.manual_metadata))
        except Exception:
            hints.append(str(photo.manual_metadata))

    corpus = " ".join(hints).lower()
    if not corpus:
        return False

    laundry_tokens = ("laundry", "washer", "dryer", "washing machine")
    return any(token in corpus for token in laundry_tokens)


def _normalize_room_label(room: str | None) -> str:
    return (room or "").strip().lower().replace("_", " ")


SERVICE_ROOM_LABEL_PROMPTS = {
    "bathroom": "bathroom with sink vanity mirror toilet shower",
    "laundry room": "laundry room with washer dryer laundry machine utility sink",
    "storage": "storage room or storage closet with shelves boxes household storage",
    "garage": "garage with parking storage tools concrete floor",
}


def _score_service_room_labels(image: Image.Image) -> Dict[str, float]:
    raw_scores = openclip_model.score_labels(
        image,
        list(SERVICE_ROOM_LABEL_PROMPTS.values()),
        prompt_templates=[
            "a real estate photo of a {label}",
            "an interior listing photo of a {label}",
        ],
    )
    return {
        canonical: float(raw_scores.get(prompt_label, 0.0))
        for canonical, prompt_label in SERVICE_ROOM_LABEL_PROMPTS.items()
    }


def _is_aerial_label(room: str) -> bool:
    return "aerial" in room or "drone" in room


def _is_back_context_label(room: str) -> bool:
    return any(token in room for token in ("backyard", "exterior back", "patio", "pool", "garden", "deck"))
