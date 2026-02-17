"""Room clustering using DINOv2 + learned feature matching.

Optimized 3-stage pipeline:
1. DINOv2 clustering: Group by visual similarity (fast, semantic)
2. LightGlue/LoFTR: Verify geometric overlap within clusters (accurate)
3. (Optional) COLMAP: SfM for clusters with 3+ images

This ensures clusters contain photos that can actually be interpolated
while minimizing compute (90% of work is in cheap Stage 1).
"""
import logging
from collections import defaultdict
from typing import List

import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from sqlalchemy.orm import Session

from app.db.models import Job, JobPhoto, RoomCluster

logger = logging.getLogger(__name__)

# Try to import the optimized pipeline (DINOv2 + LightGlue)
try:
    from app.pipeline.phase1_analyze.learned_matching import cluster_photos_optimized
    USE_LEARNED_MATCHING = True
except ImportError:
    USE_LEARNED_MATCHING = False

# Fallback imports (ORB-based)
from app.pipeline.phase1_analyze.overlap_detector import (
    compute_overlap_for_photos,
    cluster_by_overlap,
    order_photos_by_overlap,
)


def cluster_photos_by_room(
    db: Session,
    job: Job,
    photos: List[JobPhoto],
    s3_client=None,
    use_overlap_detection: bool = True,
) -> List[RoomCluster]:
    """Cluster photos by room type with visual overlap detection.

    Uses optimized 3-stage pipeline when available:
    1. DINOv2 clustering: Group by visual similarity (fast)
    2. LightGlue/LoFTR: Verify geometric overlap (accurate)
    3. (Optional) COLMAP: SfM for high-quality clusters

    Falls back to ORB-based clustering if learned models unavailable.

    Args:
        db: Database session
        job: Job instance
        photos: List of JobPhoto instances with embeddings and room labels
        s3_client: S3 client for downloading images (required for overlap detection)
        use_overlap_detection: Whether to use visual overlap detection

    Returns:
        List of created RoomCluster instances
    """
    # Filter out excluded photos and SORT BY POSITION
    # Position order is critical for temporal window matching (adjacent photos = same room)
    active_photos = sorted([p for p in photos if not p.exclude], key=lambda p: p.position or 0)
    logger.info(f"Clustering {len(active_photos)} photos for job {job.id}")

    if not active_photos:
        return []

    # Try optimized pipeline (DINOv2 + LightGlue)
    if USE_LEARNED_MATCHING and use_overlap_detection and s3_client:
        return _cluster_with_learned_matching(db, job, active_photos, s3_client)

    # Fallback to original ORB-based pipeline
    return _cluster_with_orb(db, job, active_photos, s3_client, use_overlap_detection)


def _cluster_with_learned_matching(
    db: Session,
    job: Job,
    photos: List[JobPhoto],
    s3_client,
) -> List[RoomCluster]:
    """Cluster using optimized DINOv2 + LightGlue pipeline."""
    logger.info("Using optimized DINOv2 + LightGlue pipeline")

    # Download all images and collect room labels
    images = []
    photo_ids = []
    room_labels = []  # Room labels for cross-room mismatch check
    photo_map = {}  # id -> JobPhoto

    for photo in photos:
        try:
            img = s3_client.download_image(photo.s3_uri)
            img = img.resize((512, 384), Image.Resampling.LANCZOS)
            images.append(img)
            photo_ids.append(photo.id)
            room_labels.append(photo.room_override or photo.room_label or "unknown")
            photo_map[photo.id] = photo
        except Exception as e:
            logger.warning(f"Failed to download photo {photo.id}: {e}")

    if len(images) < 2:
        # Not enough images for clustering
        return _create_single_cluster(db, job, photos)

    # Run optimized clustering (pass db_session and job_id to save similarity records)
    cluster_id_lists = cluster_photos_optimized(
        images, photo_ids, s3_client,
        db_session=db, job_id=job.id,
        room_labels=room_labels,
    )

    # Create RoomCluster records
    clusters = []
    for cluster_ids in cluster_id_lists:
        cluster_photos = [photo_map[pid] for pid in cluster_ids if pid in photo_map]

        if not cluster_photos:
            continue

        # Determine room type from majority vote
        room_labels = [p.room_override or p.room_label or "unknown" for p in cluster_photos]
        room_type = max(set(room_labels), key=room_labels.count)

        cluster = RoomCluster(
            job_id=job.id,
            room_type=room_type,
            image_count=len(cluster_photos),
        )
        db.add(cluster)
        db.flush()

        for photo in cluster_photos:
            photo.room_cluster_id = cluster.id

        _compute_cluster_metrics(cluster, cluster_photos)
        clusters.append(cluster)

        logger.info(
            f"Created cluster {cluster.id}: {room_type} "
            f"with {len(cluster_photos)} photos"
        )

    db.commit()
    logger.info(f"Created {len(clusters)} room clusters")
    return clusters


def _cluster_with_orb(
    db: Session,
    job: Job,
    photos: List[JobPhoto],
    s3_client,
    use_overlap_detection: bool,
) -> List[RoomCluster]:
    """Fallback clustering using ORB features."""
    logger.info("Using ORB-based clustering (fallback)")

    # Group photos by room label first
    room_groups = defaultdict(list)
    for photo in photos:
        room_label = photo.room_override or photo.room_label or "unknown"
        room_groups[room_label].append(photo)

    clusters = []

    for room_label, room_photos in room_groups.items():
        if len(room_photos) == 0:
            continue

        # Stage 2: Sub-cluster by embedding similarity (semantic)
        if len(room_photos) > 1 and all(p.embedding for p in room_photos):
            semantic_clusters = _sub_cluster_by_embedding(room_photos)
        else:
            semantic_clusters = [room_photos]

        # Stage 3: Further split by visual overlap
        for semantic_group in semantic_clusters:
            if use_overlap_detection and s3_client and len(semantic_group) > 1:
                overlap_clusters = _sub_cluster_by_overlap(semantic_group, s3_client)
            else:
                overlap_clusters = [semantic_group]

            # Create RoomCluster for each final cluster
            for overlap_group in overlap_clusters:
                cluster = RoomCluster(
                    job_id=job.id,
                    room_type=room_label,
                    image_count=len(overlap_group),
                )
                db.add(cluster)
                db.flush()

                for photo in overlap_group:
                    photo.room_cluster_id = cluster.id

                _compute_cluster_metrics(cluster, overlap_group)
                clusters.append(cluster)

                logger.info(
                    f"Created cluster {cluster.id}: {room_label} "
                    f"with {len(overlap_group)} photos"
                )

    db.commit()
    logger.info(f"Created {len(clusters)} room clusters")
    return clusters


def _create_single_cluster(
    db: Session,
    job: Job,
    photos: List[JobPhoto],
) -> List[RoomCluster]:
    """Create a single cluster for all photos (fallback)."""
    if not photos:
        return []

    room_type = photos[0].room_override or photos[0].room_label or "unknown"

    cluster = RoomCluster(
        job_id=job.id,
        room_type=room_type,
        image_count=len(photos),
    )
    db.add(cluster)
    db.flush()

    for photo in photos:
        photo.room_cluster_id = cluster.id

    _compute_cluster_metrics(cluster, photos)
    db.commit()

    return [cluster]


def _sub_cluster_by_embedding(photos: List[JobPhoto], eps: float = 0.3) -> List[List[JobPhoto]]:
    """Sub-cluster photos within same room type by embedding similarity.

    Args:
        photos: Photos with same room label
        eps: DBSCAN epsilon (distance threshold)

    Returns:
        List of photo groups (sub-clusters)
    """
    if len(photos) <= 1:
        return [photos]

    # Stack embeddings
    embeddings = np.array([p.embedding for p in photos])

    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Use DBSCAN for clustering (handles variable cluster count)
    clustering = DBSCAN(eps=eps, min_samples=1, metric="cosine")
    labels = clustering.fit_predict(embeddings)

    # Group photos by cluster label
    groups = defaultdict(list)
    for photo, label in zip(photos, labels):
        groups[label].append(photo)

    return list(groups.values())


def _sub_cluster_by_overlap(photos: List[JobPhoto], s3_client) -> List[List[JobPhoto]]:
    """Sub-cluster photos by visual overlap (shared pixels/keypoints).

    Uses ORB feature matching to detect which photos share visual content.
    Photos that don't overlap are split into separate clusters.

    Args:
        photos: Photos to cluster
        s3_client: S3 client for downloading images

    Returns:
        List of photo groups where each group has visual overlap
    """
    if len(photos) <= 1:
        return [photos]

    try:
        # Compute pairwise overlap matrix
        overlap_matrix, _ = compute_overlap_for_photos(photos, s3_client)

        # Cluster by overlap connectivity
        overlap_clusters = cluster_by_overlap(photos, overlap_matrix)

        # Order photos within each cluster for optimal interpolation
        ordered_clusters = []
        for cluster_photos in overlap_clusters:
            if len(cluster_photos) > 2:
                # Get subset of overlap matrix for this cluster
                indices = [photos.index(p) for p in cluster_photos]
                sub_matrix = overlap_matrix[np.ix_(indices, indices)]
                ordered = order_photos_by_overlap(cluster_photos, sub_matrix)
                ordered_clusters.append(ordered)
            else:
                ordered_clusters.append(cluster_photos)

        return ordered_clusters

    except Exception as e:
        logger.warning(f"Overlap detection failed, using single cluster: {e}")
        return [photos]


def _compute_cluster_metrics(cluster: RoomCluster, photos: List[JobPhoto]) -> None:
    """Compute aggregate metrics for a room cluster.

    Args:
        cluster: RoomCluster to update
        photos: Photos in the cluster
    """
    # Compute average depth variance
    depth_variances = [p.depth_variance for p in photos if p.depth_variance is not None]
    if depth_variances:
        cluster.depth_variance = float(np.mean(depth_variances))

    # Determine confidence tier based on depth
    # Thresholds tuned for real estate photos:
    # - Indoor rooms typically have 0.03-0.06 variance
    # - Outdoor/aerial typically have 0.06-0.12 variance
    if cluster.depth_variance is not None:
        if cluster.depth_variance > 0.06:
            cluster.confidence_tier = "high"
        elif cluster.depth_variance > 0.035:
            cluster.confidence_tier = "medium"
        else:
            cluster.confidence_tier = "low"
    else:
        cluster.confidence_tier = "low"

    # Check SFM eligibility
    # Requirements:
    # - 3+ photos (need multiple views for any 3D effect)
    # - At least medium confidence (some depth variation)
    # Even with medium tier, we can do partial reveals and parallax effects
    cluster.sfm_eligible = len(photos) >= 3 and cluster.confidence_tier in ("high", "medium")
