"""Room clustering using OpenCLIP embeddings."""
import logging
from collections import defaultdict
from typing import List

import numpy as np
from sklearn.cluster import DBSCAN
from sqlalchemy.orm import Session

from app.db.models import Job, JobPhoto, RoomCluster

logger = logging.getLogger(__name__)


def cluster_photos_by_room(
    db: Session,
    job: Job,
    photos: List[JobPhoto],
) -> List[RoomCluster]:
    """Cluster photos by room type.

    Uses a combination of:
    1. Room label from AI classification
    2. Embedding similarity for photos with same room label

    Args:
        db: Database session
        job: Job instance
        photos: List of JobPhoto instances with embeddings and room labels

    Returns:
        List of created RoomCluster instances
    """
    logger.info(f"Clustering {len(photos)} photos for job {job.id}")

    # Group photos by room label first
    room_groups = defaultdict(list)
    for photo in photos:
        if photo.exclude:
            continue
        room_label = photo.room_override or photo.room_label or "unknown"
        room_groups[room_label].append(photo)

    clusters = []

    for room_label, room_photos in room_groups.items():
        if len(room_photos) == 0:
            continue

        # For rooms with multiple photos, sub-cluster by embedding similarity
        if len(room_photos) > 1 and all(p.embedding for p in room_photos):
            sub_clusters = _sub_cluster_by_embedding(room_photos)
        else:
            sub_clusters = [room_photos]

        # Create RoomCluster for each sub-cluster
        for i, sub_photos in enumerate(sub_clusters):
            cluster = RoomCluster(
                job_id=job.id,
                room_type=room_label,
                image_count=len(sub_photos),
            )
            db.add(cluster)
            db.flush()  # Get cluster ID

            # Assign photos to cluster
            for photo in sub_photos:
                photo.room_cluster_id = cluster.id

            # Compute cluster-level depth metrics
            _compute_cluster_metrics(cluster, sub_photos)

            clusters.append(cluster)
            logger.info(f"Created cluster {cluster.id}: {room_label} with {len(sub_photos)} photos")

    db.commit()
    logger.info(f"Created {len(clusters)} room clusters")
    return clusters


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
