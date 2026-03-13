"""Room clustering using DINOv2 + LoFTR geometric verification.

Optimized 3-stage pipeline:
1. DINOv2 clustering: Group by visual similarity (fast, semantic)
2. LoFTR: Verify geometric overlap within clusters (accurate)
3. (Optional) COLMAP: SfM for clusters with 3+ images

This keeps most work in cheap Stage 1 while ensuring clusters contain
photos that can actually be interpolated.
"""
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageOps
from sklearn.cluster import DBSCAN
from sqlalchemy.orm import Session

from app.db.models import Job, JobPhoto, RoomCluster, TransitionSequence, TransitionSequenceStep

logger = logging.getLogger(__name__)

# Post-cluster shot caps (applied after geometry clustering, before RoomCluster rows).
MAIN_ROOM_SHOT_CAP = 4       # living/kitchen/dining
BEDROOM_SHOT_CAP = 2
BATHROOM_SHOT_CAP_STRONG = 2
BATHROOM_SHOT_CAP_WEAK = 1
BATHROOM_STRONG_SCORE_THRESHOLD = 0.55
ROOM_INSTANCE_POSITION_GAP = 10  # Same room name across nearby clusters; split only when far apart

# Try to import the optimized pipeline (DINOv2 + LoFTR).
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
    preloaded_images: Optional[Dict[int, Image.Image]] = None,
) -> List[RoomCluster]:
    """Cluster photos by room type with visual overlap detection.

    Uses optimized 3-stage pipeline when available:
    1. DINOv2 clustering: Group by visual similarity (fast)
    2. LoFTR: Verify geometric overlap (accurate)
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

    _reset_photo_clustering_state(active_photos)

    # Try optimized pipeline (DINOv2 + LoFTR)
    if USE_LEARNED_MATCHING and use_overlap_detection and s3_client:
        return _cluster_with_learned_matching(
            db,
            job,
            active_photos,
            s3_client,
            preloaded_images=preloaded_images,
        )

    # Fallback to original ORB-based pipeline
    return _cluster_with_orb(db, job, active_photos, s3_client, use_overlap_detection)


def _cluster_with_learned_matching(
    db: Session,
    job: Job,
    photos: List[JobPhoto],
    s3_client,
    preloaded_images: Optional[Dict[int, Image.Image]] = None,
) -> List[RoomCluster]:
    """Cluster using optimized DINOv2 + LoFTR pipeline."""
    logger.info("Using optimized DINOv2 + LoFTR pipeline")

    # Download all images and collect room labels
    images = []
    photo_ids = []
    room_labels = []  # Room labels for cross-room mismatch check
    photo_map = {}  # id -> JobPhoto

    for photo in photos:
        try:
            img = preloaded_images.get(photo.id) if preloaded_images is not None else None
            if img is None:
                img = s3_client.download_image(photo.s3_uri)
                if preloaded_images is not None:
                    preloaded_images[photo.id] = img
            try:
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass
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
    cluster_result = cluster_photos_optimized(
        images, photo_ids, s3_client,
        db_session=db, job_id=job.id,
        room_labels=room_labels,
        return_metadata=True,
    )
    if isinstance(cluster_result, tuple):
        cluster_id_lists, metadata = cluster_result
    else:
        cluster_id_lists = cluster_result
        metadata = {}
    duplicate_of_map: Dict[int, int] = metadata.get("duplicate_of_map", {})
    duplicates_dropped = bool(metadata.get("duplicates_dropped", False))
    transition_sequences = metadata.get("transition_sequences", [])

    # Mark duplicate metadata on JobPhoto records.
    for photo in photo_map.values():
        photo.is_duplicate = False
        photo.duplicate_of_photo_id = None
    for dup_id, canonical_id in duplicate_of_map.items():
        dup_photo = photo_map.get(dup_id)
        if dup_photo is None:
            continue
        dup_photo.is_duplicate = True
        dup_photo.duplicate_of_photo_id = canonical_id
        if duplicates_dropped:
            # Soft-delete from this job output: keep row for audit/debug, but remove
            # it from downstream clustering/video generation.
            dup_photo.exclude = True
            dup_photo.room_cluster_id = None
            dup_photo.cluster_order = None

    if duplicates_dropped and duplicate_of_map:
        logger.info("Excluded %s duplicate photos from downstream generation", len(duplicate_of_map))

    raw_cluster_photo_groups = []
    for cluster_ids in cluster_id_lists:
        cluster_photos = [photo_map[pid] for pid in cluster_ids if pid in photo_map]
        if cluster_photos:
            raw_cluster_photo_groups.append(cluster_photos)

    ordered_cluster_photo_groups = _order_clusters_for_story(
        raw_cluster_photo_groups,
        duplicate_of_map=duplicate_of_map,
    )
    ordered_cluster_photo_groups = _apply_post_cluster_shot_caps(ordered_cluster_photo_groups)

    # Create RoomCluster records
    clusters = []
    room_instance_tracker: Dict[str, Dict[str, Any]] = {}
    photo_to_cluster_id: Dict[int, int] = {}
    for sequence_order, cluster_photos in enumerate(ordered_cluster_photo_groups):
        if not cluster_photos:
            continue

        # Determine room type from majority vote
        room_type = _majority_room_label(cluster_photos)
        base_room_name = _room_instance_base_label(room_type)
        room_instance_label = _room_instance_label_for_cluster(
            base_room_name=base_room_name,
            cluster_photos=cluster_photos,
            tracker=room_instance_tracker,
        )

        cluster = RoomCluster(
            job_id=job.id,
            room_type=room_instance_label,
            image_count=len(cluster_photos),
            sequence_order=sequence_order,
        )
        db.add(cluster)
        db.flush()

        for photo_order, photo in enumerate(cluster_photos):
            photo.room_cluster_id = cluster.id
            photo.cluster_order = photo_order
            photo_to_cluster_id[int(photo.id)] = int(cluster.id)

        _compute_cluster_metrics(cluster, cluster_photos)
        clusters.append(cluster)

        logger.info(
            f"Created cluster {cluster.id}: {room_instance_label} "
            f"with {len(cluster_photos)} photos"
        )

    _persist_transition_sequences(
        db=db,
        job_id=job.id,
        sequences=transition_sequences,
        photo_to_cluster_id=photo_to_cluster_id,
        photo_map=photo_map,
    )

    db.commit()
    logger.info(f"Created {len(clusters)} room clusters")
    return clusters


def _persist_transition_sequences(
    db: Session,
    job_id: int,
    sequences: List[Dict[str, Any]],
    photo_to_cluster_id: Dict[int, int],
    photo_map: Dict[int, JobPhoto],
) -> None:
    db.query(TransitionSequence).filter(TransitionSequence.job_id == job_id).delete(synchronize_session=False)
    if not sequences:
        return

    for rank, sequence in enumerate(sequences):
        photo_ids = [int(photo_id) for photo_id in sequence.get("photo_ids", [])]
        if len(photo_ids) < 3:
            continue
        cluster_ids = sorted(
            {
                photo_to_cluster_id.get(photo_id)
                for photo_id in photo_ids
                if photo_to_cluster_id.get(photo_id) is not None
            }
        )
        room_type_hint = None
        for photo_id in photo_ids:
            photo = photo_map.get(photo_id)
            if photo is None:
                continue
            room_type_hint = photo.room_override or photo.room_label
            if room_type_hint:
                break
        record = TransitionSequence(
            job_id=job_id,
            sequence_rank=rank,
            sequence_score=float(sequence.get("sequence_score", 0.0) or 0.0),
            certification_status=str(sequence.get("certification_status") or "usable"),
            room_type_hint=room_type_hint,
            source_cluster_ids=cluster_ids,
            motion_hint=str(sequence.get("motion_hint") or "smooth_transition"),
        )
        db.add(record)
        db.flush()
        edge_lookup = {
            (
                min(int(edge["photo_a_id"]), int(edge["photo_b_id"])),
                max(int(edge["photo_a_id"]), int(edge["photo_b_id"])),
            ): edge
            for edge in sequence.get("edges", [])
            if isinstance(edge, dict)
        }
        for step_index, photo_id in enumerate(photo_ids):
            similarity_id = None
            if step_index > 0:
                left = min(photo_ids[step_index - 1], photo_id)
                right = max(photo_ids[step_index - 1], photo_id)
                edge = edge_lookup.get((left, right))
                similarity_id = int(edge["id"]) if edge and edge.get("id") is not None else None
            db.add(
                TransitionSequenceStep(
                    transition_sequence_id=record.id,
                    step_index=step_index,
                    photo_id=photo_id,
                    photo_similarity_id=similarity_id,
                )
            )


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

    raw_cluster_photo_groups = []
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

            for overlap_group in overlap_clusters:
                if overlap_group:
                    raw_cluster_photo_groups.append(overlap_group)

    ordered_cluster_photo_groups = _order_clusters_for_story(raw_cluster_photo_groups)
    ordered_cluster_photo_groups = _apply_post_cluster_shot_caps(ordered_cluster_photo_groups)
    room_instance_tracker: Dict[str, Dict[str, Any]] = {}
    for sequence_order, overlap_group in enumerate(ordered_cluster_photo_groups):
        room_type = _majority_room_label(overlap_group)
        base_room_name = _room_instance_base_label(room_type)
        room_instance_label = _room_instance_label_for_cluster(
            base_room_name=base_room_name,
            cluster_photos=overlap_group,
            tracker=room_instance_tracker,
        )

        cluster = RoomCluster(
            job_id=job.id,
            room_type=room_instance_label,
            image_count=len(overlap_group),
            sequence_order=sequence_order,
        )
        db.add(cluster)
        db.flush()

        for photo_order, photo in enumerate(overlap_group):
            photo.room_cluster_id = cluster.id
            photo.cluster_order = photo_order

        _compute_cluster_metrics(cluster, overlap_group)
        clusters.append(cluster)

        logger.info(
            f"Created cluster {cluster.id}: {room_instance_label} "
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
    room_instance_label = _room_instance_base_label(room_type)

    cluster = RoomCluster(
        job_id=job.id,
        room_type=room_instance_label,
        image_count=len(photos),
        sequence_order=0,
    )
    db.add(cluster)
    db.flush()

    for photo_order, photo in enumerate(photos):
        photo.room_cluster_id = cluster.id
        photo.cluster_order = photo_order

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


def _reset_photo_clustering_state(photos: List[JobPhoto]) -> None:
    """Reset cluster/duplicate state before recomputing clustering."""
    for photo in photos:
        photo.room_cluster_id = None
        photo.cluster_order = None
        photo.is_duplicate = False
        photo.duplicate_of_photo_id = None


def _normalize_room_label(room_type: str | None) -> str:
    return (room_type or "unknown").strip().lower().replace("_", " ")


def _room_bucket(room_type: str | None) -> str:
    room = _normalize_room_label(room_type)

    if "drone" in room:
        return "drone"
    if "aerial" in room:
        return "aerial"
    if "front yard" in room or "exterior front" in room:
        return "exterior_front"
    if any(token in room for token in ("entrance", "foyer")):
        return "entrance"
    if "living" in room:
        return "living_room"
    if "kitchen" in room:
        return "kitchen"
    if "dining" in room:
        return "dining_room"
    if "laundry" in room:
        return "laundry"
    if "bedroom" in room:
        return "bedroom"
    if "bathroom" in room:
        return "bathroom"
    if "office" in room:
        return "office"
    if "hallway" in room:
        return "hallway"
    if "garage" in room:
        return "garage"
    if "basement" in room:
        return "basement"
    if "attic" in room:
        return "attic"
    if any(token in room for token in ("pool", "spa")):
        return "pool"
    if "patio" in room or "deck" in room:
        return "patio"
    if "backyard" in room or "garden" in room or "yard" in room:
        return "backyard"
    if "exterior back" in room:
        return "exterior_back"
    if "exterior" in room and "front" in room:
        return "exterior_front"
    if "exterior" in room:
        return "exterior_other"
    return "unknown"


def _story_stage(bucket: str, is_duplicate_cluster: bool) -> int:
    """Lower stage index is earlier in the final video."""
    if is_duplicate_cluster:
        return 90
    if bucket == "drone":
        return 0
    if bucket in {"exterior_front", "entrance", "aerial"}:
        return 1
    if bucket in {"living_room", "kitchen", "dining_room"}:
        return 2
    if bucket in {"bedroom", "bathroom", "office", "laundry"}:
        return 3
    if bucket in {"hallway", "garage", "basement", "attic", "unknown"}:
        return 4
    if bucket in {"patio", "backyard", "pool", "exterior_back", "exterior_other"}:
        return 5
    return 6


def _bucket_priority(bucket: str) -> int:
    """Ordering inside each story stage."""
    priorities = {
        "drone": 0,
        "exterior_front": 10,
        "entrance": 20,
        "aerial": 30,
        "living_room": 40,
        "kitchen": 50,
        "dining_room": 60,
        "bedroom": 70,
        "service_room": 80,
        "bathroom": 80,
        "office": 90,
        "laundry": 100,
        "hallway": 110,
        "garage": 120,
        "basement": 130,
        "attic": 140,
        "patio": 150,
        "backyard": 160,
        "pool": 170,
        "exterior_back": 180,
        "exterior_other": 190,
        "unknown": 200,
    }
    return priorities.get(bucket, 999)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _majority_room_label(photos: List[JobPhoto]) -> str:
    labels = [p.room_override or p.room_label or "unknown" for p in photos]
    if not labels:
        return "unknown"

    counts: Dict[str, int] = defaultdict(int)
    first_index: Dict[str, int] = {}
    for idx, label in enumerate(labels):
        counts[label] += 1
        if label not in first_index:
            first_index[label] = idx

    # Deterministic tie-break: highest count, then first appearance in cluster order.
    return max(counts.keys(), key=lambda label: (counts[label], -first_index[label]))


def _room_instance_base_label(room_type: str | None) -> str:
    """Normalize room label for per-job room-instance naming."""
    bucket = _room_bucket(room_type)
    bucket_to_name = {
        "drone": "drone",
        "aerial": "aerial",
        "exterior_front": "front yard",
        "entrance": "entrance",
        "living_room": "living room",
        "kitchen": "kitchen",
        "dining_room": "dining room",
        "bedroom": "bedroom",
        "bathroom": "bathroom",
        "office": "office",
        "laundry": "laundry room",
        "hallway": "hallway",
        "garage": "garage",
        "basement": "basement",
        "attic": "attic",
        "patio": "patio",
        "backyard": "backyard",
        "pool": "pool",
        "exterior_back": "exterior back",
        "exterior_other": "exterior",
        "unknown": "unknown",
    }
    return bucket_to_name.get(bucket, _normalize_room_label(room_type) or "unknown")


def _cluster_position_center(photos: List[JobPhoto]) -> float:
    positions = [int(p.position or 0) for p in photos]
    if not positions:
        return 0.0
    return float(np.median(positions))


def _cluster_floor_tag(photos: List[JobPhoto]) -> str | None:
    floor_keys = (
        "floor",
        "floor_label",
        "floor_name",
        "level",
        "story",
        "storey",
    )
    for photo in photos:
        metadata = photo.manual_metadata if isinstance(photo.manual_metadata, dict) else {}
        for key in floor_keys:
            value = metadata.get(key)
            if value is None:
                continue
            tag = str(value).strip().lower()
            if tag:
                return tag
    return None


def _room_instance_label_for_cluster(
    base_room_name: str,
    cluster_photos: List[JobPhoto],
    tracker: Dict[str, Dict[str, Any]],
) -> str:
    state = tracker.get(base_room_name)
    pos_center = _cluster_position_center(cluster_photos)
    floor_tag = _cluster_floor_tag(cluster_photos)

    if state is None:
        state = {"index": 1, "last_pos_center": pos_center, "last_floor_tag": floor_tag}
        tracker[base_room_name] = state
    else:
        last_pos_center = float(state.get("last_pos_center", pos_center))
        last_floor_tag = state.get("last_floor_tag")

        is_new_instance = False
        if floor_tag and last_floor_tag and floor_tag != last_floor_tag:
            is_new_instance = True
        elif abs(pos_center - last_pos_center) > float(ROOM_INSTANCE_POSITION_GAP):
            is_new_instance = True

        if is_new_instance:
            state["index"] = int(state.get("index", 1)) + 1

        state["last_pos_center"] = pos_center
        if floor_tag:
            state["last_floor_tag"] = floor_tag

    instance_index = int(state.get("index", 1))
    if instance_index <= 1:
        return base_room_name
    return f"{base_room_name} {instance_index}"


def _bathroom_cluster_is_strong(photos: List[JobPhoto]) -> bool:
    scores = [float(p.final_score) for p in photos if p.final_score is not None]
    if not scores:
        return False
    return float(np.mean(scores)) >= float(BATHROOM_STRONG_SCORE_THRESHOLD)


def _room_shot_cap(room_type: str | None, photos: List[JobPhoto]) -> int:
    bucket = _room_bucket(room_type)
    if bucket in {"living_room", "kitchen", "dining_room"}:
        return int(MAIN_ROOM_SHOT_CAP)
    if bucket == "bedroom":
        return int(BEDROOM_SHOT_CAP)
    if bucket == "bathroom":
        return int(BATHROOM_SHOT_CAP_STRONG if _bathroom_cluster_is_strong(photos) else BATHROOM_SHOT_CAP_WEAK)
    return len(photos)


def _apply_post_cluster_shot_caps(
    cluster_photo_groups: List[List[JobPhoto]],
) -> List[List[JobPhoto]]:
    """Apply final shot-count caps after clustering is already decided.

    This does not re-cluster; it trims oversized groups for render sequencing.
    """
    capped_groups: List[List[JobPhoto]] = []
    trimmed_total = 0
    for photos in cluster_photo_groups:
        if not photos:
            continue
        room_labels = [p.room_override or p.room_label or "unknown" for p in photos]
        room_type = max(set(room_labels), key=room_labels.count)
        cap = _room_shot_cap(room_type, photos)
        if cap <= 0:
            continue
        if len(photos) <= cap:
            capped_groups.append(photos)
            continue

        trimmed = photos[:cap]
        capped_groups.append(trimmed)
        trimmed_total += (len(photos) - cap)
        logger.info(
            "Post-filter shot cap: room=%s size=%s -> %s",
            room_type,
            len(photos),
            len(trimmed),
        )

    if trimmed_total > 0:
        logger.info("Post-filter shot caps removed %s photos across clusters", trimmed_total)
    return capped_groups


def _master_priority(photo: JobPhoto) -> int:
    """Extract user-driven master-shot priority from manual metadata."""
    metadata = photo.manual_metadata if isinstance(photo.manual_metadata, dict) else {}
    if not metadata:
        return 0

    priority = 0
    for key in ("master_priority", "shot_priority", "priority", "video_priority"):
        value = _as_int(metadata.get(key))
        if value is not None:
            priority = max(priority, value)

    is_master = False
    for key in ("is_master", "master_shot", "master", "is_hero", "hero_shot"):
        if _as_bool(metadata.get(key)):
            is_master = True
            break

    tags = metadata.get("tags")
    if isinstance(tags, list):
        if any(str(tag).strip().lower() in {"master", "hero"} for tag in tags):
            is_master = True

    return max(priority, 1 if is_master else 0)


def _order_clusters_for_story(
    cluster_photo_groups: List[List[JobPhoto]],
    duplicate_of_map: Dict[int, int] | None = None,
) -> List[List[JobPhoto]]:
    """Order clusters in a real-estate story arc with upload-order continuity."""
    duplicate_of_map = duplicate_of_map or {}

    descriptors = []
    for photos in cluster_photo_groups:
        if not photos:
            continue

        room_labels = [p.room_override or p.room_label or "unknown" for p in photos]
        room_type = max(set(room_labels), key=room_labels.count)
        bucket = _room_bucket(room_type)
        story_bucket = "service_room" if bucket in {"bathroom", "laundry"} else bucket

        positions = [p.position or 0 for p in photos]
        pos_center = float(np.median(positions)) if positions else 0.0
        quality = float(np.mean([p.final_score or 0.0 for p in photos]))
        master_priority = max(_master_priority(p) for p in photos)
        is_duplicate_cluster = all(p.id in duplicate_of_map for p in photos)
        stage = _story_stage(bucket, is_duplicate_cluster)

        descriptors.append(
            {
                "photos": photos,
                "room_bucket": bucket,
                "story_bucket": story_bucket,
                "stage": stage,
                "pos_center": pos_center,
                "quality": quality,
                "master_priority": master_priority,
            }
        )

    grouped = defaultdict(list)
    for desc in descriptors:
        grouped[desc["stage"]].append(desc)

    ordered = []
    for stage in sorted(grouped.keys()):
        group = grouped[stage]
        masters = [d for d in group if d["master_priority"] > 0]
        normals = [d for d in group if d["master_priority"] == 0]

        masters.sort(
            key=lambda d: (
                -d["master_priority"],
                _bucket_priority(d["story_bucket"]),
                d["pos_center"],
                -d["quality"],
            )
        )

        normals_by_bucket = defaultdict(list)
        for desc in normals:
            normals_by_bucket[desc["story_bucket"]].append(desc)

        ordered_normals = []
        for bucket in sorted(normals_by_bucket.keys(), key=_bucket_priority):
            bucket_normals = normals_by_bucket[bucket]
            bucket_normals.sort(key=lambda d: (d["pos_center"], -d["quality"]))
            ordered_normals.extend(bucket_normals)

        ordered.extend(masters)
        ordered.extend(ordered_normals)

    return [desc["photos"] for desc in ordered]
