"""Optimized 3-stage clustering pipeline for real estate photos.

Stage 1 (cheap): DINOv2 embedding clustering
- Group photos by visual similarity
- Keep clusters small (5-20 images)
- Fast: one forward pass per image

Stage 2 (medium): SuperPoint + LightGlue within clusters
- Only match within each DINOv2 cluster
- Build overlap graph
- Medium cost: N*(N-1)/2 matches per cluster

Stage 3 (selective SfM): Optional COLMAP
- Only for clusters with 3+ images
- Only if strong geometric consistency
- Expensive: full SfM reconstruction

This keeps 90% of compute small while getting high-quality results.
"""
import logging
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image

from app.core.config import settings

if TYPE_CHECKING:
    from app.db.models import JobPhoto

logger = logging.getLogger(__name__)

# ============================================================================
# STAGE 1: DINOv2 Embeddings
# ============================================================================

# DINOv2 model singleton
_dinov2_model = None
_dinov2_transform = None


def _load_dinov2():
    """Load DINOv2 model (lazy initialization)."""
    global _dinov2_model, _dinov2_transform

    if _dinov2_model is not None:
        return _dinov2_model, _dinov2_transform

    try:
        # Use transformers for DINOv2
        from transformers import AutoImageProcessor, AutoModel

        model_name = "facebook/dinov2-base"
        _dinov2_transform = AutoImageProcessor.from_pretrained(model_name)
        _dinov2_model = AutoModel.from_pretrained(model_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _dinov2_model = _dinov2_model.to(device)
        _dinov2_model.eval()

        logger.info(f"Loaded DINOv2 model on {device}")
        return _dinov2_model, _dinov2_transform

    except Exception as e:
        logger.error(f"Failed to load DINOv2: {e}")
        return None, None


def compute_dinov2_embeddings(images: List[Image.Image]) -> np.ndarray:
    """Compute DINOv2 embeddings for a list of images.

    Args:
        images: List of PIL Images

    Returns:
        NxD array of embeddings (D=768 for dinov2-base)
    """
    model, transform = _load_dinov2()

    if model is None:
        # Random embeddings make clustering non-deterministic and unreliable.
        raise RuntimeError(
            "DINOv2 model is unavailable. Refusing random embedding fallback "
            "because it causes inconsistent clustering."
        )

    device = next(model.parameters()).device
    embeddings = []

    with torch.no_grad():
        for img in images:
            # Ensure RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Process image
            inputs = transform(img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get CLS token embedding
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(embedding[0])

    return np.array(embeddings, dtype=np.float32)


def cluster_by_dinov2(
    images: List[Image.Image],
    photo_ids: List[int],
    max_cluster_size: int = 15,
    min_cluster_size: int = 2,
) -> List[List[int]]:
    """Cluster images using DINOv2 embeddings.

    Uses HDBSCAN for robust clustering that automatically determines
    the number of clusters.

    Args:
        images: List of PIL Images
        photo_ids: List of photo IDs
        max_cluster_size: Maximum images per cluster
        min_cluster_size: Minimum images per cluster

    Returns:
        List of photo ID lists (clusters)
    """
    from sklearn.cluster import HDBSCAN
    from sklearn.preprocessing import normalize

    n = len(images)
    if n <= 2:
        return [photo_ids]

    # Compute embeddings
    logger.info(f"Computing DINOv2 embeddings for {n} images...")
    embeddings = compute_dinov2_embeddings(images)

    # Normalize for cosine distance
    embeddings = normalize(embeddings)

    # HDBSCAN clustering
    # min_cluster_size=2 to catch even small overlapping groups
    # min_samples=1 for single-linkage-like behavior
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean",  # On normalized vectors = cosine
        cluster_selection_method="eom",
    )

    labels = clusterer.fit_predict(embeddings)

    # Group by label
    clusters_dict = defaultdict(list)
    noise_photos = []

    for i, label in enumerate(labels):
        if label == -1:
            # Noise point - will be its own cluster
            noise_photos.append(photo_ids[i])
        else:
            clusters_dict[label].append(photo_ids[i])

    # Convert to list
    clusters = list(clusters_dict.values())

    # Add noise points as singleton clusters
    for photo_id in noise_photos:
        clusters.append([photo_id])

    # Split large clusters
    final_clusters = []
    for cluster in clusters:
        if len(cluster) > max_cluster_size:
            # Split into smaller chunks
            for i in range(0, len(cluster), max_cluster_size):
                chunk = cluster[i:i + max_cluster_size]
                if len(chunk) >= min_cluster_size:
                    final_clusters.append(chunk)
                elif final_clusters:
                    # Add small remainder to previous
                    final_clusters[-1].extend(chunk)
                else:
                    final_clusters.append(chunk)
        else:
            final_clusters.append(cluster)

    logger.info(f"DINOv2 clustering: {n} images -> {len(final_clusters)} clusters")
    for i, cluster in enumerate(final_clusters):
        logger.debug(f"  Cluster {i}: {len(cluster)} images")

    return final_clusters


# ============================================================================
# STAGE 2: SuperPoint + LightGlue (within clusters)
# ============================================================================

# Matching thresholds - tuned for real estate photo transitions
# Goal: Group photos of SAME ROOM (even different angles) for smooth transitions
MIN_MATCHES_FOR_OVERLAP = 10      # Minimum matches before geometric verification
MIN_INLIERS_FOR_OVERLAP = 15      # Minimum inliers for pixel-level overlap (raised from 6 to avoid weak false positives)
MIN_INLIERS_FOR_DIRECTION = 3     # Lower threshold - just enough to compute direction vector
RANSAC_REPROJ_THRESHOLD = 1.5     # Tight reprojection threshold (pixels)
OVERLAP_THRESHOLD = 0.15          # Minimum score to connect photos

# Room label mismatch - only skip non-adjacent photos with different rooms
# For ADJACENT photos (temporal_dist=1), trust geometry even if room labels differ
# (ML room labels are often wrong, but adjacent photos are usually same room)
MIN_INLIERS_CROSS_ROOM = 30       # Require 30+ inliers for non-adjacent cross-room pairs
MIN_INLIERS_CROSS_ROOM_ADJACENT = 15  # Lower threshold for adjacent photos (ML often mislabels)

# Position gap thresholds - photos far apart in sequence need stronger evidence
# (prevents clustering photos from different physical locations with same room label)
POSITION_GAP_THRESHOLD = 3         # If gap >= 3, require higher inlier count
MIN_INLIERS_FAR_APART = 25         # Require 25+ inliers for non-adjacent same-room photos


def rooms_are_different(room1: str, room2: str) -> bool:
    """Check if two room labels represent different rooms.

    Returns True if the rooms are clearly different types.
    Returns False if they're the same or unknown.
    """
    if room1 is None or room2 is None:
        return False
    if room1 == "unknown" or room2 == "unknown":
        return False
    if room1 == room2:
        return False

    # Normalize labels
    r1 = room1.lower().strip()
    r2 = room2.lower().strip()

    # Same label after normalization
    if r1 == r2:
        return False

    # Check for similar room types that should be allowed to connect
    # (e.g., "living_room" and "living room" are the same)
    r1_words = set(r1.replace("_", " ").split())
    r2_words = set(r2.replace("_", " ").split())

    # If they share key room words, they might be same room with slight labeling difference
    key_words = {"living", "dining", "kitchen", "bedroom", "bathroom", "patio", "exterior", "outdoor", "yard", "pool"}
    r1_keys = r1_words & key_words
    r2_keys = r2_words & key_words

    # If they share key words, they're likely the same room type
    if r1_keys and r2_keys and r1_keys == r2_keys:
        return False

    # Otherwise they're different room types
    return True

# Temporal + semantic matching for same-room different-angle shots
# Adjacent photos in upload order are usually the same room
TEMPORAL_WINDOW = 2               # Check photos within ±2 positions
TEMPORAL_SEMANTIC_THRESHOLD = 0.88  # High confidence same room (balanced - not too strict)
TEMPORAL_GEOMETRIC_THRESHOLD = 0.60  # Lower threshold if we also verify geometrically
NEIGHBOR_TRUST_THRESHOLD = 0.63   # Trust immediate neighbors with strong semantic support
HIGH_CONFIDENCE_NEIGHBOR_TRUST = 0.83  # Very strong adjacent semantic support (can pass without extra context)
NEIGHBOR_SUPPORT_THRESHOLD = 0.70  # Support from nearby photos for semantic-only adjacent fallback
SAME_LABEL_NEIGHBOR_TRUST = 0.70  # Adjacent pairs with same room label can pass with moderate semantic
AMBIGUOUS_SAME_LABEL_TRUST = 0.84  # Stricter threshold for ambiguous repeated rooms
DIST2_TRUST_THRESHOLD = 0.75      # Conservative trust threshold for temporal distance=2 pairs
SAME_LABEL_DIST2_TRUST = 0.70     # Dist-2 same-label fallback to reduce over-splitting
AMBIGUOUS_DIST2_TRUST = 0.82      # Keep stricter dist-2 fallback for ambiguous rooms
CROSS_ROOM_RECOVERY_THRESHOLD = 0.72  # Adjacent cross-room fallback for known room-family confusions
SERVICE_ROOM_RECOVERY_THRESHOLD = 0.66  # Slightly lower for bathroom<->laundry confusions
CROSS_ROOM_RECOVERY_TOPK = 2       # Require mutual top-K semantic affinity for cross-room fallback
EXTERIOR_LONG_GAP_SEMANTIC_TRUST = 0.78  # Recover long-gap exterior pairs when geometry fails
EXTERIOR_LONG_GAP_MIN = 5          # Minimum sequence gap for long-gap exterior semantic recovery

# ORB pre-filter thresholds (for performance - ORB is ~10x faster than LoFTR)
ORB_QUICK_REJECT_INLIERS = 2      # If ORB finds ≤2 inliers, skip LoFTR (definitely no overlap)
ORB_QUICK_ACCEPT_INLIERS = 15     # If ORB finds ≥15 inliers, skip LoFTR (definitely overlap)

# Minimum DINOv2 similarity to even consider geometric verification
# Filters out obviously unrelated photos (aerial vs interior = ~0.05)
MIN_SEMANTIC_FOR_GEOMETRIC = 0.15  # Skip geometric check if semantic < 15%

# Cluster ordering thresholds
MIN_TRANSITION_SCORE = 0.20  # Minimum overlap score to keep photo in ordered chain
DIRECTION_CONSISTENCY_THRESHOLD = 0.5  # Cos similarity for direction vectors to be "consistent"
SEMANTIC_BRIDGE_SUPPORT_THRESHOLD = 0.70  # Remove weak semantic-only bridges if cross-support is low
HARD_TRANSITION_MIN_OVERLAP = 0.25  # Require at least ~25% overlap-like score for final transitions
HARD_TRANSITION_MIN_INLIERS = 8  # Reject transition edges without enough geometric correspondences
HARD_TRANSITION_ADJACENT_SEMANTIC_TRUST = NEIGHBOR_TRUST_THRESHOLD  # Keep trusted adjacent semantic links
HARD_TRANSITION_DIST2_SEMANTIC_TRUST = DIST2_TRUST_THRESHOLD  # Keep trusted distance-2 semantic links
HARD_TRANSITION_MAX_SEMANTIC_GAP = 2  # Split semantic-only transitions that jump too far in capture order
HARD_TRANSITION_RECOVERY_SEMANTIC_TRUST = 0.86  # semantic-recovery edges require much stronger confidence
HARD_TRANSITION_FRONT_RECOVERY_SEMANTIC_TRUST = 0.78  # Keep strong long-gap front-exterior recovery links
HARD_TRANSITION_REQUIRE_GEOMETRY = settings.REQUIRE_GEOMETRIC_TRANSITIONS
HARD_TRANSITION_REQUIRE_DIRECTION = settings.REQUIRE_DIRECTION_FOR_TRANSITIONS

# Deduplication thresholds
# Only consecutive photos with very high similarity are considered "same angle" duplicates
# Different angles of the same room typically have 0.85-0.92 similarity
DUPLICATE_SIMILARITY_THRESHOLD = 0.97  # Conservative dedup to avoid dropping transition-valuable near-duplicates
DUPLICATE_GEOMETRIC_SEMANTIC_THRESHOLD = 0.95  # Allow slightly lower semantic threshold when geometry is near-identical
DUPLICATE_GEOMETRIC_OVERLAP_THRESHOLD = 0.90  # Require very high overlap score for geometric-backed dedupe
KEEP_DUPLICATE_SINGLETON_CLUSTERS = not settings.DELETE_OBVIOUS_DUPLICATES


def compute_direction_vector(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    inlier_mask: np.ndarray,
) -> Tuple[float, float]:
    """Compute camera motion direction from matched keypoints.

    Returns the centroid shift vector (dx, dy) indicating how content
    moved from image 0 to image 1.

    If content shifted LEFT (negative dx), camera moved RIGHT.
    If content shifted UP (negative dy), camera moved DOWN.

    Returns:
        (dx, dy) normalized direction vector, or (0, 0) if insufficient matches
    """
    if inlier_mask is None or inlier_mask.sum() < 4:
        return (0.0, 0.0)

    # Get inlier points only
    inliers = inlier_mask.ravel().astype(bool)
    pts0 = mkpts0[inliers]
    pts1 = mkpts1[inliers]

    # Compute centroids
    centroid0 = pts0.mean(axis=0)
    centroid1 = pts1.mean(axis=0)

    # Direction = how content shifted from img0 to img1
    dx = centroid1[0] - centroid0[0]
    dy = centroid1[1] - centroid0[1]

    # Normalize
    magnitude = np.sqrt(dx * dx + dy * dy)
    if magnitude < 1e-6:
        return (0.0, 0.0)

    return (dx / magnitude, dy / magnitude)


def directions_consistent(dir1: Tuple[float, float], dir2: Tuple[float, float]) -> bool:
    """Check if two direction vectors are roughly consistent.

    Consistent means they point in similar or same direction (not opposite).
    Uses cosine similarity: > 0 means same general direction.
    """
    if dir1 == (0.0, 0.0) or dir2 == (0.0, 0.0):
        return True  # Unknown direction - assume OK

    cos_sim = dir1[0] * dir2[0] + dir1[1] * dir2[1]
    return cos_sim > DIRECTION_CONSISTENCY_THRESHOLD


def has_local_semantic_support(i: int, j: int, similarity: np.ndarray) -> bool:
    """Check whether an adjacent pair has local sequence context support.

    Semantic-only fallback is safer when at least one nearby photo also has
    reasonably high similarity with one of the pair endpoints.
    """
    n = similarity.shape[0]
    neighbor_indices = {i - 1, i + 1, j - 1, j + 1}
    support_scores = []
    for k in neighbor_indices:
        if k < 0 or k >= n or k == i or k == j:
            continue
        support_scores.append(max(float(similarity[i, k]), float(similarity[j, k])))

    return bool(support_scores) and max(support_scores) >= NEIGHBOR_SUPPORT_THRESHOLD


def normalize_room_label(room: str | None) -> str:
    return (room or "").strip().lower().replace("_", " ")


def room_family(room: str | None) -> str:
    room_norm = normalize_room_label(room)
    if not room_norm or room_norm == "unknown":
        return "unknown"

    if any(token in room_norm for token in ("living", "dining", "kitchen", "entrance", "foyer", "hallway")):
        return "social"
    if any(token in room_norm for token in ("bathroom", "laundry", "powder")):
        return "service"
    if any(token in room_norm for token in ("front yard", "exterior front", "driveway", "curb", "porch")):
        return "front_exterior"
    if any(token in room_norm for token in ("patio", "deck", "backyard", "garden", "pool", "exterior")):
        return "exterior"
    return room_norm


def rooms_allow_adjacent_semantic_bridge(room1: str | None, room2: str | None) -> bool:
    fam1 = room_family(room1)
    fam2 = room_family(room2)
    if fam1 == "unknown" or fam2 == "unknown":
        return False
    if fam1 != fam2:
        return False
    return fam1 in {"social", "service", "front_exterior"}


def is_exterior_like(room: str | None) -> bool:
    return room_family(room) in {"front_exterior", "exterior"}


def is_mutual_top_semantic_neighbor(i: int, j: int, similarity: np.ndarray, top_k: int = 2) -> bool:
    """Require mutual top semantic affinity to avoid weak cross-room fallbacks."""
    sorted_i = [idx for idx in np.argsort(-similarity[i]) if idx != i][:top_k]
    sorted_j = [idx for idx in np.argsort(-similarity[j]) if idx != j][:top_k]
    return (j in sorted_i) and (i in sorted_j)


def order_cluster_for_transitions(
    cluster_indices: List[int],
    photo_ids: List[int],
    adjacency: np.ndarray,
    directions: dict,
    min_score: float = MIN_TRANSITION_SCORE,
) -> Tuple[List[int], List[int]]:
    """Order photos within a cluster for optimal video transitions.

    Uses direction-aware path-finding to create a chain where:
    1. Each consecutive pair has good visual overlap
    2. Camera motion direction is consistent (no sudden reversals)
    3. Starts from an ENDPOINT (leftmost/rightmost in spatial order)

    Args:
        cluster_indices: Indices into photo_ids for this cluster
        photo_ids: Full list of photo IDs
        adjacency: Full NxN adjacency matrix (overlap scores)
        directions: Dict of (i,j) -> (dx, dy) direction vectors
        min_score: Minimum overlap score for valid transition

    Returns:
        Tuple of (ordered_main, isolated):
        - ordered_main: Ordered list of photo IDs for the main path
        - isolated: List of photo IDs not connected to main path
    """
    if len(cluster_indices) <= 1:
        return ([photo_ids[i] for i in cluster_indices], [])

    if len(cluster_indices) == 2:
        return ([photo_ids[i] for i in cluster_indices], [])

    n = len(cluster_indices)

    # Find ENDPOINT to start from (not the center!)
    # An endpoint has directions pointing mostly one way (it's at an edge of the scene)
    # A center photo has directions pointing both ways
    endpoint_scores = np.zeros(n)

    for i, idx_i in enumerate(cluster_indices):
        # Collect all direction vectors FROM this photo TO its neighbors
        outgoing_directions = []
        for j, idx_j in enumerate(cluster_indices):
            if i == j:
                continue
            if adjacency[idx_i, idx_j] < min_score:
                continue

            pair_key = (min(idx_i, idx_j), max(idx_i, idx_j))
            pair_dir = directions.get(pair_key, (0.0, 0.0))

            # Flip if we stored direction for the reverse pair
            if idx_i > idx_j:
                pair_dir = (-pair_dir[0], -pair_dir[1])

            if pair_dir != (0.0, 0.0):
                outgoing_directions.append(pair_dir)

        if len(outgoing_directions) >= 1:
            # Calculate how consistent the outgoing directions are
            # Endpoints have all directions pointing the same way
            # Centers have directions pointing opposite ways (cancel out)
            avg_dx = sum(d[0] for d in outgoing_directions) / len(outgoing_directions)
            avg_dy = sum(d[1] for d in outgoing_directions) / len(outgoing_directions)
            # Magnitude of average = consistency (1.0 = all same direction, 0.0 = cancel out)
            consistency = np.sqrt(avg_dx * avg_dx + avg_dy * avg_dy)
            endpoint_scores[i] = consistency
        else:
            # No direction info - use connection count as fallback
            endpoint_scores[i] = 0.5

    # Start from the best endpoint (highest consistency = most "edge-like")
    # If tie, prefer photo with fewer connections (more likely to be endpoint)
    connection_counts = np.zeros(n)
    for i, idx_i in enumerate(cluster_indices):
        for j, idx_j in enumerate(cluster_indices):
            if i != j and adjacency[idx_i, idx_j] >= min_score:
                connection_counts[i] += 1

    # Combine: high endpoint score, low connection count
    # Normalize and combine
    if endpoint_scores.max() > 0:
        endpoint_scores_norm = endpoint_scores / endpoint_scores.max()
    else:
        endpoint_scores_norm = endpoint_scores

    if connection_counts.max() > 0:
        connection_penalty = connection_counts / connection_counts.max()
    else:
        connection_penalty = connection_counts

    combined_score = endpoint_scores_norm - 0.3 * connection_penalty
    start_local = int(np.argmax(combined_score))

    endpoint_info = [(photo_ids[cluster_indices[i]], round(endpoint_scores[i], 2)) for i in range(n)]
    logger.debug(f"Endpoint scores: {endpoint_info}")
    logger.debug(f"Starting from endpoint: {photo_ids[cluster_indices[start_local]]}")

    # Greedy path building with direction awareness
    ordered_local = [start_local]
    remaining = set(range(n)) - {start_local}
    current_direction = (0.0, 0.0)  # Unknown initially

    while remaining:
        current_local = ordered_local[-1]
        current_idx = cluster_indices[current_local]

        # Find best next photo considering overlap AND direction
        best_next = None
        best_score = -1
        best_direction = (0.0, 0.0)

        for candidate_local in remaining:
            candidate_idx = cluster_indices[candidate_local]
            score = adjacency[current_idx, candidate_idx]

            if score < min_score:
                continue

            # Get direction for this transition
            pair_key = (min(current_idx, candidate_idx), max(current_idx, candidate_idx))
            pair_dir = directions.get(pair_key, (0.0, 0.0))

            # Flip direction if we're going in reverse order
            if current_idx > candidate_idx:
                pair_dir = (-pair_dir[0], -pair_dir[1])

            # Check direction consistency
            if current_direction != (0.0, 0.0) and pair_dir != (0.0, 0.0):
                if not directions_consistent(current_direction, pair_dir):
                    # Direction reversal - penalize score
                    score *= 0.5
                    logger.debug(f"Direction reversal penalty: {photo_ids[current_idx]}->{photo_ids[candidate_idx]}")

            if score > best_score:
                best_score = score
                best_next = candidate_local
                best_direction = pair_dir

        if best_next is None:
            # No good connection - try extending from the other end
            if len(ordered_local) > 1:
                first_local = ordered_local[0]
                first_idx = cluster_indices[first_local]

                for candidate_local in remaining:
                    candidate_idx = cluster_indices[candidate_local]
                    score = adjacency[first_idx, candidate_idx]

                    if score > best_score and score >= min_score:
                        best_score = score
                        best_next = candidate_local
                        # Will prepend, so this becomes the new first

                if best_next is not None:
                    ordered_local.insert(0, best_next)
                    remaining.remove(best_next)
                    continue

            # Fallback: insert by strongest connection to ANY point in the chain.
            # This prevents unnecessary singleton isolation in connected components
            # when endpoint extension gets stuck.
            insert_candidate = None
            insert_anchor = None
            insert_score = -1.0
            for candidate_local in remaining:
                candidate_idx = cluster_indices[candidate_local]
                for anchor_local in ordered_local:
                    anchor_idx = cluster_indices[anchor_local]
                    score = float(adjacency[anchor_idx, candidate_idx])
                    if score > insert_score:
                        insert_score = score
                        insert_candidate = candidate_local
                        insert_anchor = anchor_local

            relaxed_threshold = min_score * 0.8
            if (
                insert_candidate is not None
                and insert_anchor is not None
                and insert_score >= relaxed_threshold
            ):
                anchor_pos = ordered_local.index(insert_anchor)
                if anchor_pos <= len(ordered_local) // 2:
                    ordered_local.insert(0, insert_candidate)
                else:
                    ordered_local.append(insert_candidate)
                remaining.remove(insert_candidate)
                logger.debug(
                    "Inserted non-endpoint candidate %s with relaxed score %.3f",
                    photo_ids[cluster_indices[insert_candidate]],
                    insert_score,
                )
                continue

            # Still no good connection - these are isolated photos
            # Don't drop them - they'll be handled as separate mini-clusters
            logger.info(f"Cannot connect {len(remaining)} remaining photos - will create separate clusters")
            break

        ordered_local.append(best_next)
        remaining.remove(best_next)
        current_direction = best_direction

    # Convert back to photo IDs
    ordered_photo_ids = [photo_ids[cluster_indices[i]] for i in ordered_local]

    # Also return remaining photos as separate singleton clusters
    # They couldn't be ordered but they're still part of this connected component
    remaining_photo_ids = [photo_ids[cluster_indices[i]] for i in remaining]

    if remaining_photo_ids:
        logger.info(f"Cluster ordering: main chain={len(ordered_photo_ids)}, isolated={len(remaining_photo_ids)}")
        # Return as tuple: (ordered_main_chain, list_of_isolated_photos)
        return (ordered_photo_ids, remaining_photo_ids)

    return (ordered_photo_ids, [])


def deduplicate_and_split_cluster(
    cluster_photo_ids: List[int],
    photo_ids: List[int],
    embeddings: np.ndarray,
    adjacency: np.ndarray,
    max_size: int = 3,
    keep_duplicate_singletons: bool = True,
) -> Tuple[List[List[int]], Dict[int, int]]:
    """Remove duplicates and split large clusters into smaller ones.

    Photos are ALREADY ORDERED for optimal transitions. This function:
    1. Identifies duplicates (same angle shots) based on semantic similarity
    2. Removes duplicates that contribute LESS to transition smoothness
    3. If still > max_size, SPLITS into multiple clusters instead of dropping

    Key insight: If we have 6 good photos, we should make 2 clusters of 3,
    not drop 3 photos. This preserves content while keeping clusters manageable.

    Args:
        cluster_photo_ids: Photo IDs in this cluster (already ordered for transitions)
        photo_ids: Full list of all photo IDs (for indexing)
        embeddings: NxD normalized embedding matrix for all photos
        adjacency: NxN geometric overlap matrix (higher = better overlap)
        max_size: Maximum photos per cluster
        keep_duplicate_singletons: When True, duplicate photos are returned as
            singleton clusters. When False, duplicate photos are dropped.

    Returns:
        Tuple:
        - List of photo ID lists (one or more clusters, each with max_size or fewer photos)
        - duplicate_of map: duplicate_photo_id -> canonical_photo_id
    """
    if not cluster_photo_ids:
        return [], {}

    # Map photo IDs to matrix indices
    pid_to_idx = {pid: i for i, pid in enumerate(photo_ids)}
    cluster_indices = [pid_to_idx[pid] for pid in cluster_photo_ids if pid in pid_to_idx]

    if not cluster_indices:
        return [cluster_photo_ids], {}

    n = len(cluster_indices)
    if n == 1:
        return [cluster_photo_ids], {}

    # Compute semantic similarity within cluster
    cluster_embeddings = embeddings[cluster_indices]
    sem_sim = cluster_embeddings @ cluster_embeddings.T

    # Compute TRANSITION SCORE for each photo (overlap with prev + next in sequence)
    transition_scores = np.zeros(n)
    for i in range(n):
        idx_i = cluster_indices[i]
        if i > 0:
            idx_prev = cluster_indices[i - 1]
            transition_scores[i] += adjacency[idx_i, idx_prev]
        if i < n - 1:
            idx_next = cluster_indices[i + 1]
            transition_scores[i] += adjacency[idx_i, idx_next]

    # Endpoints get bonus
    transition_scores[0] += 0.5
    transition_scores[n - 1] += 0.5

    # Find CONSECUTIVE duplicates only
    to_remove = set()
    duplicate_of: Dict[int, int] = {}
    for i in range(n - 1):
        if i in to_remove:
            continue
        j = i + 1
        if j in to_remove:
            continue

        idx_i = cluster_indices[i]
        idx_j = cluster_indices[j]
        semantic_score = float(sem_sim[i, j])
        geometric_overlap = float(adjacency[idx_i, idx_j])
        is_duplicate_pair = (
            semantic_score >= DUPLICATE_SIMILARITY_THRESHOLD
            or (
                semantic_score >= DUPLICATE_GEOMETRIC_SEMANTIC_THRESHOLD
                and geometric_overlap >= DUPLICATE_GEOMETRIC_OVERLAP_THRESHOLD
            )
        )

        if is_duplicate_pair:
            if transition_scores[i] >= transition_scores[j]:
                to_remove.add(j)
                duplicate_of[cluster_photo_ids[j]] = cluster_photo_ids[i]
                logger.debug(
                    "Marking consecutive duplicate %s -> %s (sem=%.3f, overlap=%.3f)",
                    cluster_photo_ids[j],
                    cluster_photo_ids[i],
                    semantic_score,
                    geometric_overlap,
                )
            else:
                to_remove.add(i)
                duplicate_of[cluster_photo_ids[i]] = cluster_photo_ids[j]
                logger.debug(
                    "Marking consecutive duplicate %s -> %s (sem=%.3f, overlap=%.3f)",
                    cluster_photo_ids[i],
                    cluster_photo_ids[j],
                    semantic_score,
                    geometric_overlap,
                )

    # Keep canonical photos in order, split if needed.
    remaining_indices = [i for i in range(n) if i not in to_remove]
    remaining = [cluster_photo_ids[i] for i in remaining_indices]
    duplicate_clusters = [[cluster_photo_ids[i]] for i in range(n) if i in to_remove]
    emitted_duplicate_clusters = duplicate_clusters if keep_duplicate_singletons else []

    if len(remaining) <= max_size:
        if len(remaining) < len(cluster_photo_ids):
            if keep_duplicate_singletons:
                logger.info(
                    "Deduplicated: %s -> %s canonical (+%s duplicate singleton clusters)",
                    len(cluster_photo_ids),
                    len(remaining),
                    len(duplicate_clusters),
                )
            else:
                logger.info(
                    "Deduplicated: %s -> %s canonical (dropped %s duplicates)",
                    len(cluster_photo_ids),
                    len(remaining),
                    len(duplicate_clusters),
                )
        if not remaining:
            return emitted_duplicate_clusters, duplicate_of
        return [remaining] + emitted_duplicate_clusters, duplicate_of

    # Too many photos - SPLIT into multiple clusters instead of dropping.
    # Use constrained partitioning to GUARANTEE each cluster size <= max_size.
    m = len(remaining)
    num_clusters = (m + max_size - 1) // max_size  # Minimum required clusters

    # Find transition strengths between consecutive photos
    remaining_cluster_indices = [pid_to_idx[pid] for pid in remaining]
    transition_strengths = []
    for i in range(m - 1):
        idx_i = remaining_cluster_indices[i]
        idx_j = remaining_cluster_indices[i + 1]
        strength = adjacency[idx_i, idx_j]
        transition_strengths.append(float(strength))

    # Dynamic programming with exact cluster count (minimum needed), while
    # enforcing each segment length <= max_size and preferring weak boundaries.
    inf = float("inf")
    dp = [[inf] * (m + 1) for _ in range(num_clusters + 1)]
    prev = [[None] * (m + 1) for _ in range(num_clusters + 1)]
    dp[0][0] = 0.0

    for c in range(1, num_clusters + 1):
        min_pos = c  # at least one photo per cluster
        max_pos = min(m, c * max_size)
        for pos in range(min_pos, max_pos + 1):
            for seg_len in range(1, max_size + 1):
                start = pos - seg_len
                if start < (c - 1):
                    continue
                prev_cost = dp[c - 1][start]
                if prev_cost == inf:
                    continue

                # Cost of adding boundary before this segment.
                boundary_cost = 0.0 if start == 0 else transition_strengths[start - 1]
                cand = prev_cost + boundary_cost
                if cand < dp[c][pos]:
                    dp[c][pos] = cand
                    prev[c][pos] = start

    # Reconstruct optimal partition; fallback to fixed chunking if unreachable.
    result_clusters = []
    if dp[num_clusters][m] != inf:
        boundaries = []
        c = num_clusters
        pos = m
        while c > 0:
            start = prev[c][pos]
            if start is None:
                boundaries = []
                break
            boundaries.append((start, pos))
            pos = start
            c -= 1

        boundaries.reverse()
        for start, end in boundaries:
            result_clusters.append(remaining[start:end])

    if not result_clusters:
        # Deterministic hard fallback: contiguous chunks, each <= max_size.
        for i in range(0, m, max_size):
            result_clusters.append(remaining[i:i + max_size])

    # Safety invariant: never return oversize clusters.
    oversized = [cluster for cluster in result_clusters if len(cluster) > max_size]
    if oversized:
        logger.warning(
            "Oversized clusters after split (%s). Applying hard chunk fallback.",
            [len(c) for c in oversized],
        )
        result_clusters = []
        for i in range(0, m, max_size):
            result_clusters.append(remaining[i:i + max_size])

    # Log the split
    sizes = [len(c) for c in result_clusters]
    if keep_duplicate_singletons:
        logger.info(
            "Split cluster: %s photos -> %s canonical clusters (sizes: %s) + %s duplicate singleton clusters",
            len(cluster_photo_ids),
            len(result_clusters),
            sizes,
            len(duplicate_clusters),
        )
    else:
        logger.info(
            "Split cluster: %s photos -> %s canonical clusters (sizes: %s), dropped %s duplicates",
            len(cluster_photo_ids),
            len(result_clusters),
            sizes,
            len(duplicate_clusters),
        )

    return result_clusters + emitted_duplicate_clusters, duplicate_of


def _build_similarity_lookup(
    similarity_records: List[Dict[str, object]],
) -> Dict[Tuple[int, int], Dict[str, object]]:
    """Build quick lookup for per-pair metrics collected during stage 2."""
    lookup: Dict[Tuple[int, int], Dict[str, object]] = {}
    for record in similarity_records:
        photo_a = record.get("photo_a_id")
        photo_b = record.get("photo_b_id")
        if photo_a is None or photo_b is None:
            continue
        key = (int(min(photo_a, photo_b)), int(max(photo_a, photo_b)))
        lookup[key] = record
    return lookup


def enforce_transition_quality(
    ordered_clusters: List[List[int]],
    photo_ids: List[int],
    adjacency: np.ndarray,
    similarity: np.ndarray,
    edge_has_geometry: np.ndarray,
    similarity_records: List[Dict[str, object]],
    room_labels: Optional[List[str]] = None,
    min_overlap: float = HARD_TRANSITION_MIN_OVERLAP,
    min_inliers: int = HARD_TRANSITION_MIN_INLIERS,
    adjacent_semantic_trust: float = HARD_TRANSITION_ADJACENT_SEMANTIC_TRUST,
    dist2_semantic_trust: float = HARD_TRANSITION_DIST2_SEMANTIC_TRUST,
    max_semantic_gap: int = HARD_TRANSITION_MAX_SEMANTIC_GAP,
    recovery_semantic_trust: float = HARD_TRANSITION_RECOVERY_SEMANTIC_TRUST,
    front_recovery_semantic_trust: float = HARD_TRANSITION_FRONT_RECOVERY_SEMANTIC_TRUST,
    require_geometry: bool = HARD_TRANSITION_REQUIRE_GEOMETRY,
    require_direction: bool = HARD_TRANSITION_REQUIRE_DIRECTION,
) -> List[List[int]]:
    """Split ordered clusters at transition edges that are too weak for video cuts.

    This is a final safety gate: connected-components can still keep semantic-only
    links that are visually related but not transition-safe.
    By default this service enforces geometry-only transitions for reliability.
    """
    if not ordered_clusters:
        return []

    pid_to_idx = {pid: idx for idx, pid in enumerate(photo_ids)}
    pair_lookup = _build_similarity_lookup(similarity_records)

    refined_clusters: List[List[int]] = []
    split_count = 0

    for cluster in ordered_clusters:
        if len(cluster) <= 1:
            refined_clusters.append(cluster)
            continue

        current_chain = [cluster[0]]

        for left_pid, right_pid in zip(cluster, cluster[1:]):
            idx_left = pid_to_idx.get(left_pid)
            idx_right = pid_to_idx.get(right_pid)
            if idx_left is None or idx_right is None:
                current_chain.append(right_pid)
                continue

            overlap_score = float(adjacency[idx_left, idx_right])
            semantic_score = float(similarity[idx_left, idx_right])
            has_geometry = bool(edge_has_geometry[idx_left, idx_right])
            seq_gap = abs(idx_right - idx_left)

            record = pair_lookup.get((min(left_pid, right_pid), max(left_pid, right_pid)), {})
            raw_inliers = record.get("geometric_inliers")
            try:
                inliers = int(raw_inliers) if raw_inliers is not None else 0
            except (TypeError, ValueError):
                inliers = 0
            raw_dx = record.get("direction_dx")
            raw_dy = record.get("direction_dy")
            try:
                direction_dx = float(raw_dx) if raw_dx is not None else None
                direction_dy = float(raw_dy) if raw_dy is not None else None
            except (TypeError, ValueError):
                direction_dx = None
                direction_dy = None
            has_direction = (
                direction_dx is not None
                and direction_dy is not None
                and ((direction_dx * direction_dx) + (direction_dy * direction_dy)) > 1e-8
            )
            pair_source = str(record.get("pair_source") or "")
            room_left = room_labels[idx_left] if room_labels and idx_left < len(room_labels) else None
            room_right = room_labels[idx_right] if room_labels and idx_right < len(room_labels) else None
            both_front_exterior = (
                room_family(room_left) == "front_exterior"
                and room_family(room_right) == "front_exterior"
            )

            keep_edge = False
            keep_reason = ""

            if require_geometry:
                geometry_strong_enough = (
                    has_geometry
                    and inliers >= min_inliers
                    and overlap_score >= min_overlap
                )
                direction_ok = (not require_direction) or has_direction
                if geometry_strong_enough and direction_ok:
                    keep_edge = True
                    keep_reason = "geometry_strict"
            else:
                if has_geometry and (inliers >= min_inliers or overlap_score >= min_overlap):
                    keep_edge = True
                    keep_reason = "geometry"
                elif "semantic_recovery" in pair_source:
                    if (
                        both_front_exterior
                        and semantic_score >= front_recovery_semantic_trust
                    ):
                        keep_edge = True
                        keep_reason = "front_recovery"
                    elif seq_gap <= max_semantic_gap and semantic_score >= recovery_semantic_trust:
                        keep_edge = True
                        keep_reason = "semantic_recovery"
                elif seq_gap <= 1 and semantic_score >= adjacent_semantic_trust:
                    keep_edge = True
                    keep_reason = "adjacent_semantic"
                elif seq_gap <= 2 and semantic_score >= dist2_semantic_trust:
                    keep_edge = True
                    keep_reason = "dist2_semantic"

            if keep_edge:
                current_chain.append(right_pid)
                logger.debug(
                    "Transition kept %s -> %s (%s: overlap=%.3f, inliers=%s, sem=%.3f, gap=%s, source=%s)",
                    left_pid,
                    right_pid,
                    keep_reason,
                    overlap_score,
                    inliers,
                    semantic_score,
                    seq_gap,
                    pair_source,
                )
                continue

            split_count += 1
            logger.info(
                "Transition split %s -> %s (overlap=%.3f, inliers=%s, sem=%.3f, "
                "has_geometry=%s, has_direction=%s, gap=%s, source=%s)",
                left_pid,
                right_pid,
                overlap_score,
                inliers,
                semantic_score,
                "yes" if has_geometry else "no",
                "yes" if has_direction else "no",
                seq_gap,
                pair_source,
            )
            refined_clusters.append(current_chain)
            current_chain = [right_pid]

        if current_chain:
            refined_clusters.append(current_chain)

    logger.info(
        "Transition quality enforcement: %s ordered clusters -> %s refined clusters "
        "(splits=%s, min_overlap=%.2f, min_inliers=%s, strict_geometry=%s, require_direction=%s, adjacent_sem>=%.2f, dist2_sem>=%.2f, recovery_sem>=%.2f, front_recovery_sem>=%.2f, max_sem_gap=%s)",
        len(ordered_clusters),
        len(refined_clusters),
        split_count,
        min_overlap,
        min_inliers,
        "yes" if require_geometry else "no",
        "yes" if require_direction else "no",
        adjacent_semantic_trust,
        dist2_semantic_trust,
        recovery_semantic_trust,
        front_recovery_semantic_trust,
        max_semantic_gap,
    )

    return refined_clusters


# LightGlue/LoFTR model singleton
_matcher = None
_matcher_type = None


def _load_matcher():
    """Load learned feature matcher (lazy initialization)."""
    global _matcher, _matcher_type

    if _matcher is not None:
        return _matcher, _matcher_type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try LoFTR first (indoor checkpoint performs better for real-estate scenes).
    try:
        from kornia.feature import LoFTR
        for checkpoint in ("indoor_new", "outdoor"):
            try:
                _matcher = LoFTR(pretrained=checkpoint)
                _matcher = _matcher.to(device)
                _matcher.eval()
                _matcher_type = "loftr"
                logger.info(f"Loaded LoFTR matcher ({checkpoint}) on {device}")
                return _matcher, _matcher_type
            except Exception as ckpt_err:
                logger.debug(f"LoFTR checkpoint '{checkpoint}' unavailable: {ckpt_err}")
    except Exception as e:
        logger.debug(f"LoFTR not available: {e}")

    # Fallback to ORB
    _matcher = None
    _matcher_type = "orb"
    logger.info("Using ORB matcher (kornia not available)")
    return _matcher, _matcher_type


def match_image_pair(
    img1: Image.Image,
    img2: Image.Image,
    use_orb_prefilter: bool = False,
) -> Tuple[int, int, float, Tuple[float, float]]:
    """Match two images using learned features with ORB pre-filtering.

    Performance optimization (optional): ORB is ~10x faster than LoFTR.
    - If ORB shows ≤2 inliers → definitely no overlap, skip LoFTR
    - If ORB shows ≥15 inliers → definitely overlap, use ORB result
    - Otherwise → run LoFTR for accurate matching

    Args:
        img1: First PIL Image
        img2: Second PIL Image
        use_orb_prefilter: Whether to use ORB as pre-filter (default True)

    Returns:
        Tuple of (num_matches, num_inliers, overlap_score, direction_vector)
        direction_vector is (dx, dy) showing how content shifted from img1 to img2
    """
    matcher, matcher_type = _load_matcher()

    # If LoFTR is not available, just use ORB
    if matcher_type != "loftr" or matcher is None:
        return _match_orb(img1, img2)

    # ORB pre-filter is disabled by default for consistency.
    # It can produce unstable errors and false positives on repetitive textures.
    if use_orb_prefilter:
        orb_matches, orb_inliers, orb_score, orb_direction = _match_orb(img1, img2)

        # Definitely no overlap - skip LoFTR
        if orb_inliers <= ORB_QUICK_REJECT_INLIERS:
            logger.debug(f"ORB pre-filter: {orb_inliers} inliers ≤ {ORB_QUICK_REJECT_INLIERS} → skip LoFTR")
            return orb_matches, orb_inliers, orb_score, orb_direction

        # Definitely overlap - use ORB result (skip expensive LoFTR)
        if orb_inliers >= ORB_QUICK_ACCEPT_INLIERS:
            logger.debug(f"ORB pre-filter: {orb_inliers} inliers ≥ {ORB_QUICK_ACCEPT_INLIERS} → use ORB")
            return orb_matches, orb_inliers, orb_score, orb_direction

        # Ambiguous case - need LoFTR for accuracy
        logger.debug(f"ORB pre-filter: {orb_inliers} inliers → ambiguous, running LoFTR")

    return _match_loftr(matcher, img1, img2)


def _match_loftr(
    matcher,
    img1: Image.Image,
    img2: Image.Image,
) -> Tuple[int, int, float, Tuple[float, float]]:
    """Match using LoFTR (learned dense matching)."""
    device = next(matcher.parameters()).device

    # Convert to grayscale tensors
    img1_gray = np.array(img1.convert("L"), dtype=np.float32) / 255.0
    img2_gray = np.array(img2.convert("L"), dtype=np.float32) / 255.0

    # Resize to LoFTR input size (divisible by 8)
    h, w = 480, 640
    img1_resized = cv2.resize(img1_gray, (w, h))
    img2_resized = cv2.resize(img2_gray, (w, h))

    # Convert to tensors
    tensor1 = torch.from_numpy(img1_resized).unsqueeze(0).unsqueeze(0).to(device)
    tensor2 = torch.from_numpy(img2_resized).unsqueeze(0).unsqueeze(0).to(device)

    # Run LoFTR
    with torch.no_grad():
        input_dict = {"image0": tensor1, "image1": tensor2}
        correspondences = matcher(input_dict)

    # Extract matches
    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    confidence = correspondences["confidence"].cpu().numpy()

    # Filter by confidence (0.7 threshold)
    mask = confidence > 0.7
    mkpts0 = mkpts0[mask]
    mkpts1 = mkpts1[mask]

    num_matches = len(mkpts0)

    if num_matches < 8:
        return num_matches, 0, 0.0, (0.0, 0.0)

    # Ensure arrays are contiguous and have correct shape
    mkpts0 = np.ascontiguousarray(mkpts0, dtype=np.float32)
    mkpts1 = np.ascontiguousarray(mkpts1, dtype=np.float32)

    # Additional safety check
    if mkpts0.shape[0] < 8 or mkpts1.shape[0] < 8 or mkpts0.shape[0] != mkpts1.shape[0]:
        return num_matches, 0, 0.0, (0.0, 0.0)

    # Geometric verification
    try:
        F, inlier_mask = cv2.findFundamentalMat(
            mkpts0, mkpts1,
            cv2.FM_RANSAC,
            RANSAC_REPROJ_THRESHOLD,
            0.999
        )
    except cv2.error as e:
        logger.warning(f"findFundamentalMat failed: {e}")
        return num_matches, 0, 0.0, (0.0, 0.0)

    if inlier_mask is None:
        return num_matches, 0, 0.0, (0.0, 0.0)

    num_inliers = int(inlier_mask.sum())

    # Compute direction vector from matched keypoints
    direction = compute_direction_vector(mkpts0, mkpts1, inlier_mask)

    # Score based on inlier ratio and count
    inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0
    coverage_bonus = min(1.0, num_inliers / 100)
    score = inlier_ratio * (0.5 + 0.5 * coverage_bonus)

    return num_matches, num_inliers, score, direction


def _match_orb(
    img1: Image.Image,
    img2: Image.Image,
) -> Tuple[int, int, float, Tuple[float, float]]:
    """Fallback ORB matching."""
    img1_gray = np.array(img1.convert("L"))
    img2_gray = np.array(img2.convert("L"))

    orb = cv2.ORB_create(nfeatures=2000)

    kp1, desc1 = orb.detectAndCompute(img1_gray, None)
    kp2, desc2 = orb.detectAndCompute(img2_gray, None)

    if desc1 is None or desc2 is None:
        return 0, 0, 0.0, (0.0, 0.0)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        matches = bf.knnMatch(desc1, desc2, k=2)
    except cv2.error:
        return 0, 0, 0.0, (0.0, 0.0)

    # Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    num_matches = len(good_matches)

    if num_matches < 8:
        return num_matches, 0, 0.0, (0.0, 0.0)

    src_pts = np.ascontiguousarray([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
    dst_pts = np.ascontiguousarray([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32)

    # Safety check
    if src_pts.shape[0] < 8 or dst_pts.shape[0] < 8 or src_pts.shape[0] != dst_pts.shape[0]:
        return num_matches, 0, 0.0, (0.0, 0.0)

    try:
        F, mask = cv2.findFundamentalMat(
            src_pts, dst_pts,
            cv2.FM_RANSAC,
            RANSAC_REPROJ_THRESHOLD,
            0.999
        )
    except cv2.error as e:
        # ORB is only a fallback path; avoid noisy warnings for known degenerate cases.
        logger.debug(f"findFundamentalMat failed in ORB: {e}")
        return num_matches, 0, 0.0, (0.0, 0.0)

    if mask is None:
        return num_matches, 0, 0.0, (0.0, 0.0)

    num_inliers = int(mask.sum())

    # Compute direction vector from matched keypoints
    direction = compute_direction_vector(src_pts, dst_pts, mask)

    inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0
    coverage_bonus = min(1.0, num_inliers / 50)
    score = inlier_ratio * (0.5 + 0.5 * coverage_bonus)

    return num_matches, num_inliers, score, direction


def compute_overlap_within_cluster(
    images: List[Image.Image],
    photo_ids: List[int],
) -> np.ndarray:
    """Compute pairwise overlap within a single DINOv2 cluster.

    Args:
        images: List of PIL Images in the cluster
        photo_ids: List of photo IDs

    Returns:
        NxN overlap matrix
    """
    n = len(images)
    if n <= 1:
        return np.ones((n, n))

    overlap_matrix = np.zeros((n, n))
    np.fill_diagonal(overlap_matrix, 1.0)

    for i in range(n):
        for j in range(i + 1, n):
            num_matches, num_inliers, score = match_image_pair(images[i], images[j])

            if num_inliers >= MIN_INLIERS_FOR_OVERLAP:
                overlap_matrix[i, j] = score
                overlap_matrix[j, i] = score
                logger.info(
                    f"Overlap {photo_ids[i]} <-> {photo_ids[j]}: "
                    f"{num_matches} matches -> {num_inliers} inliers, "
                    f"score={score:.3f} ✓"
                )
            else:
                logger.debug(
                    f"No overlap {photo_ids[i]} <-> {photo_ids[j]}: "
                    f"{num_matches} matches -> {num_inliers} inliers"
                )

    return overlap_matrix


def split_cluster_by_overlap(
    images: List[Image.Image],
    photo_ids: List[int],
    overlap_matrix: np.ndarray,
) -> List[List[int]]:
    """Split a DINOv2 cluster into sub-clusters by geometric overlap.

    Uses single-linkage to build chains of overlapping photos.

    Args:
        images: List of images in the cluster
        photo_ids: List of photo IDs
        overlap_matrix: NxN overlap matrix

    Returns:
        List of photo ID lists (sub-clusters with verified overlap)
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    n = len(photo_ids)
    if n <= 1:
        return [photo_ids]

    # Check if there's ANY significant overlap
    has_overlap = (overlap_matrix > OVERLAP_THRESHOLD).sum() > n  # More than diagonal

    if not has_overlap:
        # No overlaps found - each photo is its own cluster
        logger.debug(f"No geometric overlap found in cluster of {n} photos")
        return [[pid] for pid in photo_ids]

    # Convert to distance matrix
    distance_matrix = 1.0 - np.clip(overlap_matrix, 0, 1)
    np.fill_diagonal(distance_matrix, 0)

    # Single-linkage clustering (builds chains)
    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method='single')

    # Cut at threshold
    distance_threshold = 1.0 - OVERLAP_THRESHOLD
    labels = fcluster(Z, t=distance_threshold, criterion='distance')

    # Group by label
    clusters_dict = defaultdict(list)
    for i, label in enumerate(labels):
        clusters_dict[label].append(photo_ids[i])

    sub_clusters = list(clusters_dict.values())

    logger.info(f"Overlap clustering: {n} photos -> {len(sub_clusters)} sub-clusters")

    return sub_clusters


# ============================================================================
# GRAPH-BASED CLUSTERING: DINOv2 proposes edges, geometry verifies
# ============================================================================


def _connected_nodes(edge_mask: np.ndarray, start_idx: int) -> set:
    """Return connected node indices from start_idx using DFS over boolean edge mask."""
    stack = [start_idx]
    visited = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        neighbors = np.where(edge_mask[node])[0]
        for nbr in neighbors:
            if nbr not in visited:
                stack.append(int(nbr))
    return visited


def prune_weak_semantic_bridges(
    adjacency: np.ndarray,
    similarity: np.ndarray,
    edge_has_geometry: np.ndarray,
    photo_ids: List[int],
) -> None:
    """Prune semantic-only bridge edges that connect weakly related subgraphs.

    This reduces false merges caused by transitive chaining of weak temporal
    semantic edges (A-B and B-C) when A-C semantic support is weak.
    """
    n = adjacency.shape[0]
    edge_mask = adjacency > OVERLAP_THRESHOLD

    for i in range(n):
        for j in range(i + 1, n):
            if not edge_mask[i, j]:
                continue
            if edge_has_geometry[i, j]:
                continue

            component = _connected_nodes(edge_mask, i)
            if j not in component or len(component) < 3:
                continue

            # Temporarily remove candidate bridge and check whether it truly
            # splits the component.
            edge_mask[i, j] = False
            edge_mask[j, i] = False

            comp_i = _connected_nodes(edge_mask, i)
            if j in comp_i:
                edge_mask[i, j] = True
                edge_mask[j, i] = True
                continue
            comp_j = _connected_nodes(edge_mask, j)

            max_cross_sem = max(similarity[a, b] for a in comp_i for b in comp_j)
            if max_cross_sem < SEMANTIC_BRIDGE_SUPPORT_THRESHOLD:
                adjacency[i, j] = 0.0
                adjacency[j, i] = 0.0
                logger.info(
                    "Pruned semantic bridge %s <-> %s (max cross semantic=%.3f < %.2f)",
                    photo_ids[i],
                    photo_ids[j],
                    max_cross_sem,
                    SEMANTIC_BRIDGE_SUPPORT_THRESHOLD,
                )
                # keep edge removed in edge_mask
            else:
                edge_mask[i, j] = True
                edge_mask[j, i] = True

def cluster_photos_graph_based(
    images: List[Image.Image],
    photo_ids: List[int],
    k: int = 8,
    max_cluster_size: int = 6,
    db_session=None,
    job_id: int = None,
    room_labels: List[str] = None,
    return_metadata: bool = False,
) -> List[List[int]] | Tuple[List[List[int]], Dict[str, object]]:
    """Graph-based clustering: semantic proposals + geometric verification.

    This is a cleaner architecture than semantic-first clustering:
    1. DINOv2 embeddings propose candidate edges (top-K similar pairs)
    2. Geometric verification on proposed edges only
    3. Connected components = final clusters

    Key insight: Semantic similarity is used as a FILTER (edge proposal),
    not as ground truth. This eliminates the need for cross-cluster merging.

    Complexity: O(N × K) geometric checks instead of O(N²)

    Args:
        images: List of PIL Images
        photo_ids: List of photo IDs
        k: Number of candidate neighbors per image (default 8)
        max_cluster_size: Maximum photos per cluster
        room_labels: Optional list of room labels for each photo (used to penalize cross-room connections)

    Returns:
        Final clusters, or (clusters, metadata) when return_metadata=True.
    """
    from sklearn.preprocessing import normalize
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(images)
    logger.info(f"Graph-based clustering for {n} photos (k={k})")

    # Track similarity data for database storage
    similarity_records = []  # Will store dicts for batch insert

    if n <= 1:
        clusters = [photo_ids] if photo_ids else []
        if return_metadata:
            return clusters, {"duplicate_of_map": {}}
        return clusters

    # -------------------------------------------------------------------------
    # Stage 1: Compute DINOv2 embeddings and build candidate graph
    # -------------------------------------------------------------------------
    logger.info("Stage 1: Computing DINOv2 embeddings...")
    embeddings = compute_dinov2_embeddings(images)
    embeddings = normalize(embeddings)  # For cosine similarity

    # Compute similarity matrix
    similarity = embeddings @ embeddings.T

    # -------------------------------------------------------------------------
    # Stage 1b: Build candidate pairs from DINOv2 top-K
    # -------------------------------------------------------------------------
    logger.info(f"Stage 1b: Building candidate graph (top-{k} per image)...")
    semantic_pairs = set()
    for i in range(n):
        sorted_indices = np.argsort(-similarity[i])
        neighbors = [j for j in sorted_indices if j != i][:k]
        for j in neighbors:
            pair = (min(i, j), max(i, j))
            semantic_pairs.add(pair)
    logger.info(f"  DINOv2 top-{k} proposed {len(semantic_pairs)} semantic pairs")

    # -------------------------------------------------------------------------
    # Stage 1c: Add temporal window pairs (adjacent photos likely same room)
    # -------------------------------------------------------------------------
    logger.info(f"Stage 1c: Adding temporal window pairs (window=±{TEMPORAL_WINDOW})...")
    temporal_pairs = set()
    for i in range(n):
        for offset in range(1, TEMPORAL_WINDOW + 1):
            if i + offset < n:
                temporal_pairs.add((i, i + offset))
    logger.info(f"  Added {len(temporal_pairs)} temporal pairs")

    # -------------------------------------------------------------------------
    # Stage 2a: Temporal pairs - require geometric verification unless very high semantic
    # -------------------------------------------------------------------------
    logger.info("Stage 2a: Checking temporal pairs...")
    adjacency = np.zeros((n, n))
    edge_has_geometry = np.zeros((n, n), dtype=bool)
    directions = {}  # (i, j) -> (dx, dy) direction vectors for ordering
    temporal_matched = 0
    temporal_semantic_only = 0
    temporal_geometric = 0

    for i, j in sorted(temporal_pairs):
        sem_sim = similarity[i, j]
        temporal_dist = abs(j - i)

        # Check if rooms are different
        room_i = room_labels[i] if room_labels else None
        room_j = room_labels[j] if room_labels else None
        is_cross_room = rooms_are_different(room_i, room_j)

        # For NON-ADJACENT cross-room pairs (temporal_dist > 1), skip entirely
        # But for ADJACENT photos (temporal_dist = 1), DO geometric verification
        # because ML room labels are often wrong for adjacent photos
        if is_cross_room and temporal_dist > 1:
            logger.info(f"  Temporal {photo_ids[i]} <-> {photo_ids[j]}: "
                       f"SKIP - different rooms ({room_i} vs {room_j}), dist={temporal_dist}")
            similarity_records.append({
                "photo_a_id": photo_ids[min(i, j)],
                "photo_b_id": photo_ids[max(i, j)],
                "pair_source": "temporal_cross_room",
                "dinov2_similarity": float(sem_sim),
                "geometric_matches": None,
                "geometric_inliers": None,
                "geometric_score": None,
                "direction_dx": None,
                "direction_dy": None,
                "is_connected": 0,
            })
            continue

        room_info = f" [cross-room: {room_i}/{room_j}]" if is_cross_room else ""
        logger.info(f"  Temporal {photo_ids[i]} <-> {photo_ids[j]} (dist={temporal_dist}): "
                   f"semantic={sem_sim:.3f}{room_info}")

        is_matched = False
        num_matches = None
        num_inliers = None
        geo_score = None
        direction = (0.0, 0.0)

        # Cross-room pairs need geometric verification with higher threshold
        # But for ADJACENT cross-room (temporal_dist=1), use lower threshold since ML often mislabels
        if is_cross_room:
            min_inliers_required = MIN_INLIERS_CROSS_ROOM_ADJACENT if temporal_dist == 1 else MIN_INLIERS_CROSS_ROOM
        else:
            min_inliers_required = MIN_INLIERS_FOR_OVERLAP

        # Very high semantic similarity - trust it for clustering (but NOT for cross-room)
        if sem_sim >= TEMPORAL_SEMANTIC_THRESHOLD and not is_cross_room:
            adjacency[i, j] = sem_sim
            adjacency[j, i] = sem_sim
            temporal_matched += 1
            temporal_semantic_only += 1
            is_matched = True

            # Still run geometric to get direction for ordering
            num_matches, num_inliers, geo_score, direction = match_image_pair(images[i], images[j])
            if num_inliers >= MIN_INLIERS_FOR_DIRECTION and direction != (0.0, 0.0):
                directions[(i, j)] = direction
                dir_str = f"dir=({direction[0]:.2f},{direction[1]:.2f})"
                logger.info(f"    ✓ MATCHED (semantic >= {TEMPORAL_SEMANTIC_THRESHOLD}, {dir_str} from {num_inliers} inliers)")
            else:
                logger.info(f"    ✓ MATCHED (semantic >= {TEMPORAL_SEMANTIC_THRESHOLD}, no direction - {num_inliers} inliers)")

            if num_inliers is not None and num_inliers >= MIN_INLIERS_FOR_OVERLAP:
                edge_has_geometry[i, j] = True
                edge_has_geometry[j, i] = True

        # Moderate semantic OR cross-room - require geometric verification
        elif sem_sim >= TEMPORAL_GEOMETRIC_THRESHOLD or is_cross_room:
            logger.info(f"    Verifying geometrically (need {min_inliers_required} inliers)...")
            num_matches, num_inliers, geo_score, direction = match_image_pair(images[i], images[j])

            if num_inliers >= min_inliers_required:
                adjacency[i, j] = geo_score
                adjacency[j, i] = geo_score
                edge_has_geometry[i, j] = True
                edge_has_geometry[j, i] = True
                directions[(i, j)] = direction
                temporal_matched += 1
                temporal_geometric += 1
                is_matched = True
                dir_str = f"dir=({direction[0]:.2f},{direction[1]:.2f})" if direction != (0.0, 0.0) else "dir=unknown"
                logger.info(f"    ✓ MATCHED (geometric: {num_inliers} inliers >= {min_inliers_required}, score={geo_score:.3f}, {dir_str})")
            # Fallback: Immediate neighbors with good semantic similarity - trust even without geometric
            elif temporal_dist == 1 and sem_sim >= NEIGHBOR_TRUST_THRESHOLD and not is_cross_room:
                room_i_norm = normalize_room_label(room_i)
                room_j_norm = normalize_room_label(room_j)
                same_room_label = (
                    room_i_norm != ""
                    and room_i_norm != "unknown"
                    and room_i_norm == room_j_norm
                )
                ambiguous_rooms = {"bathroom"}
                same_label_threshold = (
                    AMBIGUOUS_SAME_LABEL_TRUST
                    if room_i_norm in ambiguous_rooms
                    else SAME_LABEL_NEIGHBOR_TRUST
                )
                same_label_trust = same_room_label and sem_sim >= same_label_threshold
                local_support = has_local_semantic_support(i, j, similarity)
                allow_semantic_neighbor = (
                    sem_sim >= HIGH_CONFIDENCE_NEIGHBOR_TRUST
                    or local_support
                    or same_label_trust
                )
                if not allow_semantic_neighbor:
                    logger.info(
                        "    ✗ No neighbor semantic trust (sem=%.3f, support=%s, same_label=%s)",
                        sem_sim,
                        "yes" if local_support else "no",
                        "yes" if same_label_trust else "no",
                    )
                else:
                    adjacency[i, j] = sem_sim
                    adjacency[j, i] = sem_sim
                    temporal_matched += 1
                    temporal_semantic_only += 1
                    is_matched = True

                    if num_inliers >= MIN_INLIERS_FOR_DIRECTION and direction != (0.0, 0.0):
                        directions[(i, j)] = direction
                        dir_str = f"dir=({direction[0]:.2f},{direction[1]:.2f})"
                        logger.info(
                            "    ✓ MATCHED (neighbor trust: semantic %.3f, support=%s, same_label=%s, %s from %s inliers)",
                            sem_sim,
                            "yes" if local_support else "no",
                            "yes" if same_label_trust else ("high-confidence" if sem_sim >= HIGH_CONFIDENCE_NEIGHBOR_TRUST else "no"),
                            dir_str,
                            num_inliers,
                        )
                    else:
                        logger.info(
                            "    ✓ MATCHED (neighbor trust: semantic %.3f, support=%s, same_label=%s, no direction)",
                            sem_sim,
                            "yes" if local_support else "no",
                            "yes" if same_label_trust else ("high-confidence" if sem_sim >= HIGH_CONFIDENCE_NEIGHBOR_TRUST else "no"),
                        )
            # Cross-room adjacent fallback (strict): recover known classifier confusions only
            # (e.g., living<->entrance, bathroom<->laundry) when pair is mutually top-semantic.
            elif temporal_dist == 1 and is_cross_room:
                compatible_families = rooms_allow_adjacent_semantic_bridge(room_i, room_j)
                local_support = has_local_semantic_support(i, j, similarity)
                mutual_top = is_mutual_top_semantic_neighbor(i, j, similarity, top_k=CROSS_ROOM_RECOVERY_TOPK)
                recovery_threshold = (
                    SERVICE_ROOM_RECOVERY_THRESHOLD
                    if room_family(room_i) == "service" and room_family(room_j) == "service"
                    else CROSS_ROOM_RECOVERY_THRESHOLD
                )
                if (
                    compatible_families
                    and sem_sim >= recovery_threshold
                    and (mutual_top or local_support)
                ):
                    adjacency[i, j] = sem_sim
                    adjacency[j, i] = sem_sim
                    temporal_matched += 1
                    temporal_semantic_only += 1
                    is_matched = True
                    logger.info(
                        "    ✓ MATCHED (cross-room recovery: sem=%.3f >= %.2f, mutual_top=%s, support=%s, %s/%s)",
                        sem_sim,
                        recovery_threshold,
                        "yes" if mutual_top else "no",
                        "yes" if local_support else "no",
                        room_i,
                        room_j,
                    )
                else:
                    logger.info(
                        "    ✗ Cross-room recovery rejected (sem=%.3f, threshold=%.2f, mutual_top=%s, support=%s, compatible=%s)",
                        sem_sim,
                        recovery_threshold,
                        "yes" if mutual_top else "no",
                        "yes" if local_support else "no",
                        "yes" if compatible_families else "no",
                    )
            # Fallback for distance-2 temporal neighbors:
            # allow very high semantic matches when geometry fails.
            elif temporal_dist == 2 and not is_cross_room:
                room_i_norm = normalize_room_label(room_i)
                room_j_norm = normalize_room_label(room_j)
                same_room_label = (
                    room_i_norm != ""
                    and room_i_norm != "unknown"
                    and room_i_norm == room_j_norm
                )

                dist2_threshold = DIST2_TRUST_THRESHOLD
                if same_room_label:
                    if room_i_norm in {"bathroom"}:
                        dist2_threshold = AMBIGUOUS_DIST2_TRUST
                    else:
                        dist2_threshold = SAME_LABEL_DIST2_TRUST

                if sem_sim >= dist2_threshold:
                    adjacency[i, j] = sem_sim
                    adjacency[j, i] = sem_sim
                    temporal_matched += 1
                    temporal_semantic_only += 1
                    is_matched = True

                    if num_inliers >= MIN_INLIERS_FOR_DIRECTION and direction != (0.0, 0.0):
                        directions[(i, j)] = direction
                        logger.info(
                            "    ✓ MATCHED (dist2 trust: semantic %.3f >= %.2f, "
                            "same_label=%s, dir=(%.2f,%.2f) from %s inliers)",
                            sem_sim,
                            dist2_threshold,
                            "yes" if same_room_label else "no",
                            direction[0],
                            direction[1],
                            num_inliers,
                        )
                    else:
                        logger.info(
                            "    ✓ MATCHED (dist2 trust: semantic %.3f >= %.2f, same_label=%s, no direction)",
                            sem_sim,
                            dist2_threshold,
                            "yes" if same_room_label else "no",
                        )
                else:
                    logger.info(
                        "    ✗ Dist2 semantic below trust threshold (%.3f < %.2f)",
                        sem_sim,
                        dist2_threshold,
                    )
            else:
                cross_note = " [cross-room]" if is_cross_room else ""
                logger.info(f"    ✗ No geometric match ({num_inliers} inliers < {min_inliers_required}){cross_note}")
        else:
            logger.info(f"    ✗ No match (semantic {sem_sim:.3f} < {TEMPORAL_GEOMETRIC_THRESHOLD})")

        # Track for database storage
        pair_source = "both" if (i, j) in semantic_pairs else "temporal_window"
        similarity_records.append({
            "photo_a_id": photo_ids[min(i, j)],
            "photo_b_id": photo_ids[max(i, j)],
            "pair_source": pair_source,
            "dinov2_similarity": float(sem_sim),
            "geometric_matches": num_matches,
            "geometric_inliers": num_inliers,
            "geometric_score": float(geo_score) if geo_score else None,
            "direction_dx": float(direction[0]) if direction != (0.0, 0.0) else None,
            "direction_dy": float(direction[1]) if direction != (0.0, 0.0) else None,
            "is_connected": 1 if is_matched else 0,
        })

    logger.info(f"Stage 2a: {temporal_matched}/{len(temporal_pairs)} temporal pairs matched "
               f"(semantic-only={temporal_semantic_only}, geometric={temporal_geometric})")

    # -------------------------------------------------------------------------
    # Stage 2b: Non-temporal pairs - use GEOMETRIC verification (pixel overlap)
    # -------------------------------------------------------------------------
    # Only check semantic pairs that aren't already covered by temporal
    geometric_pairs = semantic_pairs - temporal_pairs
    logger.info(f"Stage 2b: Geometric verification on {len(geometric_pairs)} non-temporal pairs...")
    geometric_matched = 0

    skipped_low_semantic = 0
    skipped_cross_room = 0
    semantic_recovered = 0
    for idx, (i, j) in enumerate(sorted(geometric_pairs)):
        sem_sim = similarity[i, j]

        # Check if rooms are different - SKIP cross-room pairs entirely
        room_i = room_labels[i] if room_labels else None
        room_j = room_labels[j] if room_labels else None
        is_cross_room = rooms_are_different(room_i, room_j)

        if is_cross_room:
            logger.info(f"  [{idx+1}/{len(geometric_pairs)}] SKIP {photo_ids[i]} <-> {photo_ids[j]} "
                       f"(different rooms: {room_i} vs {room_j})")
            skipped_cross_room += 1
            similarity_records.append({
                "photo_a_id": photo_ids[min(i, j)],
                "photo_b_id": photo_ids[max(i, j)],
                "pair_source": "dinov2_topk_cross_room",
                "dinov2_similarity": float(sem_sim),
                "geometric_matches": None,
                "geometric_inliers": None,
                "geometric_score": None,
                "direction_dx": None,
                "direction_dy": None,
                "is_connected": 0,
            })
            continue

        # Skip pairs with very low semantic similarity (obviously unrelated)
        if sem_sim < MIN_SEMANTIC_FOR_GEOMETRIC:
            logger.info(f"  [{idx+1}/{len(geometric_pairs)}] SKIP {photo_ids[i]} <-> {photo_ids[j]} "
                       f"(DINOv2 similarity={sem_sim:.3f} < {MIN_SEMANTIC_FOR_GEOMETRIC})")
            skipped_low_semantic += 1
            # Still record for database but mark as not matched
            similarity_records.append({
                "photo_a_id": photo_ids[min(i, j)],
                "photo_b_id": photo_ids[max(i, j)],
                "pair_source": "dinov2_topk_skipped",
                "dinov2_similarity": float(sem_sim),
                "geometric_matches": None,
                "geometric_inliers": None,
                "geometric_score": None,
                "direction_dx": None,
                "direction_dy": None,
                "is_connected": 0,
            })
            continue

        # Photos far apart in sequence need stronger evidence to be connected
        # (prevents clustering photos from different physical locations with same room label)
        position_gap = abs(j - i)
        if position_gap >= POSITION_GAP_THRESHOLD:
            min_inliers = MIN_INLIERS_FAR_APART
            gap_note = f", gap={position_gap} -> need {min_inliers} inliers"
        else:
            min_inliers = MIN_INLIERS_FOR_OVERLAP
            gap_note = ""

        logger.info(f"  [{idx+1}/{len(geometric_pairs)}] Checking {photo_ids[i]} <-> {photo_ids[j]} "
                   f"(DINOv2 similarity={sem_sim:.3f}{gap_note})...")

        num_matches, num_inliers, score, direction = match_image_pair(images[i], images[j])

        is_matched = num_inliers >= min_inliers
        pair_source = "dinov2_topk"
        if is_matched:
            adjacency[i, j] = max(adjacency[i, j], score)  # Keep higher score
            adjacency[j, i] = max(adjacency[j, i], score)
            edge_has_geometry[i, j] = True
            edge_has_geometry[j, i] = True
            directions[(i, j)] = direction  # Store direction for ordering
            geometric_matched += 1
            dir_str = f"dir=({direction[0]:.2f},{direction[1]:.2f})" if direction != (0.0, 0.0) else "dir=unknown"
            logger.info(f"    ✓ MATCHED: {num_matches} matches, {num_inliers} inliers, score={score:.3f}, {dir_str}")
        else:
            same_room_label = (
                normalize_room_label(room_i) != ""
                and normalize_room_label(room_i) != "unknown"
                and normalize_room_label(room_i) == normalize_room_label(room_j)
            )
            exterior_pair = is_exterior_like(room_i) and is_exterior_like(room_j)
            if (
                same_room_label
                and exterior_pair
                and position_gap >= EXTERIOR_LONG_GAP_MIN
                and sem_sim >= EXTERIOR_LONG_GAP_SEMANTIC_TRUST
            ):
                adjacency[i, j] = max(adjacency[i, j], sem_sim)
                adjacency[j, i] = max(adjacency[j, i], sem_sim)
                geometric_matched += 1
                semantic_recovered += 1
                is_matched = True
                pair_source = "dinov2_topk_semantic_recovery"
                logger.info(
                    "    ✓ MATCHED (exterior long-gap semantic recovery: sem=%.3f, gap=%s, rooms=%s/%s)",
                    sem_sim,
                    position_gap,
                    room_i,
                    room_j,
                )
            else:
                logger.info(f"    ✗ No match: {num_matches} matches, {num_inliers} inliers < {min_inliers}")

        # Track for database storage
        similarity_records.append({
            "photo_a_id": photo_ids[min(i, j)],
            "photo_b_id": photo_ids[max(i, j)],
            "pair_source": pair_source,
            "dinov2_similarity": float(sem_sim),
            "geometric_matches": num_matches,
            "geometric_inliers": num_inliers,
            "geometric_score": float(score) if score else None,
            "direction_dx": float(direction[0]) if direction != (0.0, 0.0) else None,
            "direction_dy": float(direction[1]) if direction != (0.0, 0.0) else None,
            "is_connected": 1 if is_matched else 0,
        })

    logger.info(
        "Stage 2b: %s/%s non-temporal pairs matched (%s cross-room, %s low-semantic, %s semantic recoveries)",
        geometric_matched,
        len(geometric_pairs),
        skipped_cross_room,
        skipped_low_semantic,
        semantic_recovered,
    )
    logger.info(f"Total edges: {temporal_matched + geometric_matched} "
               f"(temporal={temporal_matched}, geometric={geometric_matched})")

    # Prune weak semantic-only bridge edges that can cause transitive false merges.
    prune_weak_semantic_bridges(adjacency, similarity, edge_has_geometry, photo_ids)

    # -------------------------------------------------------------------------
    # Stage 3: Connected components = final clusters
    # -------------------------------------------------------------------------
    logger.info("Stage 3: Finding connected components...")

    # Convert to sparse matrix with threshold
    sparse_adj = csr_matrix(adjacency > OVERLAP_THRESHOLD)
    n_components, labels = connected_components(sparse_adj, directed=False)

    # Group photo INDICES by component (not photo_ids yet)
    clusters_by_label = defaultdict(list)
    for i, label in enumerate(labels):
        clusters_by_label[label].append(i)

    # -------------------------------------------------------------------------
    # Stage 4: Order photos within each cluster for smooth transitions
    # -------------------------------------------------------------------------
    logger.info("Stage 4: Ordering photos within clusters for transitions...")

    ordered_clusters = []
    for label, cluster_indices in clusters_by_label.items():
        if len(cluster_indices) == 1:
            # Single photo - no ordering needed
            ordered_clusters.append([photo_ids[cluster_indices[0]]])
        else:
            # Order using direction-aware algorithm
            # Returns (ordered_main_chain, isolated_photos)
            ordered_main, isolated = order_cluster_for_transitions(
                cluster_indices,
                photo_ids,
                adjacency,
                directions,
                min_score=MIN_TRANSITION_SCORE,
            )
            if ordered_main:
                ordered_clusters.append(ordered_main)
                logger.info(f"  Cluster {label}: ordered {len(cluster_indices)} -> main={len(ordered_main)}")

            # Add isolated photos as singleton clusters (they couldn't be ordered but are connected)
            for iso_pid in isolated:
                ordered_clusters.append([iso_pid])
                logger.info(f"  Cluster {label}: isolated photo {iso_pid} -> own cluster")

    # Stage 4b: Enforce hard transition quality constraints on ordered chains.
    ordered_clusters = enforce_transition_quality(
        ordered_clusters=ordered_clusters,
        photo_ids=photo_ids,
        adjacency=adjacency,
        similarity=similarity,
        edge_has_geometry=edge_has_geometry,
        similarity_records=similarity_records,
        room_labels=room_labels,
    )

    # -------------------------------------------------------------------------
    # Stage 5: Deduplicate and split large clusters
    # -------------------------------------------------------------------------
    logger.info(f"Stage 5: Deduplicating and splitting clusters (max {max_cluster_size} photos each)...")

    final_clusters = []
    duplicate_of_map: Dict[int, int] = {}
    for cluster in ordered_clusters:
        # Deduplicate obvious same-angle shots.
        split_clusters, cluster_duplicate_map = deduplicate_and_split_cluster(
            cluster,
            photo_ids,
            embeddings,
            adjacency,
            max_size=max_cluster_size,
            keep_duplicate_singletons=KEEP_DUPLICATE_SINGLETON_CLUSTERS,
        )
        final_clusters.extend(split_clusters)
        duplicate_of_map.update(cluster_duplicate_map)

    logger.info(
        f"Final result: {n} photos -> {len(final_clusters)} clusters "
        f"(sizes: {[len(c) for c in final_clusters]})"
    )
    if duplicate_of_map:
        if KEEP_DUPLICATE_SINGLETON_CLUSTERS:
            logger.info("Marked %s photos as duplicates (kept as singleton clusters)", len(duplicate_of_map))
        else:
            logger.info("Removed %s obvious duplicates from final clustering output", len(duplicate_of_map))

    # Save similarity records to database if session provided
    if db_session is not None and job_id is not None and similarity_records:
        from sqlalchemy import text

        # Backward-compatible schema detection:
        # some environments may not yet have direction_dx/direction_dy migrated.
        supports_direction = False
        try:
            col_rows = db_session.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'photo_similarities'
                      AND table_schema = current_schema()
                    """
                )
            ).fetchall()
            available_cols = {str(row[0]) for row in col_rows}
            supports_direction = {
                "direction_dx",
                "direction_dy",
            }.issubset(available_cols)
        except Exception as schema_err:
            logger.warning("Could not inspect photo_similarities schema: %s", schema_err)

        logger.info(f"Saving {len(similarity_records)} similarity records to database...")

        # Re-running clustering for a job should replace previous similarity rows.
        db_session.execute(
            text("DELETE FROM photo_similarities WHERE job_id = :job_id"),
            {"job_id": job_id},
        )

        if supports_direction:
            insert_sql = text(
                """
                INSERT INTO photo_similarities (
                    job_id, photo_a_id, photo_b_id, pair_source,
                    dinov2_similarity, geometric_matches, geometric_inliers,
                    geometric_score, direction_dx, direction_dy, is_connected
                ) VALUES (
                    :job_id, :photo_a_id, :photo_b_id, :pair_source,
                    :dinov2_similarity, :geometric_matches, :geometric_inliers,
                    :geometric_score, :direction_dx, :direction_dy, :is_connected
                )
                """
            )
        else:
            insert_sql = text(
                """
                INSERT INTO photo_similarities (
                    job_id, photo_a_id, photo_b_id, pair_source,
                    dinov2_similarity, geometric_matches, geometric_inliers,
                    geometric_score, is_connected
                ) VALUES (
                    :job_id, :photo_a_id, :photo_b_id, :pair_source,
                    :dinov2_similarity, :geometric_matches, :geometric_inliers,
                    :geometric_score, :is_connected
                )
                """
            )

        records_for_insert = []
        for record in similarity_records:
            payload = {
                "job_id": job_id,
                "photo_a_id": record["photo_a_id"],
                "photo_b_id": record["photo_b_id"],
                "pair_source": record["pair_source"],
                "dinov2_similarity": record["dinov2_similarity"],
                "geometric_matches": record["geometric_matches"],
                "geometric_inliers": record["geometric_inliers"],
                "geometric_score": record["geometric_score"],
                "is_connected": record["is_connected"],
            }
            if supports_direction:
                payload["direction_dx"] = record.get("direction_dx")
                payload["direction_dy"] = record.get("direction_dy")
            records_for_insert.append(payload)

        db_session.execute(insert_sql, records_for_insert)
        db_session.commit()
        logger.info(
            "Saved %s similarity records (direction columns: %s)",
            len(similarity_records),
            "enabled" if supports_direction else "not available",
        )

    if return_metadata:
        return final_clusters, {
            "duplicate_of_map": duplicate_of_map,
            "duplicates_dropped": not KEEP_DUPLICATE_SINGLETON_CLUSTERS,
        }
    return final_clusters


# ============================================================================
# MAIN PIPELINE ENTRY POINT
# ============================================================================

def cluster_photos_optimized(
    images: List[Image.Image],
    photo_ids: List[int],
    s3_client=None,
    db_session=None,
    job_id: int = None,
    room_labels: List[str] = None,
    return_metadata: bool = False,
) -> List[List[int]] | Tuple[List[List[int]], Dict[str, object]]:
    """Run optimized graph-based clustering pipeline.

    Uses the "propose + verify" pattern:
    1. DINOv2 embeddings propose candidate edges (top-K similar pairs)
    2. Geometric verification (LoFTR/ORB) confirms overlap
    3. Connected components = final clusters

    This is cleaner than semantic-first clustering because:
    - No artificial semantic boundaries to fix later
    - Semantic similarity is a filter, not ground truth
    - O(N × K) geometric checks instead of O(N²)

    Args:
        images: List of PIL Images
        photo_ids: List of photo IDs
        s3_client: S3 client (unused, for API compatibility)
        db_session: Optional SQLAlchemy session for saving similarity data
        job_id: Optional job ID for saving similarity data
        room_labels: Optional list of room labels for each photo (used to penalize cross-room connections)

    Returns:
        Final clusters, or (clusters, metadata) when return_metadata=True.
    """
    n = len(images)
    # Adaptive semantic neighborhood size:
    # - small jobs: keep checks tight
    # - medium jobs: balanced recall/compute
    # - large jobs: broader proposal set
    if n <= 80:
        candidate_k = 4
    elif n <= 140:
        candidate_k = 5
    else:
        candidate_k = 6

    logger.info("Adaptive candidate k=%s for %s photos", candidate_k, n)

    return cluster_photos_graph_based(
        images,
        photo_ids,
        k=candidate_k,
        max_cluster_size=3,  # Limit to 3 best photos per cluster
        db_session=db_session,
        job_id=job_id,
        room_labels=room_labels,
        return_metadata=return_metadata,
    )
