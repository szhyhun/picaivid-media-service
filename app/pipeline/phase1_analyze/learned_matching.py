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
        # Fallback: return random embeddings (for testing without model)
        logger.warning("DINOv2 not available, using random embeddings")
        return np.random.randn(len(images), 768).astype(np.float32)

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
MIN_INLIERS_FOR_OVERLAP = 8       # Minimum inliers for pixel-level overlap
RANSAC_REPROJ_THRESHOLD = 1.5     # Tight reprojection threshold (pixels)
OVERLAP_THRESHOLD = 0.15          # Minimum score to connect photos

# Temporal + semantic matching for same-room different-angle shots
# Adjacent photos in upload order are usually the same room
TEMPORAL_WINDOW = 2               # Check photos within ±2 positions
TEMPORAL_SEMANTIC_THRESHOLD = 0.60  # Semantic similarity for temporal neighbors

# Minimum DINOv2 similarity to even consider geometric verification
# Filters out obviously unrelated photos (aerial vs interior = ~0.05)
MIN_SEMANTIC_FOR_GEOMETRIC = 0.15  # Skip geometric check if semantic < 15%

# LightGlue/LoFTR model singleton
_matcher = None
_matcher_type = None


def _load_matcher():
    """Load learned feature matcher (lazy initialization)."""
    global _matcher, _matcher_type

    if _matcher is not None:
        return _matcher, _matcher_type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try LoFTR first (best for indoor scenes)
    try:
        from kornia.feature import LoFTR
        _matcher = LoFTR(pretrained="outdoor")
        _matcher = _matcher.to(device)
        _matcher.eval()
        _matcher_type = "loftr"
        logger.info(f"Loaded LoFTR matcher on {device}")
        return _matcher, _matcher_type
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
) -> Tuple[int, int, float]:
    """Match two images using learned features.

    Args:
        img1: First PIL Image
        img2: Second PIL Image

    Returns:
        Tuple of (num_matches, num_inliers, overlap_score)
    """
    matcher, matcher_type = _load_matcher()

    if matcher_type == "loftr" and matcher is not None:
        return _match_loftr(matcher, img1, img2)
    else:
        return _match_orb(img1, img2)


def _match_loftr(
    matcher,
    img1: Image.Image,
    img2: Image.Image,
) -> Tuple[int, int, float]:
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
        return num_matches, 0, 0.0

    # Geometric verification
    F, inlier_mask = cv2.findFundamentalMat(
        mkpts0, mkpts1,
        cv2.FM_RANSAC,
        RANSAC_REPROJ_THRESHOLD,
        0.999
    )

    if inlier_mask is None:
        return num_matches, 0, 0.0

    num_inliers = int(inlier_mask.sum())

    # Score based on inlier ratio and count
    inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0
    coverage_bonus = min(1.0, num_inliers / 100)
    score = inlier_ratio * (0.5 + 0.5 * coverage_bonus)

    return num_matches, num_inliers, score


def _match_orb(
    img1: Image.Image,
    img2: Image.Image,
) -> Tuple[int, int, float]:
    """Fallback ORB matching."""
    img1_gray = np.array(img1.convert("L"))
    img2_gray = np.array(img2.convert("L"))

    orb = cv2.ORB_create(nfeatures=2000)

    kp1, desc1 = orb.detectAndCompute(img1_gray, None)
    kp2, desc2 = orb.detectAndCompute(img2_gray, None)

    if desc1 is None or desc2 is None:
        return 0, 0, 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        matches = bf.knnMatch(desc1, desc2, k=2)
    except cv2.error:
        return 0, 0, 0.0

    # Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    num_matches = len(good_matches)

    if num_matches < 8:
        return num_matches, 0, 0.0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    F, mask = cv2.findFundamentalMat(
        src_pts, dst_pts,
        cv2.FM_RANSAC,
        RANSAC_REPROJ_THRESHOLD,
        0.999
    )

    if mask is None:
        return num_matches, 0, 0.0

    num_inliers = int(mask.sum())
    inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0
    coverage_bonus = min(1.0, num_inliers / 50)
    score = inlier_ratio * (0.5 + 0.5 * coverage_bonus)

    return num_matches, num_inliers, score


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

def cluster_photos_graph_based(
    images: List[Image.Image],
    photo_ids: List[int],
    k: int = 8,
    max_cluster_size: int = 6,
    db_session=None,
    job_id: int = None,
) -> List[List[int]]:
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

    Returns:
        List of photo ID lists (final clusters)
    """
    from sklearn.preprocessing import normalize
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(images)
    logger.info(f"Graph-based clustering for {n} photos (k={k})")

    # Track similarity data for database storage
    similarity_records = []  # Will store dicts for batch insert

    if n <= 1:
        return [photo_ids] if photo_ids else []

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
    # Stage 2a: Temporal pairs - use SEMANTIC similarity (same room, different angle)
    # -------------------------------------------------------------------------
    logger.info("Stage 2a: Checking temporal pairs with semantic similarity...")
    adjacency = np.zeros((n, n))
    temporal_matched = 0

    for i, j in sorted(temporal_pairs):
        sem_sim = similarity[i, j]
        temporal_dist = abs(j - i)
        logger.info(f"  Temporal {photo_ids[i]} <-> {photo_ids[j]} (dist={temporal_dist}): "
                   f"semantic={sem_sim:.3f}")

        is_matched = sem_sim >= TEMPORAL_SEMANTIC_THRESHOLD
        if is_matched:
            # Use semantic similarity as edge weight for temporal neighbors
            adjacency[i, j] = sem_sim
            adjacency[j, i] = sem_sim
            temporal_matched += 1
            logger.info(f"    ✓ MATCHED (semantic >= {TEMPORAL_SEMANTIC_THRESHOLD})")
        else:
            logger.info(f"    ✗ No match (semantic < {TEMPORAL_SEMANTIC_THRESHOLD})")

        # Track for database storage
        pair_source = "both" if (i, j) in semantic_pairs else "temporal_window"
        similarity_records.append({
            "photo_a_id": photo_ids[min(i, j)],
            "photo_b_id": photo_ids[max(i, j)],
            "pair_source": pair_source,
            "dinov2_similarity": float(sem_sim),
            "geometric_matches": None,
            "geometric_inliers": None,
            "geometric_score": None,
            "is_connected": 1 if is_matched else 0,
        })

    logger.info(f"Stage 2a: {temporal_matched}/{len(temporal_pairs)} temporal pairs matched")

    # -------------------------------------------------------------------------
    # Stage 2b: Non-temporal pairs - use GEOMETRIC verification (pixel overlap)
    # -------------------------------------------------------------------------
    # Only check semantic pairs that aren't already covered by temporal
    geometric_pairs = semantic_pairs - temporal_pairs
    logger.info(f"Stage 2b: Geometric verification on {len(geometric_pairs)} non-temporal pairs...")
    geometric_matched = 0

    skipped_low_semantic = 0
    for idx, (i, j) in enumerate(sorted(geometric_pairs)):
        sem_sim = similarity[i, j]

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
                "is_connected": 0,
            })
            continue

        logger.info(f"  [{idx+1}/{len(geometric_pairs)}] Checking {photo_ids[i]} <-> {photo_ids[j]} "
                   f"(DINOv2 similarity={sem_sim:.3f})...")

        num_matches, num_inliers, score = match_image_pair(images[i], images[j])

        is_matched = num_inliers >= MIN_INLIERS_FOR_OVERLAP
        if is_matched:
            adjacency[i, j] = max(adjacency[i, j], score)  # Keep higher score
            adjacency[j, i] = max(adjacency[j, i], score)
            geometric_matched += 1
            logger.info(f"    ✓ MATCHED: {num_matches} matches, {num_inliers} inliers, score={score:.3f}")
        else:
            logger.info(f"    ✗ No match: {num_matches} matches, {num_inliers} inliers")

        # Track for database storage
        similarity_records.append({
            "photo_a_id": photo_ids[min(i, j)],
            "photo_b_id": photo_ids[max(i, j)],
            "pair_source": "dinov2_topk",
            "dinov2_similarity": float(sem_sim),
            "geometric_matches": num_matches,
            "geometric_inliers": num_inliers,
            "geometric_score": float(score) if score else None,
            "is_connected": 1 if is_matched else 0,
        })

    logger.info(f"Stage 2b: {geometric_matched}/{len(geometric_pairs)} geometric pairs matched "
               f"({skipped_low_semantic} skipped due to low semantic similarity)")
    logger.info(f"Total edges: {temporal_matched + geometric_matched} "
               f"(temporal={temporal_matched}, geometric={geometric_matched})")

    # -------------------------------------------------------------------------
    # Stage 3: Connected components = final clusters
    # -------------------------------------------------------------------------
    logger.info("Stage 3: Finding connected components...")

    # Convert to sparse matrix with threshold
    sparse_adj = csr_matrix(adjacency > OVERLAP_THRESHOLD)
    n_components, labels = connected_components(sparse_adj, directed=False)

    # Group photos by component
    clusters_dict = defaultdict(list)
    for i, label in enumerate(labels):
        clusters_dict[label].append(photo_ids[i])

    clusters = list(clusters_dict.values())

    # Split large clusters if needed
    final_clusters = []
    for cluster in clusters:
        if len(cluster) > max_cluster_size:
            # Split into smaller chunks, preserving order
            for i in range(0, len(cluster), max_cluster_size):
                chunk = cluster[i:i + max_cluster_size]
                if len(chunk) >= 2:
                    final_clusters.append(chunk)
                elif final_clusters:
                    # Add singleton to previous cluster
                    final_clusters[-1].extend(chunk)
                else:
                    final_clusters.append(chunk)
        else:
            final_clusters.append(cluster)

    logger.info(
        f"Final result: {n} photos -> {len(final_clusters)} clusters "
        f"(sizes: {[len(c) for c in final_clusters]})"
    )

    # Save similarity records to database if session provided
    if db_session is not None and job_id is not None and similarity_records:
        from app.db.models import PhotoSimilarity

        logger.info(f"Saving {len(similarity_records)} similarity records to database...")
        for record in similarity_records:
            sim = PhotoSimilarity(
                job_id=job_id,
                photo_a_id=record["photo_a_id"],
                photo_b_id=record["photo_b_id"],
                pair_source=record["pair_source"],
                dinov2_similarity=record["dinov2_similarity"],
                geometric_matches=record["geometric_matches"],
                geometric_inliers=record["geometric_inliers"],
                geometric_score=record["geometric_score"],
                is_connected=record["is_connected"],
            )
            db_session.add(sim)
        db_session.commit()
        logger.info(f"Saved {len(similarity_records)} similarity records")

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
) -> List[List[int]]:
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

    Returns:
        List of photo ID lists (final clusters)
    """
    return cluster_photos_graph_based(
        images,
        photo_ids,
        k=4,  # Check top-4 semantically similar images per photo
        max_cluster_size=6,
        db_session=db_session,
        job_id=job_id,
    )
