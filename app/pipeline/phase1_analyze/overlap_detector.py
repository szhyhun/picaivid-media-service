"""Visual overlap detection for photo clustering.

Uses ORB feature matching WITH GEOMETRIC VERIFICATION (RANSAC) to detect
actual scene overlap between images. This filters out false matches from
similar textures (e.g., wood floors, white walls) that appear in different rooms.

Uses FUNDAMENTAL MATRIX (not homography) because:
- Homography assumes planar scenes (walls only)
- Fundamental matrix works for any 3D scene (rooms with furniture, depth)
- Better for real estate photos with viewpoint changes

Key insight: Two photos of DIFFERENT rooms can have many ORB matches if they
have similar textures. But those matches won't be geometrically consistent
(won't satisfy epipolar constraints). RANSAC filters these out.
"""
import logging
from typing import List, Tuple, TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from app.db.models import JobPhoto

logger = logging.getLogger(__name__)

# Overlap detection thresholds - TUNED FOR REAL ESTATE PHOTOS
# Real estate photos often have:
# - 20-40% visual overlap between adjacent shots
# - Wide-angle lens distortion
# - Significant viewpoint changes (not just rotation)
MIN_MATCHES_FOR_OVERLAP = 10   # Minimum raw matches before geometric verification
MIN_INLIERS_FOR_OVERLAP = 5    # Minimum RANSAC inliers to confirm overlap
RANSAC_REPROJ_THRESHOLD = 3.0  # Pixels for fundamental matrix (epipolar distance)
OVERLAP_THRESHOLD = 0.08       # Minimum overlap score to connect photos


def compute_pairwise_overlap(
    images: List[Image.Image],
    photo_ids: List[int] = None,
) -> np.ndarray:
    """Compute visual overlap scores between all image pairs.

    Uses ORB feature matching WITH GEOMETRIC VERIFICATION (RANSAC).

    The key insight: Similar textures (wood floors, white walls) create
    many ORB matches between DIFFERENT rooms. But these matches won't be
    geometrically consistent - they won't form a valid transformation.

    RANSAC filters out these false matches by finding only the subset
    of matches that are geometrically consistent (inliers).

    Args:
        images: List of PIL Images
        photo_ids: Optional list of photo IDs for logging

    Returns:
        NxN matrix of overlap scores (0.0 = no overlap, 1.0 = high overlap)
    """
    n = len(images)
    if n <= 1:
        return np.ones((n, n))

    overlap_matrix = np.zeros((n, n))
    np.fill_diagonal(overlap_matrix, 1.0)

    # Initialize ORB detector with more features for better matching
    orb = cv2.ORB_create(nfeatures=1000)

    # Use BFMatcher without crossCheck for ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Compute keypoints and descriptors for all images
    keypoints_list = []
    descriptors_list = []

    for i, img in enumerate(images):
        # Convert PIL to grayscale numpy array
        img_gray = np.array(img.convert("L"))

        # Detect keypoints and compute descriptors
        kp, desc = orb.detectAndCompute(img_gray, None)
        keypoints_list.append(kp)
        descriptors_list.append(desc)

        if photo_ids:
            logger.debug(f"Photo {photo_ids[i]}: {len(kp) if kp else 0} keypoints")

    # Compute pairwise matches with geometric verification
    for i in range(n):
        for j in range(i + 1, n):
            desc_i = descriptors_list[i]
            desc_j = descriptors_list[j]
            kp_i = keypoints_list[i]
            kp_j = keypoints_list[j]

            # Skip if either image has no descriptors
            if desc_i is None or desc_j is None:
                continue

            # Match descriptors using knnMatch for ratio test
            try:
                matches = bf.knnMatch(desc_i, desc_j, k=2)
            except cv2.error:
                continue

            # Apply Lowe's ratio test to filter ambiguous matches
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n_match = match_pair
                    if m.distance < 0.75 * n_match.distance:
                        good_matches.append(m)

            if len(good_matches) < MIN_MATCHES_FOR_OVERLAP:
                if photo_ids:
                    logger.debug(
                        f"Overlap {photo_ids[i]} <-> {photo_ids[j]}: "
                        f"only {len(good_matches)} good matches (need {MIN_MATCHES_FOR_OVERLAP})"
                    )
                continue

            # GEOMETRIC VERIFICATION with RANSAC using FUNDAMENTAL MATRIX
            # Fundamental matrix works for 3D scenes (unlike homography which assumes planar)
            # Extract matched point coordinates
            src_pts = np.float32([kp_i[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            dst_pts = np.float32([kp_j[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

            # Find fundamental matrix with RANSAC
            # This filters out matches that don't satisfy epipolar geometry
            _, mask = cv2.findFundamentalMat(
                src_pts, dst_pts,
                cv2.FM_RANSAC,
                RANSAC_REPROJ_THRESHOLD,  # Epipolar distance threshold in pixels
                0.99  # Confidence level
            )

            if mask is None:
                if photo_ids:
                    logger.debug(
                        f"Overlap {photo_ids[i]} <-> {photo_ids[j]}: "
                        f"RANSAC failed (no valid fundamental matrix)"
                    )
                continue

            # Count inliers (geometrically consistent matches)
            # mask is Nx1 array where 1 = inlier, 0 = outlier
            inliers = int(mask.ravel().sum()) if mask is not None else 0

            if inliers < MIN_INLIERS_FOR_OVERLAP:
                if photo_ids:
                    logger.debug(
                        f"Overlap {photo_ids[i]} <-> {photo_ids[j]}: "
                        f"only {inliers} inliers (need {MIN_INLIERS_FOR_OVERLAP}) - "
                        f"likely different scenes with similar textures"
                    )
                continue

            # Compute overlap score based on INLIER count (not raw matches)
            # Normalize by the number of good matches to get inlier ratio
            inlier_ratio = inliers / len(good_matches)

            # Also factor in absolute inlier count
            # More inliers = higher confidence of real overlap
            max_kp = max(len(kp_i), len(kp_j))
            coverage = inliers / max_kp if max_kp > 0 else 0

            # Combined score: inlier ratio * coverage bonus
            score = inlier_ratio * (1 + coverage)
            score = min(1.0, score)  # Cap at 1.0

            overlap_matrix[i, j] = score
            overlap_matrix[j, i] = score

            if photo_ids:
                logger.info(
                    f"Overlap {photo_ids[i]} <-> {photo_ids[j]}: "
                    f"{len(good_matches)} matches -> {inliers} inliers "
                    f"({inlier_ratio:.1%}), score={score:.3f} ✓"
                )

    return overlap_matrix


def cluster_by_overlap(
    photos: List["JobPhoto"],
    overlap_matrix: np.ndarray,
    threshold: float = OVERLAP_THRESHOLD,
    max_cluster_size: int = 4,
) -> List[List["JobPhoto"]]:
    """Cluster photos into groups with visual overlap using single-linkage clustering.

    Uses SINGLE-LINKAGE to build camera paths through overlapping photos.
    This allows chain connections: if A overlaps B and B overlaps C,
    then A-B-C are clustered together even if A doesn't directly overlap C.

    This is ideal for building sequences like:
    - Corner of room → middle of room → other corner

    The geometric verification (fundamental matrix) prevents false chains
    between different rooms.

    Args:
        photos: List of JobPhoto objects
        overlap_matrix: NxN overlap score matrix
        threshold: Minimum overlap score to connect photos
        max_cluster_size: Maximum photos per cluster (clips work best with 2-4 images)

    Returns:
        List of photo groups (clusters that can be interpolated)
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    n = len(photos)
    if n <= 1:
        return [photos] if photos else []

    # Convert overlap similarity to distance (1 - overlap)
    # Clip to avoid negative distances from diagonal
    distance_matrix = 1.0 - np.clip(overlap_matrix, 0, 1)
    np.fill_diagonal(distance_matrix, 0)

    # Convert to condensed distance matrix for linkage
    condensed = squareform(distance_matrix, checks=False)

    # Hierarchical clustering with SINGLE linkage
    # Single linkage: clusters merge if ANY member pair overlaps
    # This builds chains: A-B-C where A→B and B→C overlap
    Z = linkage(condensed, method='single')

    # Cut tree at distance threshold (1 - overlap_threshold)
    # Photos are connected if they have at least threshold overlap
    distance_threshold = 1.0 - threshold
    labels = fcluster(Z, t=distance_threshold, criterion='distance')

    # Group photos by cluster label
    clusters_dict = {}
    for i, label in enumerate(labels):
        if label not in clusters_dict:
            clusters_dict[label] = []
        clusters_dict[label].append(photos[i])

    clusters = list(clusters_dict.values())

    # Split any clusters that are too large (LTX-2 works best with 2-4 keyframes)
    final_clusters = []
    for cluster in clusters:
        if len(cluster) > max_cluster_size:
            # Split large clusters into smaller ones
            split_clusters = _split_large_cluster(
                cluster, photos, overlap_matrix, max_cluster_size
            )
            final_clusters.extend(split_clusters)
        else:
            final_clusters.append(cluster)

    # Log cluster info
    logger.info(
        f"Split {n} photos into {len(final_clusters)} overlap clusters: "
        f"{[len(c) for c in final_clusters]}"
    )

    return final_clusters


def _split_large_cluster(
    cluster: List["JobPhoto"],
    all_photos: List["JobPhoto"],
    overlap_matrix: np.ndarray,
    max_size: int,
) -> List[List["JobPhoto"]]:
    """Split a large cluster into smaller chunks.

    Splits based on overlap ordering - groups consecutive overlapping photos.

    Args:
        cluster: Photos to split
        all_photos: Full photo list (for matrix indexing)
        overlap_matrix: Full overlap matrix
        max_size: Maximum cluster size

    Returns:
        List of smaller clusters
    """
    if len(cluster) <= max_size:
        return [cluster]

    # Get indices in original photo list
    indices = [all_photos.index(p) for p in cluster]
    sub_matrix = overlap_matrix[np.ix_(indices, indices)]

    # Order by overlap for optimal splitting
    ordered = order_photos_by_overlap(cluster, sub_matrix)

    # Split into chunks of max_size
    chunks = []
    for i in range(0, len(ordered), max_size):
        chunk = ordered[i:i + max_size]
        if len(chunk) >= 2:  # Need at least 2 photos for interpolation
            chunks.append(chunk)
        elif chunks:  # Add single photo to previous chunk
            chunks[-1].extend(chunk)
        else:
            chunks.append(chunk)

    return chunks


def order_photos_by_overlap(
    photos: List["JobPhoto"],
    overlap_matrix: np.ndarray,
) -> List["JobPhoto"]:
    """Order photos in a cluster for optimal interpolation sequence.

    Uses a greedy approach to order photos by maximum overlap,
    creating a path through the viewpoints.

    Args:
        photos: List of JobPhoto objects in a cluster
        overlap_matrix: NxN overlap score matrix (for this cluster)

    Returns:
        Ordered list of photos for video generation
    """
    n = len(photos)
    if n <= 2:
        return photos

    # Greedy ordering: start from first photo, always go to most overlapping neighbor
    ordered = [photos[0]]
    remaining = set(range(1, n))

    current = 0
    while remaining:
        # Find most overlapping unvisited neighbor
        best_neighbor = None
        best_score = -1

        for neighbor in remaining:
            score = overlap_matrix[current, neighbor]
            if score > best_score:
                best_score = score
                best_neighbor = neighbor

        if best_neighbor is not None:
            ordered.append(photos[best_neighbor])
            remaining.remove(best_neighbor)
            current = best_neighbor
        else:
            # No overlap found, just add remaining in order
            for idx in sorted(remaining):
                ordered.append(photos[idx])
            break

    return ordered


def compute_overlap_for_photos(
    photos: List["JobPhoto"],
    s3_client,
) -> Tuple[np.ndarray, List[Image.Image]]:
    """Download photos and compute overlap matrix.

    Args:
        photos: List of JobPhoto objects with s3_uri
        s3_client: S3 client for downloading images

    Returns:
        Tuple of (overlap_matrix, images)
    """
    # Download all images
    images = []
    photo_ids = []

    for photo in photos:
        try:
            img = s3_client.download_image(photo.s3_uri)
            # Resize for faster feature detection
            img = img.resize((512, 384), Image.Resampling.LANCZOS)
            images.append(img)
            photo_ids.append(photo.id)
        except Exception as e:
            logger.warning(f"Failed to download photo {photo.id}: {e}")
            # Use placeholder black image
            images.append(Image.new("RGB", (512, 384), color="black"))
            photo_ids.append(photo.id)

    # Compute overlap
    overlap_matrix = compute_pairwise_overlap(images, photo_ids)

    return overlap_matrix, images
