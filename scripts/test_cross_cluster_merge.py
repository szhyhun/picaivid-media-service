#!/usr/bin/env python3
"""Test clustering against expected photo groupings.

Compares clustering results to expected groups.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.models import Job, JobPhoto
from app.services.storage.s3_client import S3Client
from app.pipeline.phase1_analyze.learned_matching import cluster_photos_optimized

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# EXPECTED CLUSTERS (from user analysis)
# ============================================================================
# Photos that SHOULD be in the same cluster
EXPECTED_SAME_CLUSTER = [
    [569, 570],           # Front yard - good pair
    [581, 582],           # Dining room - same cluster
    [595, 596, 597],      # Living/dining room transition - same cluster
    [601, 602],           # Bathroom - same cluster
    [605, 606],           # Bedroom - same cluster
]

# Photos that should be in DIFFERENT clusters
EXPECTED_DIFFERENT_CLUSTERS = [
    ([569, 570], [572, 573]),  # 572, 573 should be separate from 569, 570
]


def check_same_cluster(photo_ids: list, clusters: list) -> tuple:
    """Check if all photo_ids are in the same cluster.

    Returns (success, cluster_info)
    """
    # Find which cluster each photo is in
    photo_to_cluster = {}
    for cluster_idx, cluster in enumerate(clusters):
        for pid in cluster:
            photo_to_cluster[pid] = cluster_idx

    # Check if all expected photos are in the same cluster
    cluster_indices = set()
    missing = []
    for pid in photo_ids:
        if pid in photo_to_cluster:
            cluster_indices.add(photo_to_cluster[pid])
        else:
            missing.append(pid)

    if missing:
        return False, f"Photos {missing} not found in any cluster"

    if len(cluster_indices) == 1:
        cluster_idx = list(cluster_indices)[0]
        return True, f"All in cluster {cluster_idx}: {clusters[cluster_idx]}"
    else:
        info = []
        for pid in photo_ids:
            info.append(f"{pid}→cluster{photo_to_cluster[pid]}")
        return False, f"Split across clusters: {', '.join(info)}"


def check_different_clusters(group1: list, group2: list, clusters: list) -> tuple:
    """Check if group1 and group2 are in DIFFERENT clusters.

    Returns (success, info)
    """
    photo_to_cluster = {}
    for cluster_idx, cluster in enumerate(clusters):
        for pid in cluster:
            photo_to_cluster[pid] = cluster_idx

    # Get cluster for each group
    g1_clusters = set(photo_to_cluster.get(pid, -1) for pid in group1)
    g2_clusters = set(photo_to_cluster.get(pid, -1) for pid in group2)

    # Check if they share any cluster
    overlap = g1_clusters & g2_clusters
    if -1 in overlap:
        overlap.remove(-1)

    if overlap:
        return False, f"Groups share cluster(s): {overlap}"
    else:
        return True, f"Group1 in {g1_clusters}, Group2 in {g2_clusters}"


def main():
    engine = create_engine(settings.DATABASE_URL)
    Session = sessionmaker(bind=engine)
    db = Session()

    job = db.query(Job).order_by(Job.id.desc()).first()
    if not job:
        print("No jobs found")
        return

    print(f"\n{'='*70}")
    print(f"Testing clustering on Job {job.id}")
    print(f"{'='*70}")

    # Collect all photo IDs we need to test
    all_test_ids = set()
    for group in EXPECTED_SAME_CLUSTER:
        all_test_ids.update(group)
    for g1, g2 in EXPECTED_DIFFERENT_CLUSTERS:
        all_test_ids.update(g1)
        all_test_ids.update(g2)

    print(f"\nTest photo IDs: {sorted(all_test_ids)}")

    # Load only the photos we need
    photos = db.query(JobPhoto).filter(
        JobPhoto.job_id == job.id,
        JobPhoto.id.in_(all_test_ids)
    ).order_by(JobPhoto.id).all()

    print(f"Found {len(photos)} photos")

    # Download images
    s3_client = S3Client()
    images = []
    photo_ids = []
    photo_map = {}

    for photo in photos:
        try:
            img = s3_client.download_image(photo.s3_uri)
            img = img.resize((512, 384), Image.Resampling.LANCZOS)
            images.append(img)
            photo_ids.append(photo.id)
            photo_map[photo.id] = photo
            print(f"  Downloaded {photo.id}: {photo.room_label}")
        except Exception as e:
            logger.warning(f"Failed to download photo {photo.id}: {e}")

    print(f"\nDownloaded {len(images)} images")
    print()

    # Run clustering
    print("="*70)
    print("RUNNING CLUSTERING...")
    print("="*70)

    cluster_lists = cluster_photos_optimized(images, photo_ids, s3_client, db_session=db, job_id=job.id)

    print()
    print("="*70)
    print("CLUSTERING RESULTS:")
    print("="*70)

    for i, cluster in enumerate(cluster_lists):
        labels = []
        for pid in cluster:
            photo = photo_map.get(pid)
            if photo:
                labels.append(f"{pid}({photo.room_label})")
            else:
                labels.append(str(pid))
        print(f"  Cluster {i}: {', '.join(labels)}")

    print()
    print("="*70)
    print("VALIDATION:")
    print("="*70)

    # Check EXPECTED_SAME_CLUSTER
    print("\n1. Photos that SHOULD be in same cluster:")
    all_passed = True
    for group in EXPECTED_SAME_CLUSTER:
        success, info = check_same_cluster(group, cluster_lists)
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"   {status}: {group}")
        print(f"          {info}")
        if not success:
            all_passed = False

    # Check EXPECTED_DIFFERENT_CLUSTERS
    print("\n2. Photos that should be in DIFFERENT clusters:")
    for g1, g2 in EXPECTED_DIFFERENT_CLUSTERS:
        success, info = check_different_clusters(g1, g2, cluster_lists)
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"   {status}: {g1} vs {g2}")
        print(f"          {info}")
        if not success:
            all_passed = False

    print()
    print("="*70)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - tuning needed")
    print("="*70)

    db.close()


if __name__ == "__main__":
    main()
