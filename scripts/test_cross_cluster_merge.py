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

# -----------------------------------------------------------------------------
# TEST SET 1: Original listing (photo IDs 569-705)
# -----------------------------------------------------------------------------
EXPECTED_SAME_CLUSTER = [
    [569, 570],           # Front yard - good pair
    [581, 582],           # Dining room - same cluster
    [595, 596, 597],      # Living/dining room transition - same cluster
    [601, 602],           # Bathroom - same cluster
    [605, 606],           # Bedroom - same cluster
    [699, 700, 701, 702], # pan_left sequence - room A
    [703, 704, 705],      # different room B (should NOT merge with 699-702)
]

EXPECTED_DIFFERENT_CLUSTERS = [
    ([569, 570], [572, 573]),  # 572, 573 should be separate from 569, 570
    ([699, 700, 701, 702], [703, 704, 705]),  # Different rooms despite being adjacent in upload
]

# -----------------------------------------------------------------------------
# TEST SET 2: Listing with photo IDs 861-912
# Goal: Max 3 photos per cluster, remove duplicates, proper ordering
# NOTE: Photos 861, 906, 907, 910, 911 not in DB - removed from tests
# NOTE: Max cluster size is 3, so some photos may be removed during dedup
# -----------------------------------------------------------------------------
EXPECTED_SAME_CLUSTER_SET2 = [
    # Living room photos 883-889 form one connected component due to overlap
    # After dedup (max 3), some photos will be removed - test remaining pairs
    [887, 889],           # Living room - these have geometric overlap and should stay together
    [890, 891],           # First bedroom
    [897, 898],           # Second bedroom (DIFFERENT from 890-891)
    # [901, 903] - basement photos have 0 geometric inliers, so they're separate (correct)
]

EXPECTED_DIFFERENT_CLUSTERS_SET2 = [
    ([890, 891], [897, 898]),           # Two different bedrooms
    ([908], [912]),                     # Two different patios
    ([901], [903]),                     # Basement photos - no geometric overlap, should be separate
]

# Expected ordering: endpoint (edge photo) should be first or last
# The order should create a smooth transition (camera movement in one direction)
EXPECTED_ORDER_SET2 = [
    {"photos": [887, 889], "expected_start": 887, "note": "887 should be endpoint"},
    {"photos": [890, 891], "expected_start": 890, "note": "890 should be endpoint (first bedroom)"},
]

# Sequence tests: verify that photos are ordered for smooth transitions
# These test that the order within a cluster makes sense for video generation
EXPECTED_SEQUENCES_SET2 = [
    # Each test: list of photo IDs that should appear in order (or reverse order)
    # The test passes if they appear consecutively in that order within the same cluster
    {"photos": [887, 889], "note": "Living room photos should be consecutive"},
    {"photos": [890, 891], "note": "First bedroom photos should be consecutive"},
]

# -----------------------------------------------------------------------------
# TEST SET 3: Listing with photo IDs 805-815
# NOTE: Photo 806 not in DB - removed from tests
# -----------------------------------------------------------------------------
EXPECTED_SAME_CLUSTER_SET3 = [
    [805, 814],           # Front of house photos
]

EXPECTED_DIFFERENT_CLUSTERS_SET3 = [
    ([805, 814], [815]),  # 815 is back of house, should be separate
]

EXPECTED_ORDER_SET3 = [
    # No ordering tests - only 2 photos in cluster
]

# -----------------------------------------------------------------------------
# TEST SET 4: Listing with photo IDs 1633-1677 (user-reported inconsistencies)
# -----------------------------------------------------------------------------
EXPECTED_SAME_CLUSTER_SET4 = [
    [1659, 1661],                 # Same room with potential transition
    [1649, 1650, 1651],           # Known good kitchen cluster
    [1648, 1652, 1653],           # User reports these should also group together
    [1633, 1634, 1677],           # Front-yard sequence should stay together
]

EXPECTED_DIFFERENT_CLUSTERS_SET4 = [
    ([1649, 1650, 1651], [1673, 1675]),  # Kitchen vs basement
    ([1673], [1675]),                    # User confirmed this pair should stay separated
]

# -----------------------------------------------------------------------------
# TEST SET 5: Listing with photo IDs 1600-1606 (user-reported merge mistakes)
# -----------------------------------------------------------------------------
EXPECTED_SAME_CLUSTER_SET5 = [
    [1604, 1605],                 # 1604 should group with 1605
]

EXPECTED_DIFFERENT_CLUSTERS_SET5 = [
    ([1600], [1606]),             # Different bathrooms despite visual similarity
    ([1603], [1604, 1605]),       # 1603 should not be in 1604/1605 cluster
]

# -----------------------------------------------------------------------------
# TEST SET 6: Listing with photo IDs 1929-1988 (latest regression report)
# -----------------------------------------------------------------------------
EXPECTED_SAME_CLUSTER_SET6 = [
    [1929, 1988],                 # Exterior front pair with strong transition potential
    [1938, 1939],                 # Living/entrance adjacency should stay together
    [1960, 1961],                 # Bathroom/laundry confusion should be recovered
]

EXPECTED_DIFFERENT_CLUSTERS_SET6 = [
    ([1929, 1988], [1930]),       # Roof/aerial should not hijack front-exterior cluster
    ([1938, 1939], [1960, 1961]), # Different interior zones
]

# -----------------------------------------------------------------------------
# TEST SET 7: Listing with photo IDs 2094-2099 (patio transition quality)
# -----------------------------------------------------------------------------
EXPECTED_SAME_CLUSTER_SET7 = []
CONTEXT_PHOTOS_SET7 = [2096, 2097, 2098]  # Preserve real temporal spacing between 2094/2095 and 2099

EXPECTED_DIFFERENT_CLUSTERS_SET7 = [
    ([2099], [2094, 2095]),       # 2099 is opposite patio side and should not chain with 2094/2095
]

# -----------------------------------------------------------------------------
# TEST SET 8: Listing 2428-2432 (latest duplicate review)
# -----------------------------------------------------------------------------
EXPECTED_SAME_CLUSTER_SET8 = []
EXPECTED_DIFFERENT_CLUSTERS_SET8 = []
EXPECTED_DUPLICATE_PAIRS_SET8 = [
    (2428, 2431),
    (2429, 2432),
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


def check_order(expected_photos: list, expected_start: int, clusters: list) -> tuple:
    """Check if cluster starts (or ends) with expected endpoint photo.

    Returns (success, info)
    """
    for cluster in clusters:
        if expected_start in cluster:
            # Check if expected_start is at position 0 or -1 (either end is OK)
            if cluster[0] == expected_start:
                return True, f"Starts with {expected_start}: {cluster}"
            elif cluster[-1] == expected_start:
                return True, f"Ends with {expected_start}: {cluster}"
            else:
                pos = cluster.index(expected_start)
                return False, f"{expected_start} at position {pos}, not endpoint: {cluster}"

    return False, f"{expected_start} not found in any cluster"


def check_sequence(expected_sequence: list, clusters: list) -> tuple:
    """Check if photos appear consecutively in order (or reverse order) within a cluster.

    This validates that photos are ordered for smooth transitions.

    Returns (success, info)
    """
    # Find which cluster contains all the expected photos
    for cluster in clusters:
        if all(pid in cluster for pid in expected_sequence):
            # Found the cluster - check if photos are consecutive
            positions = [cluster.index(pid) for pid in expected_sequence]

            # Check if positions are consecutive (ascending or descending)
            is_ascending = all(positions[i] + 1 == positions[i+1] for i in range(len(positions)-1))
            is_descending = all(positions[i] - 1 == positions[i+1] for i in range(len(positions)-1))

            if is_ascending:
                return True, f"Consecutive (forward): {[cluster[p] for p in positions]}"
            elif is_descending:
                return True, f"Consecutive (reverse): {[cluster[p] for p in positions]}"
            else:
                # Not consecutive - show actual positions
                actual_order = [(pid, cluster.index(pid)) for pid in expected_sequence]
                return False, f"Not consecutive: {actual_order} in cluster {cluster}"

    # Photos are in different clusters
    photo_clusters = {}
    for pid in expected_sequence:
        for cidx, cluster in enumerate(clusters):
            if pid in cluster:
                photo_clusters[pid] = cidx
                break
        else:
            photo_clusters[pid] = None

    return False, f"Photos in different clusters: {photo_clusters}"


def run_test_set(
    db,
    s3_client,
    test_ids,
    same_cluster_tests,
    different_cluster_tests,
    order_tests=None,
    sequence_tests=None,
    duplicate_pairs=None,
    set_name="",
):
    """Run a single test set and return results."""

    # Find job with most matching photos
    jobs = db.query(Job).order_by(Job.id.desc()).limit(30).all()
    best_job = None
    best_match_count = 0

    for j in jobs:
        match_count = db.query(JobPhoto.id).filter(
            JobPhoto.job_id == j.id,
            JobPhoto.id.in_(test_ids)
        ).count()
        if match_count > best_match_count:
            best_match_count = match_count
            best_job = j

    if best_job is None or best_match_count < 2:
        print(f"\n  Skipping {set_name}: No matching photos found")
        return None

    job = best_job

    # Load photos that exist in this job
    photos = db.query(JobPhoto).filter(
        JobPhoto.job_id == job.id,
        JobPhoto.id.in_(test_ids)
    ).order_by(JobPhoto.id).all()

    found_ids = {p.id for p in photos}
    missing_from_job = sorted(set(test_ids) - found_ids)

    print(f"\n  Job {job.id}: {len(found_ids)}/{len(test_ids)} expected photos in DB")
    print(f"  Photo IDs in job: {sorted(found_ids)}")
    if missing_from_job:
        print(f"  ⚠ Photos NOT in this job: {missing_from_job}")

    # Download images
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
        except Exception as e:
            logger.warning(f"Failed to download photo {photo.id}: {e}")

    print(f"  Downloaded {len(images)} images")

    # Run clustering
    print("\n  Running clustering...")
    room_labels = [
        photo_map[pid].room_override or photo_map[pid].room_label or "unknown"
        for pid in photo_ids
    ]
    cluster_lists = cluster_photos_optimized(
        images,
        photo_ids,
        s3_client,
        db_session=db,
        job_id=job.id,
        room_labels=room_labels,
    )

    # Show clusters
    print("\n  CLUSTERS:")
    for i, cluster in enumerate(cluster_lists):
        labels = [f"{pid}({photo_map[pid].room_label})" if pid in photo_map else str(pid) for pid in cluster]
        print(f"    {i}: {' -> '.join(labels)}")

    # Run validations
    all_passed = True

    # Build photo->cluster lookup for debug output
    photo_to_cluster = {}
    for cluster_idx, cluster in enumerate(cluster_lists):
        for pid in cluster:
            photo_to_cluster[pid] = (cluster_idx, cluster)

    print("\n  SAME CLUSTER TESTS:")
    for group in same_cluster_tests:
        existing = [p for p in group if p in photo_ids]
        if len(existing) < 2:
            continue  # Skip if not enough photos available
        success, info = check_same_cluster(existing, cluster_lists)
        status = "✓" if success else "✗"
        print(f"    {status} {existing}: {info}")
        if not success:
            all_passed = False
            # Show where each photo actually ended up
            for pid in existing:
                if pid in photo_to_cluster:
                    cidx, c = photo_to_cluster[pid]
                    print(f"       └─ {pid} is in cluster {cidx}: {c}")
                else:
                    print(f"       └─ {pid} NOT IN ANY CLUSTER")

    print("\n  DIFFERENT CLUSTER TESTS:")
    for g1, g2 in different_cluster_tests:
        g1_exist = [p for p in g1 if p in photo_ids]
        g2_exist = [p for p in g2 if p in photo_ids]
        if not g1_exist or not g2_exist:
            continue  # Skip if missing photos
        success, info = check_different_clusters(g1_exist, g2_exist, cluster_lists)
        status = "✓" if success else "✗"
        print(f"    {status} {g1_exist} vs {g2_exist}: {info}")
        if not success:
            all_passed = False
            # Show the shared cluster contents
            g1_cidx = set(photo_to_cluster.get(pid, (-1, None))[0] for pid in g1_exist)
            g2_cidx = set(photo_to_cluster.get(pid, (-1, None))[0] for pid in g2_exist)
            shared = g1_cidx & g2_cidx - {-1}
            for cidx in shared:
                print(f"       └─ Shared cluster {cidx}: {cluster_lists[cidx]}")

    if order_tests:
        print("\n  ORDERING TESTS:")
        for test in order_tests:
            if test["expected_start"] not in photo_ids:
                continue
            success, info = check_order(test["photos"], test["expected_start"], cluster_lists)
            status = "✓" if success else "✗"
            print(f"    {status} {test['note']}: {info}")
            if not success:
                all_passed = False
                # Show the actual cluster containing this photo
                expected_start = test["expected_start"]
                if expected_start in photo_to_cluster:
                    cidx, cluster = photo_to_cluster[expected_start]
                    pos = cluster.index(expected_start) if expected_start in cluster else -1
                    print(f"       └─ {expected_start} at position {pos} in: {cluster}")

    if sequence_tests:
        print("\n  SEQUENCE TESTS (consecutive photo ordering):")
        for test in sequence_tests:
            # Check if all expected photos are in the test set
            if not all(pid in photo_ids for pid in test["photos"]):
                continue
            success, info = check_sequence(test["photos"], cluster_lists)
            status = "✓" if success else "✗"
            print(f"    {status} {test['note']}: {info}")
            if not success:
                all_passed = False

    if duplicate_pairs:
        print("\n  DUPLICATE PAIR TESTS:")
        photos_in_clusters = {pid for cluster in cluster_lists for pid in cluster}
        for a, b in duplicate_pairs:
            a_present = a in photos_in_clusters
            b_present = b in photos_in_clusters
            success = not (a_present and b_present)
            status = "✓" if success else "✗"
            present_text = f"{a}: {'in' if a_present else 'out'}, {b}: {'in' if b_present else 'out'}"
            print(f"    {status} [{a}, {b}] dedupe: {present_text}")
            if not success:
                all_passed = False

    return all_passed


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test clustering against expected photo groupings')
    parser.add_argument('--job-id', type=int, help='Job ID to test (default: auto-detect)')
    parser.add_argument('--test-set', type=str, choices=['1', '2', '3', '4', '5', '6', '7', '8', 'all'], default='all',
                       help='Test set: 1=original, 2=861-912, 3=805-815, 4=1633-1677, 5=1600-1606, 6=1929-1988, 7=2094-2099, 8=2428-2432 duplicates, all=run all')
    parser.add_argument('--list-jobs', action='store_true', help='List available jobs with photo counts')
    args = parser.parse_args()

    engine = create_engine(settings.DATABASE_URL)
    Session = sessionmaker(bind=engine)
    db = Session()
    s3_client = S3Client()

    if args.list_jobs:
        jobs = db.query(Job).order_by(Job.id.desc()).limit(15).all()
        print("\nRecent jobs:")
        for j in jobs:
            photo_count = db.query(JobPhoto).filter(JobPhoto.job_id == j.id).count()
            photo_range = db.query(JobPhoto.id).filter(JobPhoto.job_id == j.id).order_by(JobPhoto.id).all()
            if photo_range:
                min_id, max_id = photo_range[0][0], photo_range[-1][0]
                print(f"  Job {j.id}: {photo_count} photos (IDs {min_id}-{max_id})")
        db.close()
        return

    print("="*70)
    print("CLUSTERING TEST SUITE")
    print("="*70)

    results = []

    # Test Set 1: Original (569-705)
    if args.test_set in ['1', 'all']:
        print("\n" + "="*70)
        print("TEST SET 1: Original listing (569-705)")
        print("="*70)

        test_ids = set()
        for group in EXPECTED_SAME_CLUSTER:
            test_ids.update(group)
        for g1, g2 in EXPECTED_DIFFERENT_CLUSTERS:
            test_ids.update(g1)
            test_ids.update(g2)

        result = run_test_set(db, s3_client, test_ids,
                             EXPECTED_SAME_CLUSTER,
                             EXPECTED_DIFFERENT_CLUSTERS,
                             order_tests=None,
                             sequence_tests=None,
                             set_name="Set 1")
        if result is not None:
            results.append(("Set 1: Original", result))

    # Test Set 2: Listing 861-912
    if args.test_set in ['2', 'all']:
        print("\n" + "="*70)
        print("TEST SET 2: Listing 861-912")
        print("="*70)

        test_ids = set()
        for group in EXPECTED_SAME_CLUSTER_SET2:
            test_ids.update(group)
        for g1, g2 in EXPECTED_DIFFERENT_CLUSTERS_SET2:
            test_ids.update(g1)
            test_ids.update(g2)

        result = run_test_set(db, s3_client, test_ids,
                             EXPECTED_SAME_CLUSTER_SET2,
                             EXPECTED_DIFFERENT_CLUSTERS_SET2,
                             order_tests=EXPECTED_ORDER_SET2,
                             sequence_tests=EXPECTED_SEQUENCES_SET2,
                             set_name="Set 2")
        if result is not None:
            results.append(("Set 2: 861-912", result))

    # Test Set 3: Listing 805-815
    if args.test_set in ['3', 'all']:
        print("\n" + "="*70)
        print("TEST SET 3: Listing 805-815")
        print("="*70)

        test_ids = set()
        for group in EXPECTED_SAME_CLUSTER_SET3:
            test_ids.update(group)
        for g1, g2 in EXPECTED_DIFFERENT_CLUSTERS_SET3:
            test_ids.update(g1)
            test_ids.update(g2)

        result = run_test_set(db, s3_client, test_ids,
                             EXPECTED_SAME_CLUSTER_SET3,
                             EXPECTED_DIFFERENT_CLUSTERS_SET3,
                             order_tests=EXPECTED_ORDER_SET3,
                             sequence_tests=None,
                             set_name="Set 3")
        if result is not None:
            results.append(("Set 3: 805-815", result))

    # Test Set 4: Listing 1633-1677 (user inconsistencies)
    if args.test_set in ['4', 'all']:
        print("\n" + "="*70)
        print("TEST SET 4: Listing 1633-1677 (user inconsistencies)")
        print("="*70)

        test_ids = set()
        for group in EXPECTED_SAME_CLUSTER_SET4:
            test_ids.update(group)
        for g1, g2 in EXPECTED_DIFFERENT_CLUSTERS_SET4:
            test_ids.update(g1)
            test_ids.update(g2)

        result = run_test_set(db, s3_client, test_ids,
                             EXPECTED_SAME_CLUSTER_SET4,
                             EXPECTED_DIFFERENT_CLUSTERS_SET4,
                             order_tests=None,
                             sequence_tests=None,
                             set_name="Set 4")
        if result is not None:
            results.append(("Set 4: 1633-1677", result))

    # Test Set 5: Listing 1600-1606 (user inconsistencies)
    if args.test_set in ['5', 'all']:
        print("\n" + "="*70)
        print("TEST SET 5: Listing 1600-1606 (user inconsistencies)")
        print("="*70)

        test_ids = set()
        for group in EXPECTED_SAME_CLUSTER_SET5:
            test_ids.update(group)
        for g1, g2 in EXPECTED_DIFFERENT_CLUSTERS_SET5:
            test_ids.update(g1)
            test_ids.update(g2)

        result = run_test_set(db, s3_client, test_ids,
                             EXPECTED_SAME_CLUSTER_SET5,
                             EXPECTED_DIFFERENT_CLUSTERS_SET5,
                             order_tests=None,
                             sequence_tests=None,
                             set_name="Set 5")
        if result is not None:
            results.append(("Set 5: 1600-1606", result))

    # Test Set 6: Listing 1929-1988 (latest user regressions)
    if args.test_set in ['6', 'all']:
        print("\n" + "="*70)
        print("TEST SET 6: Listing 1929-1988 (latest regressions)")
        print("="*70)

        test_ids = set()
        for group in EXPECTED_SAME_CLUSTER_SET6:
            test_ids.update(group)
        for g1, g2 in EXPECTED_DIFFERENT_CLUSTERS_SET6:
            test_ids.update(g1)
            test_ids.update(g2)

        result = run_test_set(db, s3_client, test_ids,
                             EXPECTED_SAME_CLUSTER_SET6,
                             EXPECTED_DIFFERENT_CLUSTERS_SET6,
                             order_tests=None,
                             sequence_tests=None,
                             set_name="Set 6")
        if result is not None:
            results.append(("Set 6: 1929-1988", result))

    # Test Set 7: Listing 2094-2099 (patio transition quality)
    if args.test_set in ['7', 'all']:
        print("\n" + "="*70)
        print("TEST SET 7: Listing 2094-2099 (patio transition quality)")
        print("="*70)

        test_ids = set()
        test_ids.update(CONTEXT_PHOTOS_SET7)
        for group in EXPECTED_SAME_CLUSTER_SET7:
            test_ids.update(group)
        for g1, g2 in EXPECTED_DIFFERENT_CLUSTERS_SET7:
            test_ids.update(g1)
            test_ids.update(g2)

        result = run_test_set(db, s3_client, test_ids,
                             EXPECTED_SAME_CLUSTER_SET7,
                             EXPECTED_DIFFERENT_CLUSTERS_SET7,
                             order_tests=None,
                             sequence_tests=None,
                             set_name="Set 7")
        if result is not None:
            results.append(("Set 7: 2094-2099", result))

    # Test Set 8: Listing 2428-2432 (duplicate review)
    if args.test_set in ['8', 'all']:
        print("\n" + "="*70)
        print("TEST SET 8: Listing 2428-2432 (duplicate review)")
        print("="*70)

        test_ids = {2428, 2429, 2430, 2431, 2432}
        result = run_test_set(
            db,
            s3_client,
            test_ids,
            EXPECTED_SAME_CLUSTER_SET8,
            EXPECTED_DIFFERENT_CLUSTERS_SET8,
            order_tests=None,
            sequence_tests=None,
            duplicate_pairs=EXPECTED_DUPLICATE_PAIRS_SET8,
            set_name="Set 8",
        )
        if result is not None:
            results.append(("Set 8: 2428-2432", result))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    if results and all(r[1] for r in results):
        print("\nALL TESTS PASSED!")
    elif results:
        print("\nSOME TESTS FAILED - tuning needed")
    else:
        print("\nNo tests ran - check photo IDs")

    db.close()


if __name__ == "__main__":
    main()
