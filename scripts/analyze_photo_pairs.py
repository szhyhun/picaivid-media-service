#!/usr/bin/env python3
"""Analyze specific photo pairs to tune clustering thresholds.

Tests both semantic similarity and geometric overlap for given photo pairs.
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
from app.pipeline.phase1_analyze.learned_matching import (
    compute_dinov2_embeddings,
    match_image_pair,
    MIN_INLIERS_FOR_OVERLAP,
    OVERLAP_THRESHOLD,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Photo pairs to analyze (should be same cluster)
SHOULD_MATCH = [
    (569, 570),   # Good pair in cluster #173
    (581, 582),   # Should be same cluster
    (595, 596),   # Should be same cluster (living/dining overlap)
    (596, 597),   # Should be same cluster
    (595, 597),   # Should be same cluster
    (601, 602),   # Should be same cluster
    (605, 606),   # Should be same cluster
]

# Photo pairs that should NOT match
SHOULD_NOT_MATCH = [
    (570, 572),   # Different view in cluster #173
    (572, 573),   # These should be separate from 569-570
]


def main():
    engine = create_engine(settings.DATABASE_URL)
    Session = sessionmaker(bind=engine)
    db = Session()

    # Get latest job
    job = db.query(Job).order_by(Job.id.desc()).first()
    if not job:
        print("No jobs found")
        return

    print(f"\nAnalyzing photos from Job {job.id}")
    print("=" * 70)

    # Collect all photo IDs we need
    all_photo_ids = set()
    for a, b in SHOULD_MATCH + SHOULD_NOT_MATCH:
        all_photo_ids.add(a)
        all_photo_ids.add(b)

    # Load photos
    photos = db.query(JobPhoto).filter(
        JobPhoto.job_id == job.id,
        JobPhoto.id.in_(all_photo_ids)
    ).all()

    photo_map = {p.id: p for p in photos}
    print(f"Loaded {len(photos)} photos")

    # Download images
    s3_client = S3Client()
    images = {}

    for photo in photos:
        try:
            img = s3_client.download_image(photo.s3_uri)
            img = img.resize((512, 384), Image.Resampling.LANCZOS)
            images[photo.id] = img
            print(f"  Downloaded photo {photo.id}: {photo.room_label}")
        except Exception as e:
            logger.warning(f"Failed to download photo {photo.id}: {e}")

    print()

    # Compute DINOv2 embeddings for semantic similarity
    print("Computing DINOv2 embeddings...")
    photo_ids_list = list(images.keys())
    images_list = [images[pid] for pid in photo_ids_list]

    from sklearn.preprocessing import normalize
    embeddings = compute_dinov2_embeddings(images_list)
    embeddings = normalize(embeddings)

    # Create embedding lookup
    embedding_map = {pid: embeddings[i] for i, pid in enumerate(photo_ids_list)}

    print()
    print("=" * 70)
    print("PAIRS THAT SHOULD MATCH:")
    print("=" * 70)

    for a, b in SHOULD_MATCH:
        if a not in images or b not in images:
            print(f"  {a} <-> {b}: SKIPPED (photo not found)")
            continue

        # Semantic similarity
        sim = (embedding_map[a] @ embedding_map[b].T).item()

        # Geometric match
        num_matches, num_inliers, score, _direction = match_image_pair(images[a], images[b])

        # Status
        passes_inliers = num_inliers >= MIN_INLIERS_FOR_OVERLAP
        passes_score = score > OVERLAP_THRESHOLD

        status = "✓ PASS" if (passes_inliers and passes_score) else "✗ FAIL"

        print(f"\n  {a} <-> {b}: {status}")
        print(f"    Room labels: {photo_map[a].room_label} <-> {photo_map[b].room_label}")
        print(f"    Semantic similarity: {sim:.3f}")
        print(f"    Geometric: {num_matches} matches, {num_inliers} inliers, score={score:.3f}")
        print(f"    Thresholds: inliers>={MIN_INLIERS_FOR_OVERLAP} ({passes_inliers}), score>{OVERLAP_THRESHOLD} ({passes_score})")

    print()
    print("=" * 70)
    print("PAIRS THAT SHOULD NOT MATCH:")
    print("=" * 70)

    for a, b in SHOULD_NOT_MATCH:
        if a not in images or b not in images:
            print(f"  {a} <-> {b}: SKIPPED (photo not found)")
            continue

        # Semantic similarity
        sim = (embedding_map[a] @ embedding_map[b].T).item()

        # Geometric match
        num_matches, num_inliers, score, _direction = match_image_pair(images[a], images[b])

        # Status (should NOT pass)
        passes_inliers = num_inliers >= MIN_INLIERS_FOR_OVERLAP
        passes_score = score > OVERLAP_THRESHOLD

        status = "✓ CORRECT (no match)" if not (passes_inliers and passes_score) else "✗ FALSE POSITIVE"

        print(f"\n  {a} <-> {b}: {status}")
        print(f"    Room labels: {photo_map[a].room_label} <-> {photo_map[b].room_label}")
        print(f"    Semantic similarity: {sim:.3f}")
        print(f"    Geometric: {num_matches} matches, {num_inliers} inliers, score={score:.3f}")

    print()
    print("=" * 70)
    print("TEMPORAL ADJACENCY ANALYSIS:")
    print("=" * 70)

    # Get all photos in order by ID (assuming ID order ≈ upload order)
    all_photos = db.query(JobPhoto).filter(
        JobPhoto.job_id == job.id,
        JobPhoto.exclude == False
    ).order_by(JobPhoto.id).all()

    print(f"\nPhotos in ID order (first 20):")
    for i, p in enumerate(all_photos[:20]):
        print(f"  {i}: ID={p.id}, room={p.room_label}")

    # Check if our "should match" pairs are temporally adjacent
    print(f"\nTemporal distance for SHOULD_MATCH pairs:")
    id_to_idx = {p.id: i for i, p in enumerate(all_photos)}

    for a, b in SHOULD_MATCH:
        if a in id_to_idx and b in id_to_idx:
            dist = abs(id_to_idx[a] - id_to_idx[b])
            print(f"  {a} <-> {b}: distance = {dist} (adjacent={dist <= 2})")

    db.close()


if __name__ == "__main__":
    main()
