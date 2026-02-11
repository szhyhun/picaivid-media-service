#!/usr/bin/env python
"""Test script to debug and tune AI models.

Usage:
    source venv/bin/activate
    python test_models.py
"""
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_s3_connection():
    """Test S3/MinIO connection and list files."""
    print("\n=== Testing S3/MinIO Connection ===")
    from app.services.storage.s3_client import s3_client
    from app.core.config import settings

    print(f"S3 Endpoint: {settings.S3_ENDPOINT}")
    print(f"S3 Bucket: {settings.S3_BUCKET}")

    try:
        # List objects in bucket
        response = s3_client.client.list_objects_v2(
            Bucket=settings.S3_BUCKET,
            MaxKeys=10
        )

        if 'Contents' in response:
            print(f"\nFound {len(response['Contents'])} objects:")
            for obj in response['Contents']:
                print(f"  - {obj['Key']} ({obj['Size']} bytes)")
            return response['Contents'][0]['Key'] if response['Contents'] else None
        else:
            print("No objects found in bucket")
            return None
    except Exception as e:
        print(f"Error connecting to S3: {e}")
        return None


def test_download_image(key: str):
    """Test downloading an image from S3."""
    print(f"\n=== Testing Image Download: {key} ===")
    from app.services.storage.s3_client import s3_client

    try:
        # Try with raw key (correct way)
        image = s3_client.download_image(key)
        print(f"Success! Image size: {image.size}, mode: {image.mode}")
        return image
    except Exception as e:
        print(f"Error downloading image: {e}")

        # Try with s3:// prefix (the buggy way)
        try:
            s3_uri = f"s3://{key}"
            print(f"Trying s3_uri: {s3_uri}")
            image = s3_client.download_image(s3_uri)
            print(f"Success with s3:// prefix! This shows the _parse_key bug.")
            return image
        except Exception as e2:
            print(f"Also failed with s3:// prefix: {e2}")
            return None


def test_openclip(image):
    """Test OpenCLIP model."""
    print("\n=== Testing OpenCLIP Model ===")
    from app.models.openclip import openclip_model

    try:
        # Test embedding
        print("Computing embedding...")
        embedding = openclip_model.get_embedding(image)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding sample: {embedding[:5]}")

        # Test room classification
        print("\nClassifying room...")
        room_type, confidence = openclip_model.classify_room(image)
        print(f"Room type: {room_type}")
        print(f"Confidence: {confidence:.3f}")

        return True
    except Exception as e:
        print(f"Error with OpenCLIP: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_midas(image):
    """Test MiDaS depth model."""
    print("\n=== Testing MiDaS Model ===")
    from app.models.midas import midas_model

    try:
        print("Analyzing depth...")
        depth_metrics = midas_model.analyze_depth(image)

        print(f"Depth variance: {depth_metrics['variance']:.4f}")
        print(f"Depth std: {depth_metrics['std']:.4f}")
        print(f"Depth layers: {depth_metrics['depth_layers']}")
        print(f"Depth range: {depth_metrics['depth_range']:.2f}")
        print(f"Min depth: {depth_metrics['min_depth']:.2f}")
        print(f"Max depth: {depth_metrics['max_depth']:.2f}")

        tier = midas_model.get_confidence_tier(depth_metrics)
        print(f"Confidence tier: {tier}")

        return True
    except Exception as e:
        print(f"Error with MiDaS: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rails_photo_reader():
    """Test reading photos from Rails DB."""
    print("\n=== Testing Rails Photo Reader ===")
    from app.services.rails.photo_reader import rails_photo_reader

    try:
        # Get project IDs from jobs table or just list some photos
        from app.db.session import get_db_context
        from app.db.models import Job

        with get_db_context() as db:
            job = db.query(Job).first()
            if job:
                print(f"Found job: {job.id} for project: {job.project_id}")
                photos = rails_photo_reader.get_photos_by_project_id(job.project_id)
                print(f"Found {len(photos)} photos:")
                for p in photos[:3]:
                    print(f"  - {p.id}: {p.filename} -> {p.s3_object_key}")
                return photos
            else:
                print("No jobs found in database")
                return []
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    print("=" * 60)
    print("AI Model Testing Script")
    print("=" * 60)

    # Test S3 connection
    first_key = test_s3_connection()

    if first_key:
        # Download test image
        image = test_download_image(first_key)

        if image:
            # Test models
            test_openclip(image)
            test_midas(image)

    # Test Rails photo reader
    test_rails_photo_reader()

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
