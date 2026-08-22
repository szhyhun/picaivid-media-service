#!/usr/bin/env python3
"""Pre-download all ML models for production deployment.

This script downloads all required models to the local cache directory.
Run this during Docker build or AMI creation to avoid runtime downloads.

Usage:
    python scripts/download_models.py [--cache-dir ./ml_models]

For Docker:
    RUN python scripts/download_models.py

For S3 backup (after downloading):
    aws s3 sync ./ml_models s3://your-bucket/ml-models/
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add app to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_openclip_models(cache_dir: Path) -> None:
    """Download OpenCLIP models for embeddings."""
    import open_clip

    openclip_cache = cache_dir / "openclip"
    openclip_cache.mkdir(parents=True, exist_ok=True)

    # Set cache directory
    os.environ["OPEN_CLIP_CACHE"] = str(openclip_cache)

    model_name = "ViT-B-32"
    pretrained = "openai"

    logger.info(f"Downloading OpenCLIP {model_name}/{pretrained}...")

    # This downloads and caches the model
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        cache_dir=str(openclip_cache),
    )

    logger.info(f"OpenCLIP model downloaded")


def download_ltx2_models(cache_dir: Path) -> None:
    """Download LTX-2 video generation model (~15GB)."""
    import torch
    from diffusers import LTX2ImageToVideoPipeline

    hf_cache = cache_dir / "huggingface"
    hf_cache.mkdir(parents=True, exist_ok=True)

    model_id = "Lightricks/LTX-2"
    logger.info(f"Downloading {model_id} (this may take a while, ~15GB)...")

    # Download the pipeline (includes VAE, transformer, scheduler)
    pipe = LTX2ImageToVideoPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        cache_dir=str(hf_cache),
    )

    logger.info(f"LTX-2 model downloaded successfully")
    del pipe  # Free memory


def verify_models(cache_dir: Path) -> bool:
    """Verify all models are properly cached."""
    openclip_cache = cache_dir / "openclip"
    if not openclip_cache.exists():
        logger.error("OpenCLIP cache directory not found")
        return False
    model_files = [path for path in openclip_cache.rglob("*") if path.is_file()]
    if not model_files:
        logger.error("OpenCLIP model cache is empty")
        return False
    logger.info("Found %s OpenCLIP cache files", len(model_files))
    logger.info("Model verification passed")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download ML models for production")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./ml_models",
        help="Directory to cache models (default: ./ml_models)"
    )
    parser.add_argument(
        "--skip-openclip",
        action="store_true",
        help="Skip OpenCLIP model download"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing models, don't download"
    )
    parser.add_argument(
        "--include-ltx2",
        action="store_true",
        help="Also download LTX-2 video generation model (~15GB, requires CUDA)"
    )

    args = parser.parse_args()
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model cache directory: {cache_dir}")

    if args.verify_only:
        success = verify_models(cache_dir)
        sys.exit(0 if success else 1)

    # Download all models
    logger.info("=" * 50)
    logger.info("Downloading all ML models...")
    logger.info("=" * 50)

    if not args.skip_openclip:
        try:
            download_openclip_models(cache_dir)
        except Exception as e:
            logger.error(f"Failed to download OpenCLIP models: {e}")
            sys.exit(1)

    if args.include_ltx2:
        try:
            download_ltx2_models(cache_dir)
        except Exception as e:
            logger.error(f"Failed to download LTX-2 models: {e}")
            sys.exit(1)

    # Verify
    logger.info("=" * 50)
    logger.info("Verifying downloaded models...")
    logger.info("=" * 50)

    if not verify_models(cache_dir):
        sys.exit(1)

    # Print cache size
    total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
    size_mb = total_size / (1024 * 1024)
    logger.info(f"Total cache size: {size_mb:.1f} MB")

    logger.info("=" * 50)
    logger.info("All models downloaded successfully!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
