#!/bin/bash
# Sync ML models to/from S3 for AWS deployment
#
# Usage:
#   Upload to S3:   ./scripts/sync_models_s3.sh upload
#   Download from S3: ./scripts/sync_models_s3.sh download
#
# Environment variables:
#   S3_MODEL_BUCKET - S3 bucket for model storage (default: picaivid-models)
#   MODEL_CACHE_DIR - Local cache directory (default: ./ml_models)

set -e

S3_BUCKET="${S3_MODEL_BUCKET:-picaivid-models}"
S3_PREFIX="ml-models/v1"
LOCAL_DIR="${MODEL_CACHE_DIR:-./ml_models}"

case "$1" in
    upload)
        echo "Uploading models to s3://${S3_BUCKET}/${S3_PREFIX}/"
        aws s3 sync "${LOCAL_DIR}" "s3://${S3_BUCKET}/${S3_PREFIX}/" \
            --exclude "*.pyc" \
            --exclude "__pycache__/*"
        echo "Upload complete!"
        ;;
    download)
        echo "Downloading models from s3://${S3_BUCKET}/${S3_PREFIX}/"
        mkdir -p "${LOCAL_DIR}"
        aws s3 sync "s3://${S3_BUCKET}/${S3_PREFIX}/" "${LOCAL_DIR}"
        echo "Download complete!"
        ;;
    *)
        echo "Usage: $0 {upload|download}"
        echo ""
        echo "Environment variables:"
        echo "  S3_MODEL_BUCKET - S3 bucket name (default: picaivid-models)"
        echo "  MODEL_CACHE_DIR - Local directory (default: ./ml_models)"
        exit 1
        ;;
esac
