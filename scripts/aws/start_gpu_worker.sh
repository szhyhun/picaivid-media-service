#!/bin/bash
# Start GPU worker container
#
# Required environment variables:
#   DATABASE_URL - PostgreSQL connection string
#   S3_BUCKET - S3 bucket name
#   SQS_QUEUE_URL - SQS queue URL
#   ECR_REGISTRY - ECR registry URL (e.g., 123456.dkr.ecr.us-west-2.amazonaws.com)
#
# Optional:
#   AWS_REGION - defaults to us-west-2
#   IMAGE_TAG - defaults to gpu-latest

set -e

# Defaults
AWS_REGION="${AWS_REGION:-us-west-2}"
IMAGE_TAG="${IMAGE_TAG:-gpu-latest}"
CONTAINER_NAME="picaivid-gpu-worker"

# Check required variables
if [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DATABASE_URL not set"
    exit 1
fi

if [ -z "$S3_BUCKET" ]; then
    echo "ERROR: S3_BUCKET not set"
    exit 1
fi

if [ -z "$SQS_QUEUE_URL" ]; then
    echo "ERROR: SQS_QUEUE_URL not set"
    exit 1
fi

if [ -z "$ECR_REGISTRY" ]; then
    echo "ERROR: ECR_REGISTRY not set"
    exit 1
fi

IMAGE="${ECR_REGISTRY}/picaivid-media:${IMAGE_TAG}"

echo "=== Starting GPU Worker ==="
echo "Image: $IMAGE"
echo "Container: $CONTAINER_NAME"

# Stop existing container if running
if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
    echo "Stopping existing container..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin $ECR_REGISTRY

# Pull latest image
echo "Pulling latest image..."
docker pull $IMAGE

# Start container
echo "Starting container..."
docker run -d \
    --gpus all \
    --name $CONTAINER_NAME \
    --restart unless-stopped \
    -e WORKER_TYPE=gpu \
    -e DATABASE_URL="$DATABASE_URL" \
    -e AWS_REGION="$AWS_REGION" \
    -e S3_BUCKET="$S3_BUCKET" \
    -e SQS_QUEUE_URL="$SQS_QUEUE_URL" \
    -e HF_HUB_OFFLINE=1 \
    -e MODEL_CACHE_DIR=/app/ml_models \
    $IMAGE

echo ""
echo "=== Worker Started ==="
echo ""
echo "View logs: docker logs -f $CONTAINER_NAME"
echo "Stop worker: ./scripts/aws/stop_gpu_worker.sh"
